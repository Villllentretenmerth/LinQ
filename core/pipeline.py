"""Конвейер одной камеры: кадр → yolo-pose (+ yolo-det) → ByteTrack → движок карточек → журнал.

Модели общие для всех камер (сессии ONNX Runtime потокобезопасны), у каждой камеры — свой поток,
свои треки, зоны и включённые карточки.
"""
import logging
import threading
import time
from pathlib import Path

import numpy as np

from .features import floor_homography
from .engine import CompileEnv, FrameContext, Obj, build_engine, load_cards
from .geometry import Scene
from .infer import COCO80, DetModel, PoseModel, make_backend
from .ingest import Frame, open_source
from .render import ORANGE, RED, draw_evidence, to_jpeg
from .tracking import PersonTracker, VehicleTracker

log = logging.getLogger("core.pipeline")


class Models:
    """Модели узла. Время инференса каждой уходит в метрики (rep.timer) под именами yolo-pose и yolo-det."""

    def __init__(self, cfg: dict, root: Path):
        m = cfg.get("models") or {}
        backend = m.get("backend", "onnx")
        linq = dict(m.get("linq") or {})
        threads = m.get("threads") or None

        def resolve(p) -> Path | None:
            if not p:
                return None
            p = Path(p)
            return p if p.is_absolute() or p.exists() else root / p

        # облако LinQ (module: bench.linq_cloud:make_infer): программа лежит на сервере, путь передаём как есть
        remote = backend == "linq" and bool(linq.get("module"))

        def model_path(key: str) -> Path | str | None:
            if backend == "linq":                       # программа TPU от организаторов
                p = linq.get(f"{key}_program")
                return p if remote else resolve(p)
            p = resolve(m.get(key))
            if p is not None and backend == "onnx_head":
                p = p.with_name(p.stem + "_head.onnx")  # python -m core.tools.split_onnx <модель>.onnx
            return p

        linq_opts = {k: v for k, v in linq.items() if not k.endswith("_program") and k != "det_class_names"}
        def available(p) -> bool:
            return p is not None and (remote or p.exists())

        pose_path = model_path("pose")
        if not available(pose_path):
            raise FileNotFoundError(f"Нет модели позы {pose_path}. Выгрузите её: python core/export_models.py")
        self.pose = PoseModel(make_backend(backend, str(pose_path), True, linq_opts, threads),
                              conf=m.get("pose_conf", 0.25))

        self.vehicle_classes = set(m.get("vehicle_classes") or ["car", "truck"])
        self.phone_classes = set(m.get("phone_classes") or ["cell phone"])
        self.det = None
        det_path = model_path("det")
        if available(det_path):
            opts = {**linq_opts, "class_names": linq.get("det_class_names") or COCO80}
            self.det = DetModel(make_backend(backend, str(det_path), False, opts, threads),
                                wanted=self.vehicle_classes | self.phone_classes, conf=m.get("det_conf", 0.35),
                                class_conf=m.get("class_conf") or {})
        else:
            log.warning("детектор объектов не найден (%s): карточки с техникой и телефоном будут выключены", det_path)
        self.det_every = max(1, int(m.get("det_every", 1)))
        self.backend = backend


class CameraPipeline:
    def __init__(self, cam: dict, cfg: dict, models: Models, cards_raw: dict, sink, rep, offline: bool,
                 stop: threading.Event, root: Path, max_seconds: float | None = None):
        self.cam, self.cfg, self.models, self.sink, self.rep = cam, cfg, models, sink, rep
        self.id = cam["id"]
        self.name = cam.get("name", self.id)
        self.area = cam.get("area", "—")
        self.fps = float(cam.get("fps", 10))
        self.offline, self.stop, self.max_seconds = offline, stop, max_seconds
        self.tz = cfg.get("plant_tz", "Europe/Moscow")
        src = str(cam["source"])
        self.source = src if "://" in src or src.isdigit() or Path(src).exists() else str(root / src)

        self.scene = Scene.from_config(cam.get("zones"), cam.get("lines"))
        env = CompileEnv(self.scene, models=set(),
                         det_classes=models.det.classes if models.det else set(),
                         vehicle_classes=models.vehicle_classes, phone_classes=models.phone_classes)
        self.engine, skipped = build_engine(cam, cards_raw, env, self.tz, float(cfg.get("end_after_sec", 10)))
        for pid, reason in skipped:
            log.warning("[%s] карточка %s выключена: %s", self.id, pid, reason)
        for c in self.engine.cards:
            fu = f"; затем {', '.join(f.label for f in c.followups)}" if c.followups else ""
            log.info("[%s] карточка %s включена: %s%s", self.id, c.id, c.cond.label, fu)
        self.need_det = "det" in self.engine.needs and models.det is not None

        keep = max([15.0] + [f.seconds + 5 for c in self.engine.cards for f in c.followups])
        lost = float(cfg.get("lost_sec", 2.0))
        self.persons = PersonTracker(self.fps, lost, keep, floor=floor_homography(cam.get("floor")))
        self.vehicles = VehicleTracker(self.fps, lost) if self.need_det else None
        self._last_objs: tuple[list, list] = ([], [])
        self.n = 0
        self.events = 0
        self.t_start = None
        self._first_ts: float | None = None

    # ---------- цикл ----------

    def run(self) -> None:
        if self.sink:
            self.sink.register_camera(self.id, self.name, self.area, str(self.cam["source"]))
        frames, reader = open_source(self.source, self.fps, self.offline, bool(self.cam.get("loop")))
        log.info("[%s] старт: %s, %g FPS обработки, режим %s", self.id, self.source, self.fps,
                 "офлайн (покадрово)" if self.offline else "реального времени")
        last: Frame | None = None
        self.t_start = time.time()
        try:
            for fr in frames:
                if self.stop.is_set():
                    break
                self.process(fr)
                last = fr
                if self.max_seconds and fr.ts - (self._first_ts or fr.ts) >= self.max_seconds:
                    break
        except Exception:
            log.exception("[%s] конвейер упал", self.id)
        finally:
            if last is not None:                     # закрыть всё открытое в конце видео
                for a in self.engine.close(last.ts):
                    self.emit(a, last, None)
            if reader is not None:
                reader.close()
            wall = time.time() - self.t_start
            log.info("[%s] стоп: %d кадров за %.1f с (%.1f кадр/с), событий: %d", self.id, self.n, wall,
                     self.n / max(wall, 1e-6), self.events)

    def process(self, fr: Frame) -> None:
        if self._first_ts is None:
            self._first_ts = fr.ts
            if self.persons.floor is not None:
                self.persons.floor.set_frame(fr.image.shape[1], fr.image.shape[0])
        timer = self.rep.timer if self.rep else None
        pose = self.models.pose(fr.image, timer)
        persons, lost = self.persons.update(pose, fr.ts)
        if self.need_det:
            if self.n % self.models.det_every == 0:
                det = self.models.det(fr.image, timer)
                veh = self.vehicles.update(det.subset(self.models.vehicle_classes), fr.ts)
                ph = det.subset(self.models.phone_classes)
                phones = [Obj(b, float(c), n) for b, c, n in zip(ph.xyxy, ph.conf, ph.names)]
                self._last_objs = (veh, phones)
            vehicles, phones = self._last_objs
        else:
            vehicles, phones = [], []
        ctx = FrameContext(fr.ts, persons, self.scene, vehicles, phones)
        actions = [a for tid in lost for a in self.engine.track_lost(tid, fr.ts)]
        actions += self.engine.step(ctx)
        for a in actions:
            self.emit(a, fr, ctx)
        self.n += 1
        if self.rep:
            self.rep.frame(self.id)
            if fr.skipped:
                self.rep.dropped(fr.skipped)

    # ---------- в журнал ----------

    def emit(self, a, fr: Frame, ctx: FrameContext | None) -> None:
        offset = fr.recv_ts - fr.ts          # офлайн: время видео -> время на часах узла
        card = a.card
        if a.kind == "create":
            self.events += 1
            now = time.time()
            label = f"{card.name} · трек {a.track_id} · {a.confidence:.2f}"
            img = draw_evidence(fr.image, self.scene, a.bbox, label, RED if card.severity == "critical" else ORANGE,
                                a.other_bbox, a.other_label, [p.xyxy for p in ctx.persons] if ctx else None,
                                fr.recv_ts, self.name, self.tz)
            snap = None
            if self.sink:
                snap = self.sink.evidence_dir / f"{a.key.replace(':', '_')}_{int(fr.recv_ts)}.jpg"
                snap.write_bytes(to_jpeg(img))
            extra = {**a.extra, "source": "pipeline", "backend": self.models.backend,
                     "time_mode": "offline" if self.offline else "realtime",
                     "frame_size": [int(fr.image.shape[1]), int(fr.image.shape[0])]}
            if a.other_label:
                extra["other"] = {"label": a.other_label, "bbox": a.other_bbox}
            payload = dict(pattern_id=card.id, pattern_name=card.name, categories=card.categories,
                           severity=card.severity, camera_id=self.id, camera_name=self.name, area=self.area,
                           zone=a.zone, track_id=int(a.track_id), started_at=a.started_ts + offset, confirmed_at=now,
                           frame_ts=fr.recv_ts, confidence=float(np.clip(a.confidence, 0, 1)), bbox=a.bbox, extra=extra)
            log.warning("[%s] НАРУШЕНИЕ %s (%s): трек %d, зона %s, уверенность %.2f, время до тревоги %.1f с",
                        self.id, card.id, card.severity, a.track_id, a.zone or "—", a.confidence,
                        now - (a.started_ts + offset))
            if self.sink:
                self.sink.create(a.key, payload, snap)
        elif a.kind == "update":
            log.warning("[%s] %s, трек %d: %s", self.id, card.id, a.track_id, a.extra.get("followup", {}).get("condition"))
            if self.sink:
                self.sink.update(a.key, a.extra)
        elif a.kind == "end":
            log.info("[%s] %s, трек %d: событие закрыто", self.id, card.id, a.track_id)
            if self.sink:
                self.sink.end(a.key, a.ended_ts + offset)


def load_patterns(cfg: dict, root: Path) -> dict:
    p = Path(cfg.get("patterns_dir", "patterns"))
    return load_cards(p if p.is_absolute() or p.exists() else root / p)
