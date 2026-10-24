"""Прогон конвейера по размеченным роликам для оценки качества и демо.

Каждый ролик — отдельная «камера» со своими зонами и карточками (clips.yaml), режим офлайн (каждый кадр
с шагом 1/fps, результат не зависит от скорости машины). Время событий — секунды от начала ролика,
поэтому их можно сравнить с разметкой labels.csv:

    python -m core.tools.eval_clips --clips clips/clips.yaml --out out/eval [--video] [--debug]
    python -m core.tools.evaluate --labels clips/labels.csv --events out/eval/events.jsonl --report out/eval/quality.md

clips.yaml:
    defaults:
      fps: 10
      patterns: [fall, run, carry_wrong]          # карточки для всех роликов
      zones: [{name: "Весь зал", tags: [tech_area], polygon: full}]
    clips:
      - file: v02.mp4
        add_patterns: [danger_zone]               # опасная зона — только на роликах про зону
        zones: [{name: "Опасная зона", tags: [rack_forklift], polygon: [[640,385],[910,497],[605,717],[350,480]]}]
"""
import argparse
import csv
import json
import logging
import pickle
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..pipeline import CameraPipeline, Models, load_patterns
from ..render import _texts

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("core.eval_clips")

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
            (14, 16), (0, 5), (0, 6)]
NAMES_RU = {"fall": "ПАДЕНИЕ", "run": "БЕГ", "danger_zone": "ОПАСНАЯ ЗОНА", "carry_wrong": "ПОДЪЁМ СПИНОЙ",
            "crawl_under": "ПРОЛЕЗАНИЕ", "phone": "ТЕЛЕФОН", "vehicle_proximity": "РЯДОМ С ТЕХНИКОЙ"}
RED, GREEN, ZONE, GRAY = (40, 40, 230), (80, 200, 80), (0, 200, 255), (200, 200, 200)


class CachedModel:
    """Обёртка модели: первый прогон пишет выходы по кадрам, повторный — читает (подбор порогов без инференса)."""

    def __init__(self, model, path: Path):
        self.model, self.path = model, path
        self.cache = pickle.load(open(path, "rb")) if path.exists() else None
        self.out, self.i = [], 0
        for k in ("classes",):
            if hasattr(model, k):
                setattr(self, k, getattr(model, k))

    def __call__(self, image, timer=None):
        if self.cache is not None:
            r = self.cache[self.i]
            self.i += 1
            return r
        r = self.model(image, timer)
        self.out.append(r)
        return r

    def save(self):
        if self.cache is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(self.out, open(self.path, "wb"))


class EvalPipeline(CameraPipeline):
    """Тот же конвейер; дополнительно пишет события во времени ролика и (по желанию) видео с разметкой."""

    def __init__(self, *a, clip: str, video_out: Path | None = None, debug: bool = False, feats=None, **kw):
        super().__init__(*a, **kw)
        self.feats = feats
        self.clip, self.records, self.debug = clip, [], debug
        self.active: dict[int, dict] = {}       # track_id -> {card_id: (name, stage)}
        self.banner: list[tuple[float, str]] = []
        self.video_out, self._ff = video_out, None

    def t(self, ts: float) -> float:
        return round(ts - self._first_ts, 2)

    def emit(self, a, fr, ctx):
        rec = {"file": self.clip, "kind": a.kind, "pattern_id": a.card.id, "track_id": int(a.track_id),
               "confirm": self.t(a.ts)}
        if a.kind == "create":
            rec["start"] = self.t(a.started_ts or a.ts)
            rec["confidence"] = round(float(a.confidence), 3)
            rec["zone"] = a.zone
            rec["features"] = {k: (round(v, 3) if isinstance(v, float) else v)
                               for k, v in (a.extra.get("features") or {}).items()}
            self.active.setdefault(int(a.track_id), {})[a.card.id] = "detected"
            self.banner.append((a.ts, f"{NAMES_RU.get(a.card.id, a.card.id)} · трек {a.track_id} · "
                                      f"{self.t(a.ts):.1f} с"))
        elif a.kind == "update":
            rec["stage"] = a.extra.get("stage")
            self.active.setdefault(int(a.track_id), {})[a.card.id] = a.extra.get("stage") or "update"
            if a.extra.get("stage") == "motionless":
                self.banner.append((a.ts, f"{NAMES_RU.get(a.card.id, a.card.id)}: лежит без движения · "
                                          f"трек {a.track_id}"))
        elif a.kind == "end":
            self.active.get(int(a.track_id), {}).pop(a.card.id, None)
        self.records.append(rec)
        super().emit(a, fr, ctx)

    def process(self, fr):
        super().process(fr)
        if self.feats is not None:
            for p in self._persons:
                f = p.hist.features()
                self.feats.writerow({"file": self.clip, "t": self.t(fr.ts), "track": p.track_id,
                                     **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()}})
        if self.video_out is not None:
            self._draw(fr)

    # ---------- видео с разметкой ----------

    def _draw(self, fr):
        img = fr.image.copy()
        h, w = img.shape[:2]
        if self._ff is None:
            self.video_out.parent.mkdir(parents=True, exist_ok=True)
            self._ff = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                 "-r", f"{self.fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.video_out)], stdin=subprocess.PIPE)
        ov = img.copy()
        for z in self.scene.zones:
            if "tech_area" in z.tags:
                continue                                    # «весь зал» не рисуем
            cv2.fillPoly(ov, [z.polygon.astype(np.int32)], ZONE)
        img = cv2.addWeighted(ov, 0.22, img, 0.78, 0)
        for z in self.scene.zones:
            if "tech_area" not in z.tags:
                cv2.polylines(img, [z.polygon.astype(np.int32)], True, ZONE, 2, cv2.LINE_AA)
        items = []
        for p in self._persons:
            s = p.hist.last
            alert = self.active.get(p.track_id)
            col = RED if alert else GREEN
            if s.kpts is not None:
                k = s.kpts
                for i, j in SKELETON:
                    if k[i, 2] > 0.3 and k[j, 2] > 0.3:
                        cv2.line(img, tuple(map(int, k[i, :2])), tuple(map(int, k[j, :2])), col, 2, cv2.LINE_AA)
                for i in range(5, 17):
                    if k[i, 2] > 0.3:
                        cv2.circle(img, tuple(map(int, k[i, :2])), 3, col, -1, cv2.LINE_AA)
            x1, y1, x2, y2 = map(int, p.xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3 if alert else 1)
            label = f"#{p.track_id}"
            if alert:
                label += " " + ", ".join(NAMES_RU.get(c, c) for c in alert)
            if self.debug:
                f = p.hist.features()
                fmt = lambda v: "—" if v is None else f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"
                label += (f"  v={fmt(f.get('floor_speed_mps'))}м/с h/st={fmt(f.get('height_vs_standing'))}"
                          f" tor={fmt(f.get('torso_angle_deg'))}"
                          f" drop={fmt(f.get('center_drop_heights'))} knee={fmt(f.get('knee_angle_deg'))}")
            items.append((label, (x1, max(0, y1 - 30)), RED if alert else (60, 120, 60)))
        recent = [txt for ts, txt in self.banner if fr.ts - ts < 4.0][-3:]
        for i, txt in enumerate(recent):
            items.append(("⚠ " + txt, (16, 16 + i * 40), RED))
        items.append((f"{self.clip}  t={self.t(fr.ts):.1f} с  · {self.fps:g} к/с обработки", (16, h - 44),
                      (30, 30, 30)))
        img = _texts(img, items, 22)
        self._ff.stdin.write(np.ascontiguousarray(img).tobytes())

    def run(self):
        orig = self.persons.update

        def spy(pose, ts):
            persons, lost = orig(pose, ts)
            self._persons = persons
            return persons, lost
        self.persons.update = spy
        self._persons = []
        try:
            super().run()
        finally:
            if self._ff is not None:
                self._ff.stdin.close()
                self._ff.wait()


def full_frame(path: str) -> list:
    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return [[0, 0], [w, 0], [w, h], [0, h]]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="clips.yaml")
    ap.add_argument("--config", default=str(ROOT / "core" / "cameras.yaml"), help="модели и узел")
    ap.add_argument("--patterns-dir", default=None, help="другая папка карточек (для сравнения порогов)")
    ap.add_argument("--out", default="out/eval")
    ap.add_argument("--only", nargs="*", help="только эти файлы")
    ap.add_argument("--video", action="store_true", help="сохранить ролики с разметкой")
    ap.add_argument("--debug", action="store_true", help="подписывать признаки (скорость, наклон…)")
    ap.add_argument("--cache", default=None, help="папка кэша выходов моделей (повторные прогоны без инференса)")
    ap.add_argument("--features", action="store_true", help="записать признаки треков по кадрам в features.csv")
    ap.add_argument("--behavior", default=None, help="другая модель поведения (.onnx) или none — только правила карточек")
    ap.add_argument("--log-level", default="WARNING")
    a = ap.parse_args(argv)
    logging.basicConfig(level=a.log_level.upper(), format="%(levelname)-7s %(name)s: %(message)s")

    spec = yaml.safe_load(open(a.clips, encoding="utf-8"))
    base = Path(a.clips).parent
    d = spec.get("defaults") or {}
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    if a.patterns_dir:
        cfg["patterns_dir"] = a.patterns_dir
    if a.behavior:
        cfg.setdefault("models", {})["behavior"] = None if a.behavior.lower() == "none" else a.behavior
    models = Models(cfg, ROOT)
    cards = load_patterns(cfg, ROOT)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    all_recs = []
    fcsv = fw = None
    if a.features:
        fcsv = open(out / "features.csv", "w", newline="", encoding="utf-8")
        fw = csv.DictWriter(fcsv, fieldnames=["file", "t", "track", "height_px", "ref_height_px", "aspect_ratio",
                                              "track_age_sec", "person_conf", "speed_heights_per_sec", "floor_speed_mps", "height_vs_standing", "stand_drop",
                                              "torso_angle_deg", "center_drop_heights", "hips_height_ratio",
                                              "knee_angle_deg", "wrist_speed_heights_per_sec"], extrasaction="ignore")
        fw.writeheader()
    pose0, det0 = models.pose, models.det
    for c in spec["clips"]:
        if a.only and c["file"] not in a.only:
            continue
        src = str((base / c["file"]).resolve())
        zones = []
        for z in (c.get("zones") or []) + (d.get("zones") or []):
            zones.append({**z, "polygon": full_frame(src) if z["polygon"] == "full" else z["polygon"]})
        pats = list(c.get("patterns") or d.get("patterns") or []) + list(c.get("add_patterns") or [])
        cam = {"id": Path(c["file"]).stem, "name": c["file"], "area": "тест", "source": src,
               "fps": c.get("fps", d.get("fps", 10)), "patterns": pats,
               "overrides": {p: {"schedule": "always"} for p in pats}, "zones": zones,
               "floor": c.get("floor") or d.get("floor")}
        video = out / "video" / (Path(c["file"]).stem + "_marked.mp4") if a.video else None
        if a.cache:
            stem = Path(c["file"]).stem
            models.pose = CachedModel(pose0, Path(a.cache) / f"{stem}_pose.pkl")
            models.det = CachedModel(det0, Path(a.cache) / f"{stem}_det.pkl") if det0 is not None else None
        p = EvalPipeline(cam, cfg, models, cards, None, None, True, threading.Event(), ROOT,
                         clip=c["file"], video_out=video, debug=a.debug, feats=fw)
        p.run()
        if a.cache:
            models.pose.save()
            if models.det is not None:
                models.det.save()
        n = sum(1 for r in p.records if r["kind"] == "create")
        print(f"{c['file']}: кадров {p.n}, карточки {[x.id for x in p.engine.cards]}, тревог {n}"
              + "".join(f"\n    {r['pattern_id']} трек {r['track_id']}: начало {r['start']} с, тревога {r['confirm']} с"
                        for r in p.records if r["kind"] == "create"))
        all_recs += p.records
    with open(out / "events_all.jsonl", "w", encoding="utf-8") as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "events.jsonl", "w", encoding="utf-8") as f:     # формат evaluate: только тревоги
        for r in all_recs:
            if r["kind"] == "create":
                f.write(json.dumps({k: r[k] for k in ("file", "pattern_id", "track_id", "start", "confirm")},
                                   ensure_ascii=False) + "\n")
    if fcsv:
        fcsv.close()
    print(f"\nсобытия: {out / 'events.jsonl'}")


if __name__ == "__main__":
    main()
