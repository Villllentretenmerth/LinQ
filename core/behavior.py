"""behavior_v1: классификатор поведения по окну скелета одного трека (блок `person_sequence` в карточках).

Вход модели — последние WINDOW_SEC секунд трека, пересчитанные на сетку из T = WINDOW_SEC * FPS моментов
(тензор [1, T, F]). Один и тот же `window_tensor` строит окна и для обучения (core.tools.behavior_data),
и в конвейере, поэтому модель в работе видит ровно то же, на чём училась.

Признаки кадра (всё в ростах человека, поэтому не зависит от расстояния до камеры):
    поза      — 17 точек относительно центра бёдер этого кадра (x, y) и уверенность точки   — 51
    траектория — низ и центр рамки относительно последнего кадра окна, log высоты и ширины рамки
                 относительно последнего кадра (бег на камеру — рамка растёт)              — 6
    форма     — ширина/высота рамки                                                         — 1
    есть      — 1, если в этот момент у трека был кадр, иначе 0 (пропуск трекера)            — 1
Калибровка пола в признаки не входит: в чужих датасетах её нет, и модель не должна от неё зависеть.

Версия признаков 2 (core/behavior_v2.py) — модель на открытых данных NTU60 + HMDB51 + KTH: окно (85, T) по точкам
скелета и их скорости. Какая версия у модели — записано в её метаданных; конвейер строит окно нужной версии сам.

Модель считается раз в STRIDE_SEC на трек (как и шаг подтверждения в Confirmer), результат кэшируется в истории
трека, все карточки читают один и тот же ответ.
"""
import json
import logging
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from . import behavior_v2
from .features import KP_CONF, L_HIP, R_HIP, TrackHistory, bottom_center, box_center, kp

log = logging.getLogger("core.behavior")

WINDOW_SEC = 2.0
FPS = 10.0
T = int(round(WINDOW_SEC * FPS))
STRIDE_SEC = 0.5
MAX_GAP_SEC = 0.15        # ближайший кадр трека дальше этого от точки сетки — момента «нет»
MIN_PRESENT = 0.6         # окно с меньшей долей кадров не оцениваем
FEATURES = 17 * 3 + 6 + 1 + 1
FEATURE_VERSION = 1

# пары левая/правая точка COCO-17 — для зеркального отражения при обучении
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
X_COLUMNS = [3 * i for i in range(17)] + [51, 53]     # колонки с координатой x (меняют знак при отражении)


def _frame_features(s, last, ref: float) -> np.ndarray:
    v = np.zeros(FEATURES, np.float32)
    x1, y1, x2, y2 = (float(t) for t in s.xyxy)
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    if s.kpts is not None:
        hip = kp(s.kpts, L_HIP, R_HIP)
        c = hip if hip is not None else box_center(s)
        for i in range(17):
            x, y, conf = (float(t) for t in s.kpts[i])
            if conf >= KP_CONF:
                v[3 * i] = (x - c[0]) / ref
                v[3 * i + 1] = (y - c[1]) / ref
            v[3 * i + 2] = conf
    lx1, ly1, lx2, ly2 = (float(t) for t in last.xyxy)
    b, lb = bottom_center(s), bottom_center(last)
    m, lm = box_center(s), box_center(last)
    v[51:53] = (b - lb) / ref
    v[53:55] = (m - lm) / ref
    v[55] = np.log(h / max(ly2 - ly1, 1.0))
    v[56] = np.log(w / max(lx2 - lx1, 1.0))
    v[57] = w / h
    v[58] = 1.0
    return v


def window_tensor(h: TrackHistory, now: float | None = None) -> np.ndarray | None:
    """Окно [T, FEATURES], заканчивающееся в момент now (по умолчанию — последний кадр трека), или None,
    если трек слишком короткий или в окне много пропусков."""
    if not h.samples:
        return None
    now = h.last_ts if now is None else now
    samples = [s for s in h.samples if now - WINDOW_SEC - MAX_GAP_SEC <= s.ts <= now + 1e-6]
    if not samples or samples[0].ts > now - WINDOW_SEC + 0.5:
        return None
    last = samples[-1]
    ref = max(1.0, max(float(s.xyxy[3] - s.xyxy[1]) for s in samples))
    ts = np.array([s.ts for s in samples])
    out = np.zeros((T, FEATURES), np.float32)
    present = 0
    for k in range(T):
        t = now - (T - 1 - k) / FPS
        j = int(np.argmin(np.abs(ts - t)))
        if abs(ts[j] - t) <= MAX_GAP_SEC:
            out[k] = _frame_features(samples[j], last, ref)
            present += 1
    return out if present >= MIN_PRESENT * T else None


class BehaviorModel:
    """ONNX-модель behavior_v1: вход [1, T, F], выход — вероятности классов. Классы и размер окна читаются
    из метаданных ONNX (их пишет core.tools.train_behavior)."""

    def __init__(self, path: str | Path, threads: int | None = None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.classes: list[str] = json.loads(meta["classes"])
        shape = json.loads(meta.get("input_shape", f"[{T}, {FEATURES}]"))
        self.version = int(meta.get("feature_version", 0))
        expected = {FEATURE_VERSION: (T, FEATURES), 2: (behavior_v2.CHANNELS, behavior_v2.T)}
        if self.version not in expected or tuple(shape) != expected[self.version]:
            raise ValueError(f"{path}: модель обучена на окнах {shape}, версия признаков "
                             f"{meta.get('feature_version')}, а конвейер строит {expected}")
        self.name = meta.get("name", "behavior_v1")
        self.enabled: set[str] = set(self.classes)   # какие классы карточки берут у модели, остальные — по правилам
        self.timer = None             # rep.timer узла: время инференса уходит в метрики под именем модели

    def probs(self, h: TrackHistory) -> dict[str, float] | None:
        """Вероятности классов для трека; считаются не чаще раза в STRIDE_SEC, между ними — из кэша."""
        cache = getattr(h, "_behavior", None)
        if cache is not None and h.last_ts - cache[0] < STRIDE_SEC - 1e-6:
            return cache[1]
        x = window_tensor(h) if self.version == FEATURE_VERSION else behavior_v2.window_from_history(h)
        p = None
        if x is not None:
            with (self.timer(self.name) if self.timer else nullcontext()):
                out = self.sess.run(None, {self.input: x[None]})[0][0]
            p = {c: float(v) for c, v in zip(self.classes, out)}
        h._behavior = (h.last_ts, p)
        return p
