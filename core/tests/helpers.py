"""Синтетические люди со скелетом COCO-17 для тестов движка без моделей."""
import math

import numpy as np

from core.engine import FrameContext, Obj, Person, Vehicle
from core.features import TrackHistory
from core.geometry import Scene

# доля роста от макушки и смещение по x (в ростах) для стоящего человека
_POSE = [
    (0.06, 0.00), (0.05, -0.02), (0.05, 0.02), (0.06, -0.04), (0.06, 0.04),    # нос, глаза, уши
    (0.19, -0.11), (0.19, 0.11), (0.33, -0.14), (0.33, 0.14),                   # плечи, локти
    (0.45, -0.13), (0.45, 0.13), (0.52, -0.07), (0.52, 0.07),                   # кисти, бёдра
    (0.72, -0.07), (0.72, 0.07), (0.94, -0.07), (0.94, 0.07),                   # колени, щиколотки
]


def skeleton(cx: float, feet_y: float, height: float = 200, angle_deg: float = 0.0):
    """Рамка и 17 точек человека ростом height, стоящего ногами в (cx, feet_y); angle_deg — поворот вокруг
    точки ног (90 — лежит головой вправо)."""
    a = math.radians(angle_deg)
    pts = []
    for frac, dx in _POSE:
        x, y = dx * height, -(1 - frac) * height          # относительно точки ног, y вверх отрицательный
        xr, yr = x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)
        pts.append((cx + xr, feet_y + yr, 0.9))
    k = np.array(pts, np.float32)
    top = feet_y - height if abs(angle_deg) < 1 else k[:, 1].min() - 0.03 * height
    box = np.array([k[:, 0].min() - 0.05 * height, top, k[:, 0].max() + 0.05 * height, feet_y + 0.01 * height],
                   np.float32)
    return box, k


class Sim:
    """Прогон людей по кадрам: sim.step(ts, [(track_id, cx, feet_y, angle)], vehicles=..., phones=...)."""

    def __init__(self, scene: Scene | None = None, height: float = 200):
        self.scene = scene or Scene()
        self.height = height
        self.tracks: dict[int, TrackHistory] = {}

    def ctx(self, ts: float, people, vehicles=(), phones=()) -> FrameContext:
        persons = []
        for tid, cx, feet_y, angle in people:
            box, k = skeleton(cx, feet_y, self.height, angle)
            h = self.tracks.setdefault(tid, TrackHistory(tid, 20.0))
            h.add(ts, box, k, 0.9)
            persons.append(Person(tid, box, 0.9, h))
        return FrameContext(ts, persons, self.scene,
                            [Vehicle(*v) for v in vehicles], [Obj(np.asarray(b, np.float32), c, "cell phone") for b, c in phones])


def rect(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
