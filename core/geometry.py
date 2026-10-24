"""Зоны и линии камеры в пикселях кадра: точка в полигоне, пересечение линии, расстояние до рамки."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Zone:
    name: str
    tags: set[str]
    polygon: np.ndarray            # (N, 2)

    def contains(self, pt) -> bool:
        return point_in_polygon(pt, self.polygon)


@dataclass
class Line:
    name: str
    tags: set[str]
    p1: np.ndarray
    p2: np.ndarray


@dataclass
class Scene:
    zones: list[Zone] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)

    @classmethod
    def from_config(cls, zones: list[dict] | None, lines: list[dict] | None) -> "Scene":
        zs = [Zone(z.get("name") or f"zone-{i}", set(z.get("tags") or []), np.asarray(z["polygon"], dtype=np.float32))
              for i, z in enumerate(zones or [])]
        ls = [Line(l.get("name") or f"line-{i}", set(l.get("tags") or []),
                   np.asarray(l["points"][0], dtype=np.float32), np.asarray(l["points"][1], dtype=np.float32))
              for i, l in enumerate(lines or [])]
        for z in zs:
            if len(z.polygon) < 3:
                raise ValueError(f"У зоны «{z.name}» меньше трёх точек")
        return cls(zs, ls)

    def has_zone_tag(self, tag: str) -> bool:
        return any(tag in z.tags for z in self.zones)

    def has_line_tag(self, tag: str) -> bool:
        return any(tag in l.tags for l in self.lines)

    def zones_with(self, tags) -> list[Zone]:
        tags = {tags} if isinstance(tags, str) else set(tags)
        return [z for z in self.zones if z.tags & tags]

    def zone_at(self, pts, tags) -> Zone | None:
        """Первая зона с нужным тегом, в которую попала хотя бы одна из точек."""
        for z in self.zones_with(tags):
            for p in pts:
                if p is not None and z.contains(p):
                    return z
        return None


def point_in_polygon(pt, poly: np.ndarray) -> bool:
    """Чётность пересечений луча (точка на границе считается внутри с точностью до пикселя)."""
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _ccw(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a, b, c, d) -> bool:
    """Движение a→b пересекло линию c-d: a и b по разные стороны линии (точка на линии считается
    «перешедшей», чтобы шаг ровно на линию не терялся и не считался дважды), а место пересечения — в пределах c-d."""
    d1, d2 = _ccw(c, d, a), _ccw(c, d, b)
    d3, d4 = _ccw(a, b, c), _ccw(a, b, d)
    return ((d1 >= 0) != (d2 >= 0)) and d3 * d4 <= 0 and (d3 != 0 or d4 != 0)


def intersection_point(a, b, c, d):
    a, b, c, d = (np.asarray(v, dtype=np.float64) for v in (a, b, c, d))
    r, s = b - a, d - c
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-9:
        return (a + b) / 2
    t = ((c - a)[0] * s[1] - (c - a)[1] * s[0]) / den
    return a + t * r


def point_to_box_distance(pt, box) -> float:
    """Расстояние от точки до прямоугольника xyxy (0, если внутри)."""
    x, y = float(pt[0]), float(pt[1])
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return float(np.hypot(dx, dy))


def box_inside_ratio(inner, outer) -> float:
    """Какая доля площади inner лежит внутри outer."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area
