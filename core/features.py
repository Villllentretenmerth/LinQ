"""История трека и признаки для правил-заглушек (`fallback_rule` в карточках), пока behavior_v1 не обучен.

Все расстояния — в ростах человека (`ref_height` — рост в пикселях за последние секунды), поэтому пороги
не зависят от того, далеко человек от камеры или близко. Если у камеры откалиброван пол (`floor` в конфиге:
4 точки на полу и их координаты в метрах), считается и скорость по полу в м/с — она не зависит от того,
бежит человек поперёк кадра или прямо на камеру. Признак, который нельзя посчитать (мало истории,
не видны точки), равен None, и любое сравнение с ним ложно.
"""
import ast
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

# ключевые точки COCO-17
NOSE, L_EYE, R_EYE = 0, 1, 2
L_SH, R_SH, L_WR, R_WR = 5, 6, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16
KP_CONF = 0.3

SPEED_WINDOW = 1.0   # с — по какому отрезку считаем скорость
DROP_WINDOW = 3.0    # с — относительно какого прошлого считаем «центр опустился»
REF_WINDOW = 4.0     # с — за сколько секунд берём рост (до падения человек стоял)
FALL_WINDOW = 1.0    # с — за какое время рост «осел» (stand_drop): падение быстрее, чем сесть на пол


@dataclass
class Sample:
    ts: float
    xyxy: np.ndarray          # (4,)
    kpts: np.ndarray | None   # (17, 3) или None (детектор без позы)
    conf: float


def kp(kpts, *idx):
    """Среднее видимых точек из списка или None."""
    if kpts is None:
        return None
    pts = [kpts[i, :2] for i in idx if kpts[i, 2] >= KP_CONF]
    return np.mean(pts, axis=0) if pts else None


def feet_point(s: Sample):
    p = kp(s.kpts, L_ANK, R_ANK)
    if p is not None:
        return p
    x1, _, x2, y2 = s.xyxy
    return np.array([(x1 + x2) / 2, y2], dtype=np.float32)


def bottom_center(s: Sample):
    x1, _, x2, y2 = s.xyxy
    return np.array([(x1 + x2) / 2, y2], dtype=np.float32)


def box_center(s: Sample):
    x1, y1, x2, y2 = s.xyxy
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)


def torso_center(s: Sample):
    p = kp(s.kpts, L_SH, R_SH, L_HIP, R_HIP)
    return p if p is not None else box_center(s)


def wrist_points(s: Sample) -> list:
    if s.kpts is None:
        return []
    return [s.kpts[i, :2] for i in (L_WR, R_WR) if s.kpts[i, 2] >= KP_CONF]


def body_point(s: Sample, name: str) -> list:
    """Точки тела для in_zone: feet, wrists, hips, head, center."""
    if name == "feet":
        return [feet_point(s)]
    if name == "wrists":
        return wrist_points(s)
    if name == "hips":
        p = kp(s.kpts, L_HIP, R_HIP)
        return [p if p is not None else box_center(s)]
    if name == "head":
        p = kp(s.kpts, NOSE, L_EYE, R_EYE)
        if p is None:
            x1, y1, x2, y2 = s.xyxy
            p = np.array([(x1 + x2) / 2, y1 + 0.08 * (y2 - y1)], dtype=np.float32)
        return [p]
    if name == "center":
        return [box_center(s)]
    raise ValueError(f"Неизвестная точка тела: {name} (feet, wrists, hips, head, center)")


def _angle_at(a, b, c) -> float | None:
    if a is None or b is None or c is None:
        return None
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return None
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / n, -1, 1))))


class TrackHistory:
    """История одного трека человека: рамка, скелет, уверенность — за последние keep_sec секунд."""

    def __init__(self, track_id: int, keep_sec: float = 15.0, floor=None):
        self.id = track_id
        self.floor = floor        # FloorCalib камеры (калибровка пола) или None
        self.keep_sec = keep_sec
        self.samples: deque[Sample] = deque()
        self.first_ts: float | None = None
        self._feat_cache: tuple[float, dict] | None = None

    def add(self, ts: float, xyxy, kpts, conf: float) -> None:
        self.samples.append(Sample(ts, np.asarray(xyxy, np.float32),
                                   None if kpts is None else np.asarray(kpts, np.float32), float(conf)))
        if self.first_ts is None:
            self.first_ts = ts
        while self.samples and self.samples[0].ts < ts - self.keep_sec:
            self.samples.popleft()
        self._feat_cache = None

    def inherit(self, other: "TrackHistory") -> None:
        """Продолжить историю другого трека (смена ID при падении или перекрытии)."""
        self.samples = deque(other.samples)
        self.first_ts = other.first_ts
        self._feat_cache = None

    @property
    def last(self) -> Sample:
        return self.samples[-1]

    @property
    def last_ts(self) -> float:
        return self.samples[-1].ts

    def since(self, ts: float) -> list[Sample]:
        return [s for s in self.samples if s.ts >= ts]

    def ref_height(self) -> float:
        now = self.last_ts
        hs = [s.xyxy[3] - s.xyxy[1] for s in self.samples if s.ts >= now - REF_WINDOW]
        return max(1.0, float(max(hs)))

    def body_length(self) -> float:
        """Длина тела в пикселях при любой позе (лежащий человек — по ширине рамки)."""
        x1, y1, x2, y2 = self.last.xyxy
        return max(self.ref_height(), float(max(x2 - x1, y2 - y1)))

    # ---------- признаки ----------

    def features(self) -> dict:
        if self._feat_cache and self._feat_cache[0] == self.last_ts:
            return self._feat_cache[1]
        f = compute_features(self)
        self._feat_cache = (self.last_ts, f)
        return f

    def motion(self, seconds: float) -> float | None:
        """Насколько сместился центр тела за последние seconds секунд, в длинах тела.
        None — если трек ещё не прожил seconds секунд (нечего сравнивать)."""
        now = self.last_ts
        win = self.since(now - seconds)
        if not win or now - win[0].ts < 0.9 * seconds or (self.first_ts is not None and now - self.first_ts < 0.9 * seconds):
            return None
        centers = np.array([box_center(s) for s in win])
        med = np.median(centers, axis=0)
        return float(np.max(np.linalg.norm(centers - med, axis=1)) / self.body_length())


class FloorCalib:
    """Калибровка пола камеры по 4 точкам с известными координатами в метрах.

    Даёт две вещи: перевод точки кадра в метры на полу (скорость в м/с не зависит от направления движения)
    и ожидаемый рост стоящего человека в пикселях в этой точке пола (лежащего «вдоль взгляда камеры»
    видно по тому, что он намного ниже, чем стоящий там же). Фокусное расстояние, если не задано,
    оценивается из той же гомографии (оси пола перпендикулярны), центр кадра — главная точка.
    """

    def __init__(self, image_pts, world_m, focal_px: float | None = None):
        import cv2
        img, wld = np.float32(image_pts), np.float32(world_m)
        if img.shape != (4, 2) or wld.shape != (4, 2):
            raise ValueError("floor: нужны ровно 4 точки image и 4 точки world_m")
        self.H = cv2.getPerspectiveTransform(img, wld).astype(np.float64)        # кадр -> пол, м
        self.G = np.linalg.inv(self.H)                                              # пол -> кадр
        self.focal = focal_px
        self.P = None

    def set_frame(self, w: int, h: int) -> None:
        if self.P is not None:
            return
        cx, cy = w / 2.0, h / 2.0
        g1, g2, g3 = self.G[:, 0], self.G[:, 1], self.G[:, 2]
        f = self.focal
        if not f:
            a1, b1, c1 = g1[0] - cx * g1[2], g1[1] - cy * g1[2], g1[2]
            a2, b2, c2 = g2[0] - cx * g2[2], g2[1] - cy * g2[2], g2[2]
            f2 = -(a1 * a2 + b1 * b2) / (c1 * c2) if abs(c1 * c2) > 1e-12 else -1
            f = math.sqrt(f2) if f2 > 0 else float(w)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        M = np.linalg.inv(K) @ self.G
        lam = (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1])) / 2
        r1, r2, t = M[:, 0] / np.linalg.norm(M[:, 0]), M[:, 1] / np.linalg.norm(M[:, 1]), M[:, 2] / lam
        if t[2] < 0:
            r1, r2, t = -r1, -r2, -t
        r3 = np.cross(r1, r2)
        R = np.stack([r1, r2, r3], 1)
        if (-R.T @ t)[2] < 0:                 # ось «вверх» должна смотреть к камере
            r3 = -r3
            R = np.stack([r1, r2, r3], 1)
        self.focal = f
        self.P = K @ np.hstack([R, t[:, None]])

    def to_floor(self, pt) -> np.ndarray | None:
        v = self.H @ np.array([float(pt[0]), float(pt[1]), 1.0])
        return None if abs(v[2]) < 1e-9 else v[:2] / v[2]

    def standing_px(self, pt, height_m: float = 1.7) -> float | None:
        """Сколько пикселей занимал бы стоящий человек ростом height_m, стоящий в этой точке пола."""
        X = self.to_floor(pt)
        if X is None or self.P is None:
            return None
        a = self.P @ np.array([X[0], X[1], 0.0, 1.0])
        b = self.P @ np.array([X[0], X[1], height_m, 1.0])
        if a[2] <= 0 or b[2] <= 0:
            return None
        return float(np.linalg.norm(a[:2] / a[2] - b[:2] / b[2]))


def floor_homography(cfg: dict | None) -> "FloorCalib | None":
    """floor: {image: [[u, v] ×4], world_m: [[x, y] ×4], focal_px: (необязательно)}"""
    if not cfg:
        return None
    return FloorCalib(cfg["image"], cfg["world_m"], cfg.get("focal_px"))


def compute_features(h: TrackHistory) -> dict:
    s = h.last
    now = s.ts
    ref = h.ref_height()
    x1, y1, x2, y2 = s.xyxy
    w, ht = float(x2 - x1), float(y2 - y1)
    f: dict = {"height_px": ht, "ref_height_px": ref, "aspect_ratio": w / max(ht, 1.0),
               "track_age_sec": now - (h.first_ts or now), "person_conf": s.conf}

    # скорость точки ног (низ рамки устойчивее щиколоток)
    past = h.since(now - SPEED_WINDOW)
    if past and now - past[0].ts >= 0.3:
        d = float(np.linalg.norm(bottom_center(s) - bottom_center(past[0])))
        f["speed_heights_per_sec"] = d / (now - past[0].ts) / ref
    else:
        f["speed_heights_per_sec"] = None

    # скорость по полу, м/с (если пол откалиброван): точка ног -> метры через гомографию
    f["floor_speed_mps"] = None
    if h.floor is not None and past and now - past[0].ts >= 0.3:
        a, b = h.floor.to_floor(bottom_center(past[0])), h.floor.to_floor(bottom_center(s))
        if a is not None and b is not None:
            f["floor_speed_mps"] = float(np.linalg.norm(b - a)) / (now - past[0].ts)
    # рост сейчас относительно роста стоящего в этой точке пола (1 — стоит, ~0,3 — лежит, ~0,6 — сидит)
    stand = h.floor.standing_px(bottom_center(s)) if h.floor is not None else None
    f["height_vs_standing"] = ht / stand if stand else None
    # резкое «оседание»: за FALL_WINDOW секунд рост относительно стоящего упал на stand_drop
    # (берём максимум за последние DROP_WINDOW секунд, как center_drop). Ловит падение «вдоль взгляда» камеры,
    # когда наклон корпуса в кадре почти не меняется; сесть на пол — медленнее, присесть — не так низко.
    f["stand_drop"] = None
    if stand:
        hist = [(p.ts, float(p.xyxy[3] - p.xyxy[1]) / sp) for p in h.since(now - DROP_WINDOW - FALL_WINDOW)
                if (sp := h.floor.standing_px(bottom_center(p)))]
        best = None
        for ts_i, r_i in hist:
            if ts_i < now - DROP_WINDOW - 1e-9:
                continue
            before = [r for t, r in hist if ts_i - FALL_WINDOW - 1e-9 <= t < ts_i]
            if before:
                d_i = max(before) - r_i
                best = d_i if best is None else max(best, d_i)
        f["stand_drop"] = None if best is None else max(0.0, best)

    # наклон корпуса: 0° — стоит, 90° — лежит; без точек — по пропорциям рамки
    sh, hip = kp(s.kpts, L_SH, R_SH), kp(s.kpts, L_HIP, R_HIP)
    if sh is not None and hip is not None and np.linalg.norm(sh - hip) > 1:
        v = sh - hip
        f["torso_angle_deg"] = math.degrees(math.acos(float(np.clip(-v[1] / np.linalg.norm(v), -1, 1))))
    else:
        f["torso_angle_deg"] = math.degrees(math.atan2(w, max(ht, 1.0)))

    # насколько опустился центр корпуса относительно самого высокого положения за DROP_WINDOW
    past = h.since(now - DROP_WINDOW)
    if len(past) >= 2:
        top = min(float(torso_center(p)[1]) for p in past)
        f["center_drop_heights"] = max(0.0, float(torso_center(s)[1]) - top) / ref
    else:
        f["center_drop_heights"] = None

    # высота бёдер над полом относительно роста (пролезание, ползком)
    hip_y = hip[1] if hip is not None else None
    ank = kp(s.kpts, L_ANK, R_ANK)
    floor_y = ank[1] if ank is not None else y2
    f["hips_height_ratio"] = None if hip_y is None else max(0.0, float(floor_y - hip_y)) / ref

    # угол в колене (180° — прямая нога)
    if s.kpts is not None:
        angles = [a for a in (_angle_at(kp(s.kpts, L_HIP), kp(s.kpts, L_KNEE), kp(s.kpts, L_ANK)),
                              _angle_at(kp(s.kpts, R_HIP), kp(s.kpts, R_KNEE), kp(s.kpts, R_ANK))) if a is not None]
        f["knee_angle_deg"] = float(np.mean(angles)) if angles else None
    else:
        f["knee_angle_deg"] = None

    # скорость кистей (резкие движения рук)
    past = h.since(now - 0.5)
    if past and now - past[0].ts >= 0.2 and s.kpts is not None and past[0].kpts is not None:
        sp = [float(np.linalg.norm(s.kpts[i, :2] - past[0].kpts[i, :2])) / (now - past[0].ts) / ref
              for i in (L_WR, R_WR) if s.kpts[i, 2] >= KP_CONF and past[0].kpts[i, 2] >= KP_CONF]
        f["wrist_speed_heights_per_sec"] = max(sp) if sp else None
    else:
        f["wrist_speed_heights_per_sec"] = None
    return f


KNOWN_FEATURES = {"height_px", "ref_height_px", "aspect_ratio", "track_age_sec", "person_conf",
                  "speed_heights_per_sec", "floor_speed_mps", "height_vs_standing", "stand_drop", "torso_angle_deg", "center_drop_heights", "hips_height_ratio",
                  "knee_angle_deg", "wrist_speed_heights_per_sec"}


# ---------------------------------------------------------------- правила-заглушки

class RuleError(ValueError):
    pass


_ALLOWED = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.Compare,
            ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq, ast.Name, ast.Load, ast.Constant,
            ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div)


class Rule:
    """Безопасное выражение из карточки: "torso_angle_deg > 60 and center_drop_heights > 0.4".
    Разрешены только сравнения, and/or/not, арифметика и имена признаков из KNOWN_FEATURES."""

    def __init__(self, expr: str, known: set[str] = KNOWN_FEATURES):
        self.expr = expr
        try:
            self.tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise RuleError(f"синтаксис правила «{expr}»: {e.msg}") from None
        names = set()
        for node in ast.walk(self.tree):
            if not isinstance(node, _ALLOWED):
                raise RuleError(f"в правиле «{expr}» недопустимо: {type(node).__name__}")
            if isinstance(node, ast.Name):
                names.add(node.id)
        unknown = names - known
        if unknown:
            raise RuleError(f"признаков нет в конвейере: {', '.join(sorted(unknown))}")
        self.names = names

    def __call__(self, feats: dict) -> bool:
        return bool(self._eval(self.tree.body, feats))

    def _eval(self, n, f):
        if isinstance(n, ast.BoolOp):
            vals = (self._eval(v, f) for v in n.values)
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.UnaryOp):
            v = self._eval(n.operand, f)
            if isinstance(n.op, ast.Not):
                return not v
            return None if v is None else -v
        if isinstance(n, ast.Compare):
            left = self._eval(n.left, f)
            for op, comp in zip(n.ops, n.comparators):
                right = self._eval(comp, f)
                if left is None or right is None:
                    return False
                ok = {ast.Gt: left > right, ast.GtE: left >= right, ast.Lt: left < right,
                      ast.LtE: left <= right, ast.Eq: left == right, ast.NotEq: left != right}[type(op)]
                if not ok:
                    return False
                left = right
            return True
        if isinstance(n, ast.BinOp):
            a, b = self._eval(n.left, f), self._eval(n.right, f)
            if a is None or b is None:
                return None
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            return a / b if b else None
        if isinstance(n, ast.Name):
            return f.get(n.id)
        if isinstance(n, ast.Constant):
            return n.value
        raise RuleError(type(n).__name__)
