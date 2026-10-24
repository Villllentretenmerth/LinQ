"""Движок карточек: читает patterns/*.yaml, собирает из блоков условия и решает, когда отправить событие.

Карточки не запускают модели — они читают общие выходы (треки людей со скелетом, техника, телефоны).
Цепочка для каждой пары «карточка × трек»:

    расписание → зоны include/exclude → условия (И; any_of — ИЛИ) → подтверждение N из M окон (+ hold_sec)
    → дедупликация (одно событие на трек + блокировка повторов по камера+зона+тип) → событие

Условие `still` (неподвижность) в карточке, где есть и другие условия, — второй этап: тревога уходит сразу после
подтверждения остального, а неподвижность дописывается в событие обновлением (падение → «лежит без движения 10 с»).

Блок, для которого нет модели, заменяется `fallback_rule` карточки; если нет и его — карточка выключается
с предупреждением в лог. Движок не зависит от моделей, поэтому тестируется на синтетических треках.
"""
import itertools
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from .features import Rule, RuleError, TrackHistory, body_point
from .geometry import Scene, box_inside_ratio, intersection_point, point_to_box_distance, segments_intersect

log = logging.getLogger("core.engine")


class Unsupported(Exception):
    """Карточку (или условие) нельзя включить на этой камере: нет модели, зоны или линии."""


# ---------------------------------------------------------------- что видит движок на кадре

@dataclass
class Person:
    track_id: int
    xyxy: np.ndarray
    conf: float
    hist: TrackHistory


@dataclass
class Vehicle:
    track_id: int
    xyxy: np.ndarray
    conf: float
    cls: str
    speed_px: float | None = None     # скорость центра рамки, пикселей в секунду


@dataclass
class Obj:
    xyxy: np.ndarray
    conf: float
    cls: str


@dataclass
class FrameContext:
    ts: float
    persons: list[Person]
    scene: Scene
    vehicles: list[Vehicle] = field(default_factory=list)
    phones: list[Obj] = field(default_factory=list)


# ---------------------------------------------------------------- условия

class Cond:
    needs: frozenset = frozenset()   # "det" — нужен детектор объектов
    label = ""

    def __call__(self, ctx: FrameContext, p: Person) -> tuple[bool, dict]:
        raise NotImplementedError


class AllOf(Cond):
    def __init__(self, items: list[Cond]):
        self.items = items
        self.needs = frozenset().union(*(c.needs for c in items)) if items else frozenset()
        self.label = " И ".join(c.label for c in items)

    def __call__(self, ctx, p):
        info: dict = {}
        scores = []
        for c in self.items:
            ok, i = c(ctx, p)
            if not ok:
                return False, {}
            scores.append(i.pop("score", 1.0))
            _merge_info(info, i)
        info["score"] = min(scores) if scores else 1.0
        return True, info


class AnyOf(Cond):
    def __init__(self, items: list[Cond]):
        self.items = items
        self.needs = frozenset().union(*(c.needs for c in items))
        self.label = "(" + " ИЛИ ".join(c.label for c in items) + ")"

    def __call__(self, ctx, p):
        for c in self.items:
            ok, i = c(ctx, p)
            if ok:
                return True, i
        return False, {}


def _merge_info(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if k == "features":
            dst.setdefault("features", {}).update(v)
        elif k not in dst:
            dst[k] = v


class RuleCond(Cond):
    def __init__(self, rule: Rule, label: str):
        self.rule, self.label = rule, label

    def __call__(self, ctx, p):
        f = p.hist.features()
        if not self.rule(f):
            return False, {}
        return True, {"features": {n: _r(f.get(n)) for n in sorted(self.rule.names)}, "rules": [self.rule.expr]}


class BehaviorCond(Cond):
    """Класс поведения от модели (behavior_v1) по окну скелета трека: вероятность класса >= min_score."""

    def __init__(self, model, cls: str, min_score: float):
        self.model, self.cls, self.min_score = model, cls, float(min_score)
        self.label = f"{getattr(model, 'name', 'behavior')}:{cls}≥{self.min_score:g}"

    def __call__(self, ctx, p):
        probs = self.model.probs(p.hist)
        s = (probs or {}).get(self.cls, 0.0)
        if s < self.min_score:
            return False, {}
        return True, {"score": s, "features": {f"p_{self.cls}": _r(s)}, "rules": [self.label]}


class InZone(Cond):
    def __init__(self, point: str, tag: str):
        body_point_names = {"feet", "wrists", "hips", "head", "center"}
        if point not in body_point_names:
            raise Unsupported(f"in_zone: неизвестная точка тела {point}")
        self.point, self.tag = point, tag
        self.label = f"{point} в зоне [{tag}]"

    def __call__(self, ctx, p):
        z = ctx.scene.zone_at(body_point(p.hist.last, self.point), self.tag)
        return (True, {"zone": z.name}) if z else (False, {})


class CrossLine(Cond):
    """Точка ног пересекла линию с тегом (не через зону-исключение, например проход). Пересечение — событие
    мгновенное, поэтому условие «помнит» его latch_sec секунд, чтобы подтверждение N из M успело сработать."""

    def __init__(self, line_tag: str, exclude_tag: str | None = None, latch_sec: float = 2.0):
        self.line_tag, self.exclude_tag, self.latch = line_tag, exclude_tag, latch_sec
        self.last_cross: dict[int, tuple[float, str]] = {}
        self.label = f"пересёк линию [{line_tag}]"

    def __call__(self, ctx, p):
        s = p.hist.samples
        if len(s) >= 2:
            a, b = body_point(s[-2], "feet")[0], body_point(s[-1], "feet")[0]
            for line in ctx.scene.lines:
                if self.line_tag in line.tags and segments_intersect(a, b, line.p1, line.p2):
                    x = intersection_point(a, b, line.p1, line.p2)
                    if self.exclude_tag and ctx.scene.zone_at([x], self.exclude_tag):
                        continue
                    self.last_cross[p.track_id] = (ctx.ts, line.name)
        hit = self.last_cross.get(p.track_id)
        if hit and ctx.ts - hit[0] <= self.latch:
            return True, {"line": hit[1]}
        return False, {}


class Still(Cond):
    def __init__(self, seconds: float, max_motion_heights: float = 0.05):
        self.seconds, self.max_motion = float(seconds), float(max_motion_heights)
        self.label = f"неподвижен {self.seconds:g} с"

    def __call__(self, ctx, p):
        m = p.hist.motion(self.seconds)
        if m is None or m > self.max_motion:
            return False, {}
        return True, {"motionless_sec": self.seconds, "motion_heights": _r(m)}


class VehicleNear(Cond):
    """Человек ближе max_distance_h ростов к движущейся технике (рамка техники вместо маски)."""

    needs = frozenset({"det"})

    def __init__(self, max_distance_h: float = 1.0, other_min_speed: float = 0.0):
        self.max_d, self.min_speed = float(max_distance_h), float(other_min_speed or 0)
        self.label = f"техника ближе {self.max_d:g} роста"

    def __call__(self, ctx, p):
        if not ctx.vehicles:
            return False, {}
        feet = body_point(p.hist.last, "feet")[0]
        ref = p.hist.ref_height()
        best = None
        for v in ctx.vehicles:
            if box_inside_ratio(p.xyxy, v.xyxy) > 0.7:     # человек внутри рамки техники — водитель
                continue
            d = point_to_box_distance(feet, v.xyxy) / ref
            if d > self.max_d:
                continue
            speed = None if v.speed_px is None else v.speed_px / ref
            if self.min_speed and (speed is None or speed < self.min_speed):
                continue
            if best is None or d < best[0]:
                best = (d, v, speed)
        if best is None:
            return False, {}
        d, v, speed = best
        return True, {"other_bbox": _box(v.xyxy), "other_label": f"{v.cls} #{v.track_id}", "score": v.conf,
                      "features": {"distance_heights": _r(d), "vehicle_speed_heights_per_sec": _r(speed)}}


class PhoneNearHand(Cond):
    """Телефон из COCO-детектора рядом с кистью (пока нет crop_attr_v1)."""

    needs = frozenset({"det"})

    def __init__(self, part: str = "hand", min_person_height_px: float = 0, max_dist_h: float = 0.15):
        self.part, self.min_h, self.max_d = part, float(min_person_height_px or 0), max_dist_h
        self.label = "телефон у кисти" if part == "hand" else "телефон у лица/кисти"

    def __call__(self, ctx, p):
        if not ctx.phones:
            return False, {}
        s = p.hist.last
        if s.xyxy[3] - s.xyxy[1] < self.min_h:      # человек слишком мелкий — сценарий не работает
            return False, {}
        ref = p.hist.ref_height()
        pts = body_point(s, "wrists") + (body_point(s, "head") if self.part == "face_hand" else [])
        for ph in sorted(ctx.phones, key=lambda o: -o.conf):
            if pts:
                d = min(point_to_box_distance(pt, ph.xyxy) for pt in pts) / ref
                ok = d <= self.max_d
            else:                                    # кисти не видны — телефон хотя бы внутри рамки человека
                d, ok = None, box_inside_ratio(ph.xyxy, p.xyxy) > 0.8
            if ok:
                return True, {"other_bbox": _box(ph.xyxy), "other_label": ph.cls, "score": ph.conf,
                              "features": {"phone_to_wrist_heights": _r(d)}}
        return False, {}


def _r(v, nd=3):
    return None if v is None else round(float(v), nd)


def _box(b) -> list[float]:
    return [round(float(x), 1) for x in b]


# ---------------------------------------------------------------- сборка условий из карточки

@dataclass
class CompileEnv:
    scene: Scene
    models: set[str] = field(default_factory=set)       # обученные модели атрибутов (пока пусто)
    behavior: dict = field(default_factory=dict)        # модели поведения: имя -> объект с .classes и .probs(hist)
    det_classes: set[str] = field(default_factory=set)  # что умеет yolo-det (COCO: car, truck, cell phone...)
    vehicle_classes: set[str] = field(default_factory=lambda: {"car", "truck"})
    phone_classes: set[str] = field(default_factory=lambda: {"cell phone"})


def compile_condition(c: dict, env: CompileEnv) -> Cond:
    if not isinstance(c, dict):
        raise Unsupported(f"непонятное условие: {c!r}")
    if "any_of" in c or "all_of" in c:
        items, reasons = [], []
        for sub in c.get("any_of") or c.get("all_of"):
            try:
                items.append(compile_condition(sub, env))
            except Unsupported as e:
                if "all_of" in c:
                    raise
                reasons.append(str(e))
        if not items:
            raise Unsupported("ни одна ветка any_of: " + "; ".join(reasons))
        return AnyOf(items) if "any_of" in c else AllOf(items)
    if "in_zone" in c:
        z = c["in_zone"]
        if not env.scene.has_zone_tag(z["zone_tag"]):
            raise Unsupported(f"на камере нет зоны с тегом {z['zone_tag']}")
        return InZone(z.get("point", "feet"), z["zone_tag"])
    if "cross_line" in c:
        z = c["cross_line"]
        if not env.scene.has_line_tag(z["line_tag"]):
            raise Unsupported(f"на камере нет линии с тегом {z['line_tag']}")
        return CrossLine(z["line_tag"], z.get("exclude_tag"), z.get("latch_sec", 2.0))
    if "still" in c:
        return Still(c["still"].get("seconds", 10), c["still"].get("max_motion_heights", 0.05))
    if "rule" in c:
        try:
            return RuleCond(Rule(c["rule"]), c["rule"])
        except RuleError as e:
            raise Unsupported(f"правило: {e}") from None

    block = c.get("block")
    model = c.get("model")
    if block == "person_sequence":
        bm = env.behavior.get(model)
        # класса нет в модели (или он выключен в конфиге узла) — работает fallback_rule
        if bm is not None and c.get("class") in getattr(bm, "enabled", bm.classes):
            return BehaviorCond(bm, c["class"], c.get("min_score", 0.7))
        if c.get("fallback_rule"):
            try:
                return RuleCond(Rule(c["fallback_rule"]), f"{c.get('class')}≈({c['fallback_rule']})")
            except RuleError as e:
                raise Unsupported(f"fallback_rule: {e}") from None
        raise Unsupported(f"нет модели {model} (класс {c.get('class')}) и нет fallback_rule")
    if block == "bodypart_crop":
        if model in env.models:
            raise Unsupported(f"модель {model} пока не подключена к конвейеру")
        if c.get("class") == "phone" and env.phone_classes & env.det_classes:
            return PhoneNearHand(c.get("part", "hand"), c.get("min_person_height_px", 0), c.get("max_dist_h", 0.15))
        raise Unsupported(f"нет модели {model} (класс {c.get('class')})")
    if block == "pair_interaction":
        if c.get("other") == "vehicle":
            if not env.vehicle_classes & env.det_classes:
                raise Unsupported("в детекторе нет классов техники")
            return VehicleNear(c.get("max_distance_h", 1.0), c.get("other_min_speed", 0.0))
        if c.get("fallback_rule"):
            try:
                Rule(c["fallback_rule"])
            except RuleError as e:
                raise Unsupported(f"нет модели {model}; fallback_rule: {e}") from None
        raise Unsupported(f"пара «человек — {c.get('other')}»: нет модели {model}")
    if block == "object_zone":
        missing = set(c.get("classes") or []) - env.det_classes
        raise Unsupported(f"в детекторе нет классов {', '.join(sorted(missing))}" if missing
                          else "цели object пока не поддержаны")
    raise Unsupported(f"неизвестный блок: {block or list(c)}")


# ---------------------------------------------------------------- расписание, подтверждение, дедупликация

class Schedule:
    """`always` или интервалы "07:55-12:00" (через запятую или списком; через полночь — "22:00-06:00")."""

    def __init__(self, spec, tz: str = "Europe/Moscow"):
        self.tz = ZoneInfo(tz)
        self.spec = spec
        self.always = spec in (None, "", "always", True)
        self.intervals: list[tuple[int, int]] = []
        if not self.always:
            parts = spec if isinstance(spec, list) else str(spec).split(",")
            for part in parts:
                a, b = (x.strip() for x in str(part).split("-"))
                self.intervals.append((self._min(a), self._min(b)))

    @staticmethod
    def _min(hhmm: str) -> int:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)

    def active(self, ts: float) -> bool:
        if self.always:
            return True
        t = datetime.fromtimestamp(ts, self.tz)
        m = t.hour * 60 + t.minute + t.second / 60
        for s, e in self.intervals:
            if (s <= m < e) if s <= e else (m >= s or m < e):
                return True
        return False


class Confirmer:
    """Подтверждение во времени: скользящие окна длиной window_sec с шагом stride_sec (по умолчанию 0,5 с —
    как behavior_v1 раз в 0,5 с на трек). Окно положительное, если в нём не меньше min_ratio положительных кадров.
    Нарушение подтверждено, когда положительны n из последних m окон и это держится hold_sec секунд.
    Первое окно оценивается, когда оно заполнено: трек прожил window_sec."""

    def __init__(self, window_sec: float = 1.0, n: int = 2, m: int = 3, hold_sec: float = 0.0,
                 stride_sec: float | None = None, min_ratio: float = 0.5):
        self.window, self.n, self.m, self.hold = float(window_sec), int(n), int(m), float(hold_sec or 0)
        self.stride = float(stride_sec or min(0.5, self.window / 2))
        self.min_ratio = float(min_ratio)
        self.frames: deque = deque()
        self.windows: deque = deque(maxlen=self.m)
        self.next_eval: float | None = None
        self.started_ts: float | None = None   # первый кадр-кандидат текущего эпизода
        self.hold_from: float | None = None
        self.confirmed = False
        self.satisfied = False

    def update(self, ts: float, positive: bool, score: float = 1.0) -> bool:
        """True — ровно один раз за эпизод, в момент подтверждения."""
        if self.next_eval is None:
            self.next_eval = ts + self.window
        self.frames.append((ts, bool(positive), float(score)))
        span = self.window + (self.m - 1) * self.stride
        while self.frames and self.frames[0][0] <= ts - span - 1e-9:
            self.frames.popleft()
        if positive and self.started_ts is None:
            self.started_ts = ts
        if ts + 1e-9 < self.next_eval:
            return False
        while self.next_eval <= ts + 1e-9:
            self.next_eval += self.stride
        win = [f for f in self.frames if f[0] > ts - self.window + 1e-9]
        pos = sum(1 for f in win if f[1])
        self.windows.append(pos > 0 and pos / len(win) >= self.min_ratio)
        self.satisfied = sum(self.windows) >= self.n
        if self.satisfied:
            if self.hold_from is None:
                self.hold_from = ts
            if not self.confirmed and ts - self.hold_from >= self.hold - 1e-9:
                self.confirmed = True
                return True
        else:
            self.hold_from = None
            if not any(self.windows):          # ни одного положительного окна из m — эпизод закончился
                self.started_ts = None
                self.confirmed = False
        return False

    def confidence(self) -> float:
        s = [f[2] for f in self.frames if f[1]]
        return float(np.clip(np.mean(s), 0, 1)) if s else 0.0


class Deduper:
    """per: track — одно событие на трек; per: zone — одно открытое событие на зону;
    cooldown_sec — блокировка повторов по ключу «камера + зона + тип» (в том числе после смены ID трека)."""

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.fired: set[tuple[str, int]] = set()
        self.until: dict[tuple, float] = {}
        self.open_keys: set[tuple] = set()

    def key(self, card_id: str, zone: str | None) -> tuple:
        return (self.camera_id, zone, card_id)

    def blocked(self, card: "CompiledCard", track_id: int, zone: str | None, ts: float) -> str | None:
        k = self.key(card.id, zone)
        if card.dedup_per == "track" and (card.id, track_id) in self.fired:
            return "уже было событие на этот трек"
        if card.dedup_per == "zone" and k in self.open_keys:
            return "по зоне уже открыто событие"
        if ts < self.until.get(k, float("-inf")):
            return f"блокировка повторов ещё {self.until[k] - ts:.0f} с"
        return None

    def mark(self, card: "CompiledCard", track_id: int, zone: str | None, ts: float) -> None:
        k = self.key(card.id, zone)
        self.fired.add((card.id, track_id))
        self.until[k] = ts + card.cooldown_sec
        if card.dedup_per == "zone":
            self.open_keys.add(k)

    def release(self, card: "CompiledCard", zone: str | None) -> None:
        self.open_keys.discard(self.key(card.id, zone))


# ---------------------------------------------------------------- карточка

@dataclass
class CompiledCard:
    id: str
    name: str
    categories: list[str]
    severity: str
    target: str
    cond: Cond
    followups: list[Still]
    confirm: dict
    dedup_per: str
    cooldown_sec: float
    include: list[str]
    exclude: list[str]
    schedule: Schedule
    notify: list[str]
    evidence: dict
    status: str

    @property
    def needs(self) -> frozenset:
        return self.cond.needs

    def new_confirmer(self) -> Confirmer:
        n, m = self.confirm.get("n_of_m", [2, 3])
        return Confirmer(self.confirm.get("window_sec", 1.0), n, m, self.confirm.get("hold_sec", 0),
                         self.confirm.get("stride_sec"), self.confirm.get("min_ratio", 0.5))

    def zone_gate(self, scene: Scene, p: Person) -> tuple[bool, str | None]:
        s = p.hist.last
        if self.exclude and scene.zone_at(body_point(s, "feet"), self.exclude):
            return False, None
        if self.include:
            z = scene.zone_at(body_point(s, "feet") + body_point(s, "wrists") + body_point(s, "center"), self.include)
            return (z is not None), (z.name if z else None)
        return True, None


def load_cards(patterns_dir: str | Path) -> dict[str, dict]:
    cards = {}
    for f in sorted(Path(patterns_dir).glob("*.yaml")):
        with open(f, encoding="utf-8") as fh:
            c = yaml.safe_load(fh)
        if isinstance(c, dict) and c.get("id"):
            c["_file"] = f.name
            cards[c["id"]] = c
    return cards


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def compile_card(raw: dict, env: CompileEnv, overrides: dict | None = None, tz: str = "Europe/Moscow") -> CompiledCard:
    c = _deep_merge(raw, overrides or {})
    target = c.get("target", "person")
    if target not in ("person", "pair"):
        raise Unsupported(f"цель {target}: нужна своя модель yolo-det (огонь/дым)")
    conds = c.get("conditions") or []
    main = [x for x in conds if not (isinstance(x, dict) and "still" in x)]
    stills = [x for x in conds if isinstance(x, dict) and "still" in x]
    if not main:                                   # неподвижность — единственное условие: она и есть нарушение
        main, stills = stills, []
    cond = AllOf([compile_condition(x, env) for x in main])     # сначала модели — это главная причина пропуска
    followups = [compile_condition(x, env) for x in stills]
    zones = c.get("zones") or {}
    include, exclude = list(zones.get("include") or []), list(zones.get("exclude") or [])
    if include and not any(env.scene.has_zone_tag(t) for t in include):
        raise Unsupported(f"на камере нет зон {', '.join(include)}")
    dedup = c.get("dedup") or {}
    return CompiledCard(
        id=c["id"], name=c.get("name", c["id"]), categories=list(c.get("category") or []),
        severity=c.get("severity", "warning"), target=target, cond=cond, followups=followups,
        confirm=dict(c.get("confirm") or {}), dedup_per=dedup.get("per", "track"),
        cooldown_sec=float(dedup.get("cooldown_sec", 60)), include=include, exclude=exclude,
        schedule=Schedule(c.get("schedule", "always"), tz), notify=list(c.get("notify") or []),
        evidence=dict(c.get("evidence") or {}), status=c.get("status", ""),
    )


# ---------------------------------------------------------------- движок камеры

@dataclass
class Action:
    """Что отправить в журнал: create — новое событие, update — дописать (второй этап), end — закрыть."""
    kind: str
    key: str
    card: CompiledCard
    track_id: int
    ts: float                      # время кадра, на котором принято решение
    started_ts: float | None = None
    ended_ts: float | None = None
    zone: str | None = None
    confidence: float = 0.0
    bbox: list | None = None
    other_bbox: list | None = None
    other_label: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class _State:
    confirmer: Confirmer
    event_key: str | None = None
    ended: bool = False
    last_pos_ts: float | None = None
    info: dict = field(default_factory=dict)
    zone: str | None = None
    done_followups: set = field(default_factory=set)


class Engine:
    def __init__(self, camera_id: str, cards: list[CompiledCard], end_after_sec: float = 10.0):
        self.camera_id = camera_id
        self.cards = cards
        self.end_after = end_after_sec
        self.dedup = Deduper(camera_id)
        self.states: dict[tuple[str, int], _State] = {}
        self._seq = itertools.count(1)

    @property
    def needs(self) -> frozenset:
        return frozenset().union(*(c.needs for c in self.cards)) if self.cards else frozenset()

    def _state(self, card: CompiledCard, tid: int) -> _State:
        st = self.states.get((card.id, tid))
        if st is None:
            st = self.states[(card.id, tid)] = _State(card.new_confirmer())
        return st

    def step(self, ctx: FrameContext) -> list[Action]:
        acts: list[Action] = []
        for card in self.cards:
            on = card.schedule.active(ctx.ts)
            for p in ctx.persons:
                st = self._state(card, p.track_id)
                positive, info = False, {}
                if on:
                    gate, zone = card.zone_gate(ctx.scene, p)
                    if gate:
                        positive, info = card.cond(ctx, p)
                        if positive and zone and "zone" not in info:
                            info["zone"] = zone
                score = p.conf * info.get("score", 1.0) if positive else 0.0
                fired = st.confirmer.update(ctx.ts, positive, score)
                if positive:
                    st.last_pos_ts, st.info = ctx.ts, info
                if fired and st.event_key is None:
                    acts += self._try_create(card, p, st, ctx)
                if st.event_key and not st.ended:
                    acts += self._after_create(card, p, st, ctx)
        return acts

    def _try_create(self, card, p, st, ctx) -> list[Action]:
        zone = st.info.get("zone")
        reason = self.dedup.blocked(card, p.track_id, zone, ctx.ts)
        if reason:
            log.info("[%s] %s, трек %d: подтверждено, но не отправлено — %s", self.camera_id, card.id, p.track_id, reason)
            return []
        self.dedup.mark(card, p.track_id, zone, ctx.ts)
        st.event_key = f"{self.camera_id}:{card.id}:{p.track_id}:{next(self._seq)}"
        st.zone = zone
        extra = {"conditions": card.cond.label, "features": st.info.get("features", {}),
                 "rules": st.info.get("rules", []), "notify": card.notify, "card_status": card.status,
                 "evidence": card.evidence, "stage": "confirmed"}
        if card.followups:
            extra["stage"] = "detected"
            extra["followup"] = {"condition": " И ".join(f.label for f in card.followups), "status": "waiting"}
        if st.info.get("line"):
            extra["line"] = st.info["line"]
        return [Action("create", st.event_key, card, p.track_id, ctx.ts, started_ts=st.confirmer.started_ts or ctx.ts,
                       zone=zone, confidence=round(st.confirmer.confidence(), 3), bbox=_box(p.xyxy),
                       other_bbox=st.info.get("other_bbox"), other_label=st.info.get("other_label"), extra=extra)]

    def _after_create(self, card, p, st, ctx) -> list[Action]:
        acts = []
        for i, fu in enumerate(card.followups):
            if i in st.done_followups:
                continue
            ok, info = fu(ctx, p)
            if ok:
                st.done_followups.add(i)
                acts.append(Action("update", st.event_key, card, p.track_id, ctx.ts, zone=st.zone, bbox=_box(p.xyxy),
                                   extra={"stage": "motionless", "followup": {"condition": fu.label, "status": "confirmed",
                                                                              **info}}))
        if not card.followups and st.last_pos_ts is not None and ctx.ts - st.last_pos_ts >= self.end_after:
            acts.append(self._end(card, st, p.track_id, ctx.ts))     # нарушение закончилось, трек ещё в кадре
        return acts

    def _end(self, card, st, tid, ts) -> Action:
        st.ended = True
        self.dedup.release(card, st.zone)
        return Action("end", st.event_key, card, tid, ts, ended_ts=st.last_pos_ts or ts, zone=st.zone)

    def track_lost(self, tid: int, ts: float) -> list[Action]:
        """Трек ушёл из кадра: закрыть его открытые события и забыть состояние."""
        acts = []
        for card in self.cards:
            st = self.states.pop((card.id, tid), None)
            if st and st.event_key and not st.ended:
                acts.append(self._end(card, st, tid, ts))
        return acts

    def close(self, ts: float) -> list[Action]:
        tids = {tid for (_, tid) in self.states}
        return [a for tid in tids for a in self.track_lost(tid, ts)]


def build_engine(camera: dict, cards_raw: dict[str, dict], env: CompileEnv, tz: str,
                 end_after_sec: float = 10.0) -> tuple[Engine, list[tuple[str, str]]]:
    """Собрать движок камеры из разрешённых в её конфиге карточек. Возвращает и список пропущенных с причиной."""
    cards, skipped = [], []
    for entry in camera.get("patterns") or []:
        pid, over = (entry, {}) if isinstance(entry, str) else (entry["id"], {k: v for k, v in entry.items() if k != "id"})
        over = _deep_merge(over, (camera.get("overrides") or {}).get(pid, {}))
        raw = cards_raw.get(pid)
        if raw is None:
            skipped.append((pid, "нет такой карточки в patterns/"))
            continue
        try:
            cards.append(compile_card(raw, env, over, tz))
        except Unsupported as e:
            skipped.append((pid, str(e)))
    return Engine(camera["id"], cards, end_after_sec), skipped
