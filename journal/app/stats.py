"""Агрегаты для дашбордов. Считаются в Python поверх отфильтрованных событий:
одинаково работают на PostgreSQL и SQLite и корректно учитывают часовой пояс предприятия.
Для объёмов хакатона (десятки тысяч событий) этого достаточно; при росте — перенести в SQL."""
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .db import as_utc
from .models import TRUE_STATUSES, Event

DAY_SHIFT = (8, 20)  # дневная смена 08:00–20:00, остальное — ночная


def _ms(a: datetime | None, b: datetime | None) -> int | None:
    if a is None or b is None:
        return None
    return int((as_utc(b) - as_utc(a)).total_seconds() * 1000)


def time_to_alert_ms(e: Event) -> int | None:
    return _ms(e.started_at, e.confirmed_at)


def pipeline_latency_ms(e: Event) -> int | None:
    return _ms(e.frame_ts, e.received_at)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _local(e: Event, tz: ZoneInfo) -> datetime:
    return as_utc(e.started_at).astimezone(tz)


def shift_of(local_dt: datetime) -> str:
    return "day" if DAY_SHIFT[0] <= local_dt.hour < DAY_SHIFT[1] else "night"


def summary(events: list[Event], tz: ZoneInfo) -> dict:
    by_pattern: dict[str, dict] = {}
    for e in events:
        p = by_pattern.setdefault(e.pattern_id, {"pattern_id": e.pattern_id, "name": e.pattern_name, "count": 0})
        p["count"] += 1
    shifts = Counter(shift_of(_local(e, tz)) for e in events)
    area_shift = Counter((e.area, shift_of(_local(e, tz))) for e in events)
    return {
        "total": len(events),
        "open_critical": sum(1 for e in events if e.severity == "critical" and e.status == "new"),
        "by_pattern": sorted(by_pattern.values(), key=lambda x: -x["count"]),
        "by_camera": dict(Counter(e.camera_id for e in events).most_common()),
        "by_area": dict(Counter(e.area for e in events).most_common()),
        "by_status": dict(Counter(e.status for e in events)),
        "by_severity": dict(Counter(e.severity for e in events)),
        "by_shift": {"day": shifts.get("day", 0), "night": shifts.get("night", 0)},
        "by_area_shift": [
            {"area": a, "shift": s, "count": c} for (a, s), c in sorted(area_shift.items(), key=lambda x: -x[1])
        ],
    }


def timeseries(events: list[Event], bucket: str, tz: ZoneInfo) -> list[dict]:
    """Количество событий по часам или дням (в часовом поясе предприятия), с разбивкой по паттернам."""
    rows: dict[datetime, Counter] = defaultdict(Counter)
    for e in events:
        t = _local(e, tz)
        key = t.replace(minute=0, second=0, microsecond=0)
        if bucket == "day":
            key = key.replace(hour=0)
        rows[key][e.pattern_id] += 1
    if not rows:
        return []
    step = timedelta(days=1) if bucket == "day" else timedelta(hours=1)
    out, cur, end = [], min(rows), max(rows)
    while cur <= end:  # заполняем пропуски нулями, чтобы графики были непрерывными
        c = rows.get(cur, Counter())
        out.append({"t": cur.isoformat(), "total": sum(c.values()), "by_pattern": dict(c)})
        cur = (cur + step).astimezone(tz)
    return out


def heatmap(events: list[Event], tz: ZoneInfo, rows: str = "area") -> dict:
    """Матрица «участок (или камера, паттерн) × час суток»."""
    key = {"area": lambda e: e.area, "camera": lambda e: e.camera_id, "pattern": lambda e: e.pattern_id}[rows]
    m: dict[str, list[int]] = defaultdict(lambda: [0] * 24)
    for e in events:
        m[key(e)][_local(e, tz).hour] += 1
    labels = sorted(m, key=lambda k: -sum(m[k]))
    return {"rows": labels, "hours": list(range(24)), "values": [m[k] for k in labels]}


def quality(events: list[Event]) -> dict:
    """Точность по отметкам оператора и скорость реакции системы."""

    def block(evts: list[Event]) -> dict:
        reviewed = [e for e in evts if e.status != "new"]
        true_pos = sum(1 for e in reviewed if e.status in TRUE_STATUSES)
        false_pos = sum(1 for e in reviewed if e.status == "false_alarm")
        tta = [v for v in (time_to_alert_ms(e) for e in evts) if v is not None]
        lat = [v for v in (pipeline_latency_ms(e) for e in evts) if v is not None]
        return {
            "events": len(evts),
            "reviewed": len(reviewed),
            "true_positive": true_pos,
            "false_positive": false_pos,
            "precision": round(true_pos / (true_pos + false_pos), 3) if (true_pos + false_pos) else None,
            "time_to_alert_ms_p50": percentile(tta, 0.5),
            "time_to_alert_ms_p95": percentile(tta, 0.95),
            "pipeline_latency_ms_p50": percentile(lat, 0.5),
            "pipeline_latency_ms_p95": percentile(lat, 0.95),
        }

    groups: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        groups[e.pattern_id].append(e)
    return {
        "overall": block(events),
        "by_pattern": {pid: {"name": evts[0].pattern_name, **block(evts)} for pid, evts in groups.items()},
    }
