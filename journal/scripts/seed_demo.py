"""Демо-данные для дашбордов: ~30 дней событий по 8 камерам и 5 участкам.

Паттерны берутся из YAML-карточек (../patterns), так что журнал и трафареты согласованы.
В данные заложены закономерности из сквозного примера кейса: склад отгрузки даёт
львиную долю нарушений, они сосредоточены в первые часы смены и в ночную смену,
а в последние 10 дней (после инструктажа) их становится заметно меньше.

    python -m scripts.seed_demo --reset
"""
import argparse
import glob
import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app.db import Base, SessionLocal, engine
from app.metrics import seed_demo_metrics
from app.models import Camera, Event, StatusChange

CAMERAS = [
    ("cam-01", "Склад отгрузки — ворота 1", "Склад отгрузки"),
    ("cam-02", "Склад отгрузки — стеллажи", "Склад отгрузки"),
    ("cam-07", "Погрузочная площадка", "Склад отгрузки"),
    ("cam-03", "Механический цех — станки", "Механический цех"),
    ("cam-04", "Механический цех — проход", "Механический цех"),
    ("cam-05", "Сварочный участок", "Сварочный участок"),
    ("cam-06", "Электроучасток — щитовая", "Электроучасток"),
    ("cam-08", "Периметр — ограждение", "Периметр"),
]

# Какие паттерны характерны для каких участков (вес)
AFFINITY = {
    "Склад отгрузки": {"vehicle_proximity": 5, "run": 4, "danger_zone": 4, "carry_no_ppe": 3, "carry_wrong": 2,
                       "phone": 2, "no_helmet": 2, "fall": 1},
    "Механический цех": {"phone": 4, "no_helmet": 3, "fall": 1, "crawl_under": 2, "run": 1, "fight": 1},
    "Сварочный участок": {"fire_smoke": 2, "smoking": 3, "no_helmet": 2, "phone": 1},
    "Электроучасток": {"bare_hands_wires": 4, "phone": 1, "height_no_harness": 2},
    "Периметр": {"climb_fence": 3, "smoking": 3, "fire_smoke": 1},
}
AREA_SHARE = {"Склад отгрузки": 0.42, "Механический цех": 0.25, "Сварочный участок": 0.13,
              "Электроучасток": 0.1, "Периметр": 0.1}
# Насколько оператор подтверждает тревоги паттерна (для демо precision)
PRECISION = {"fall": 0.93, "vehicle_proximity": 0.9, "run": 0.86, "phone": 0.8, "smoking": 0.75,
             "fire_smoke": 0.85, "no_helmet": 0.9, "bare_hands_wires": 0.84}
# Сколько секунд от начала до тревоги (подтверждение во времени)
CONFIRM_SEC = {"fall": (10, 13), "vehicle_proximity": (0.5, 1.2), "fire_smoke": (2, 4)}

FALLBACK_PATTERNS = {
    "run": ("Бег и резкие перемещения в зоне техники", ["behavior", "zone"], "warning"),
    "vehicle_proximity": ("Опасное сближение с движущейся техникой", ["interaction"], "critical"),
    "phone": ("Телефон при работе с оборудованием или при перемещении", ["attribute", "behavior"], "warning"),
    "fall": ("Падение, неподвижность, признаки потери сознания", ["behavior"], "critical"),
}


def load_patterns(folder: str) -> dict:
    try:
        import yaml
        out = {}
        for path in sorted(glob.glob(os.path.join(folder, "*.yaml"))):
            card = yaml.safe_load(open(path, encoding="utf-8"))
            out[card["id"]] = (card["name"], card.get("category", []), card["severity"])
        return out or FALLBACK_PATTERNS
    except Exception:
        return FALLBACK_PATTERNS


def hour_weight(area: str, hour: int) -> float:
    w = 1.0
    if hour in (8, 9, 20, 21):  # первые часы смены
        w *= 2.2
    if (hour >= 20 or hour < 8) and area == "Склад отгрузки":  # ночная смена на складе
        w *= 1.6
    if 0 <= hour < 6:
        w *= 0.6
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--per-day", type=int, default=60)
    ap.add_argument("--patterns", default=os.path.join(os.path.dirname(__file__), "..", "..", "patterns"))
    ap.add_argument("--reset", action="store_true", help="удалить все таблицы и создать заново")
    ap.add_argument("--seed", type=int, default=94)
    ap.add_argument("--no-metrics", action="store_true", help="не создавать демо-метрики узлов LinQ")
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    patterns = load_patterns(args.patterns)
    if args.reset:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    plant_tz = ZoneInfo(os.getenv("PLANT_TZ", "Europe/Moscow"))  # часы смен — по местному времени
    now = datetime.now(timezone.utc)
    start = (now.astimezone(plant_tz) - timedelta(days=args.days)).replace(hour=0, minute=0, second=0, microsecond=0)
    users = {"operator": ["оператор-1", "оператор-2"], "master": ["мастер-склад", "мастер-цех"]}

    with SessionLocal() as s:
        for cid, name, area in CAMERAS:
            s.merge(Camera(id=cid, name=name, area=area, source=f"rtsp://demo/{cid}"))
        s.flush()

        n = 0
        for d in range(args.days):
            day = start + timedelta(days=d)
            for _ in range(int(args.per_day * rnd.uniform(0.7, 1.3))):
                area = rnd.choices(list(AREA_SHARE), weights=list(AREA_SHARE.values()))[0]
                if area == "Склад отгрузки" and d >= args.days - 10 and rnd.random() < 0.45:
                    continue  # эффект инструктажа: последние 10 дней на складе меньше нарушений
                hour = rnd.choices(range(24), weights=[hour_weight(area, h) for h in range(24)])[0]
                aff = {p: w for p, w in AFFINITY[area].items() if p in patterns}
                if not aff:
                    continue
                pid = rnd.choices(list(aff), weights=list(aff.values()))[0]
                pname, cats, sev = patterns[pid]
                cam = rnd.choice([c for c in CAMERAS if c[2] == area])

                started = (day.replace(hour=hour) + timedelta(seconds=rnd.uniform(0, 3599))).astimezone(timezone.utc)
                if started > now:
                    continue
                lo, hi = CONFIRM_SEC.get(pid, (1.0, 3.0))
                confirmed = started + timedelta(seconds=rnd.uniform(lo, hi))
                received = confirmed + timedelta(milliseconds=max(80, rnd.gauss(350, 120)))
                ended = confirmed + timedelta(seconds=rnd.uniform(4, 120))

                age_h = (now - started).total_seconds() / 3600
                if age_h < 2 and rnd.random() < 0.7:
                    status = "new"
                elif rnd.random() < PRECISION.get(pid, 0.85):
                    status = rnd.choices(["confirmed", "sent_to_master", "resolved"], weights=[5, 3, 2])[0]
                else:
                    status = "false_alarm"

                e = Event(
                    event_uid=uuid.uuid4().hex, pattern_id=pid, pattern_name=pname, categories=list(cats),
                    severity=sev, camera_id=cam[0], area=area, zone=None,
                    track_id=rnd.randint(1, 500), started_at=started, confirmed_at=confirmed,
                    ended_at=ended, frame_ts=confirmed, received_at=received,
                    confidence=round(rnd.uniform(0.6, 0.98), 3), status=status, extra={"demo": True},
                )
                if status != "new":
                    who = rnd.choice(users["master" if status == "sent_to_master" else "operator"])
                    at = confirmed + timedelta(seconds=rnd.uniform(10, 240))
                    e.reviewed_by, e.reviewed_at = who, at
                    e.history.append(StatusChange(from_status="new", to_status=status, user=who, at=at))
                s.add(e)
                n += 1
        s.commit()
        m = 0 if args.no_metrics else seed_demo_metrics(s)
    print(f"Создано событий: {n}, камер: {len(CAMERAS)}, паттернов: {len(patterns)}, окон метрик узлов: {m}")


if __name__ == "__main__":
    main()
