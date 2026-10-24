"""Логика движка без моделей: подтверждение N из M, дедупликация, зоны, расписание, правила и карточки MVP."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from core.engine import CompileEnv, Confirmer, Schedule, build_engine, compile_card, load_cards
from core.features import Rule, RuleError
from core.geometry import Scene, point_in_polygon, segments_intersect

from .helpers import Sim, rect

ROOT = Path(__file__).resolve().parents[2]
CARDS = load_cards(ROOT / "patterns")
MSK = ZoneInfo("Europe/Moscow")
T0 = datetime(2026, 10, 24, 10, 0, tzinfo=MSK).timestamp()   # 10:00 МСК — внутри расписания danger_zone
FPS = 10


def scene():
    return Scene.from_config(
        [{"name": "Стеллажи", "tags": ["rack_forklift"], "polygon": rect(300, 400, 600, 700)},
         {"name": "Проезд", "tags": ["tech_area"], "polygon": rect(0, 300, 1920, 1080)},
         {"name": "Комната отдыха", "tags": ["rest_area"], "polygon": rect(1500, 300, 1900, 1080)},
         {"name": "У ограждения", "tags": ["fence_area"], "polygon": rect(900, 300, 1100, 1080)}],
        [{"name": "Ограждение", "tags": ["fence"], "points": [[1000, 300], [1000, 1080]]}])


def engine(*ids, overrides=None, env_kw=None):
    env = CompileEnv(scene(), det_classes={"car", "truck", "cell phone"}, **(env_kw or {}))
    cam = {"id": "cam-t", "patterns": list(ids), "overrides": overrides or {}}
    eng, skipped = build_engine(cam, CARDS, env, "Europe/Moscow", end_after_sec=5)
    return eng, dict(skipped)


def run(eng, sim, seconds, people_at, vehicles_at=None, phones_at=None, t_start=0.0):
    """people_at(t) -> [(tid, cx, feet_y, angle)]; возвращает действия движка с моментом (t от T0)."""
    acts = []
    n = int(round(seconds * FPS))
    for i in range(n):
        t = t_start + i / FPS
        ctx = sim.ctx(T0 + t, people_at(t), vehicles_at(t) if vehicles_at else (), phones_at(t) if phones_at else ())
        for a in eng.step(ctx):
            acts.append((round(t, 2), a))
    return acts


# ---------------------------------------------------------------- подтверждение

def feed(c: Confirmer, pattern, fps=FPS):
    """pattern(t) -> bool; возвращает время подтверждения или None."""
    for i in range(int(10 * fps)):
        t = i / fps
        if c.update(t, pattern(t)):
            return t
    return None


def test_confirm_n_of_m_needs_full_windows():
    c = Confirmer(window_sec=1.0, n=2, m=3)            # шаг окна 0,5 с
    t = feed(c, lambda t: True)
    assert t == pytest.approx(1.5)                     # первое окно в 1,0 с, второе — в 1,5 с
    assert c.started_ts == 0.0


def test_confirm_ignores_sporadic_frames():
    c = Confirmer(window_sec=1.0, n=2, m=3)
    assert feed(c, lambda t: round(t * FPS) % 4 == 0) is None    # каждый 4-й кадр — шум, окна не набираются


def test_confirm_hold_sec_delays_and_resets():
    c = Confirmer(window_sec=1.0, n=2, m=3, hold_sec=2.0)
    assert feed(c, lambda t: True) == pytest.approx(3.5)        # 1,5 с на N из M + 2 с удержания
    c2 = Confirmer(window_sec=1.0, n=2, m=3, hold_sec=2.0)
    assert feed(c2, lambda t: t < 2.5 or t > 6) == pytest.approx(9.0)   # перерыв сбрасывает удержание


def test_confirm_fires_once_per_episode():
    c = Confirmer(window_sec=1.0, n=2, m=3)
    fired = [i / FPS for i in range(80) if c.update(i / FPS, True)]
    assert fired == [pytest.approx(1.5)]


# ---------------------------------------------------------------- геометрия, правила, расписание

def test_point_in_polygon_and_line_cross():
    poly = np.array(rect(0, 0, 10, 10), np.float32)
    assert point_in_polygon((5, 5), poly) and not point_in_polygon((15, 5), poly)
    tri = np.array([[0, 0], [10, 0], [0, 10]], np.float32)
    assert point_in_polygon((2, 2), tri) and not point_in_polygon((8, 8), tri)
    assert segments_intersect((0, 5), (10, 5), (5, 0), (5, 10))
    assert not segments_intersect((0, 5), (4, 5), (5, 0), (5, 10))


def test_rule_safe_eval():
    r = Rule("torso_angle_deg > 60 and center_drop_heights > 0.4")
    assert r({"torso_angle_deg": 80, "center_drop_heights": 0.5})
    assert not r({"torso_angle_deg": 80, "center_drop_heights": None})     # нет признака — ложь
    with pytest.raises(RuleError):
        Rule("facing_each_other")                                           # неизвестный признак
    with pytest.raises(RuleError):
        Rule("__import__('os').system('x')")                               # вызовы запрещены


def test_schedule():
    s = Schedule("07:55-12:00")
    at = lambda h, m: datetime(2026, 10, 24, h, m, tzinfo=MSK).timestamp()  # noqa: E731
    assert s.active(at(10, 0)) and not s.active(at(12, 30)) and not s.active(at(7, 0))
    night = Schedule("22:00-06:00")
    assert night.active(at(23, 0)) and night.active(at(3, 0)) and not night.active(at(12, 0))
    assert Schedule("always").active(at(3, 0))


# ---------------------------------------------------------------- сборка карточек

def test_unsupported_cards_are_skipped_with_reason():
    eng, skipped = engine("fall", "run", "danger_zone", "vehicle_proximity", "phone",
                          "fire_smoke", "crawl_under", "no_helmet", "fight", "climb_fence")
    enabled = {c.id for c in eng.cards}
    assert {"fall", "run", "danger_zone", "vehicle_proximity", "phone", "climb_fence"} <= enabled
    assert "yolo-det" in skipped["fire_smoke"]
    assert "under_conveyor" in skipped["crawl_under"]
    assert "crop_attr_v1" in skipped["no_helmet"]
    assert "pair_v1" in skipped["fight"]


def test_fall_still_is_second_stage():
    card = compile_card(CARDS["fall"], CompileEnv(scene()))
    assert card.followups and card.followups[0].seconds == 10
    assert "still" not in card.cond.label and "неподвиж" not in card.cond.label


# ---------------------------------------------------------------- карточки на синтетических треках

def test_danger_zone_one_event_then_end():
    eng, _ = engine("danger_zone")
    sim = Sim(scene())
    # 2 с вне зоны, 6 с в зоне, 7 с снаружи
    acts = run(eng, sim, 15, lambda t: [(1, 200 if t < 2 or t >= 8 else 450, 650, 0)])
    creates = [(t, a) for t, a in acts if a.kind == "create"]
    assert len(creates) == 1
    t, a = creates[0]
    assert a.zone == "Стеллажи" and a.card.id == "danger_zone"
    assert 2 + 1.0 + 2 <= t <= 2 + 1.0 + 2 + 0.6            # N из M (окна наполовину в зоне, ~1 с) + hold 2 с
    assert a.started_ts == pytest.approx(T0 + 2.0)
    ends = [(t, a) for t, a in acts if a.kind == "end"]
    assert len(ends) == 1 and ends[0][1].ended_ts == pytest.approx(T0 + 7.9)


def test_danger_zone_respects_card_schedule():
    eng, _ = engine("danger_zone")                            # в карточке 07:55-12:00
    sim = Sim(scene())
    evening = datetime(2026, 10, 24, 18, 0, tzinfo=MSK).timestamp() - T0
    acts = run(eng, sim, 8, lambda t: [(1, 450, 650, 0)], t_start=evening)
    assert not [a for _, a in acts if a.kind == "create"]
    eng2, _ = engine("danger_zone", overrides={"danger_zone": {"schedule": "always"}})
    acts = run(eng2, Sim(scene()), 8, lambda t: [(1, 450, 650, 0)], t_start=evening)
    assert len([a for _, a in acts if a.kind == "create"]) == 1


def test_dedup_per_track_and_cooldown_by_zone():
    eng, _ = engine("danger_zone")                            # dedup: per track, cooldown 120 с
    sim = Sim(scene())
    # трек 1 стоит в зоне, выходит, возвращается — второго события нет (одно событие на трек)
    acts = run(eng, sim, 30, lambda t: [(1, 450 if (t < 6 or 14 <= t < 22) else 200, 650, 0)])
    assert len([a for _, a in acts if a.kind == "create"]) == 1
    # трек 2 (например, тот же человек после смены ID) в той же зоне через 30 с — блокировка повторов
    acts = run(eng, sim, 8, lambda t: [(2, 450, 650, 0)], t_start=30)
    assert not [a for _, a in acts if a.kind == "create"]
    # после cooldown — снова можно
    acts = run(eng, sim, 8, lambda t: [(3, 450, 650, 0)], t_start=200)
    assert len([a for _, a in acts if a.kind == "create"]) == 1


def test_track_lost_ends_event():
    eng, _ = engine("danger_zone")
    sim = Sim(scene())
    acts = run(eng, sim, 6, lambda t: [(1, 450, 650, 0)])
    assert [a.kind for _, a in acts] == ["create"]
    ends = eng.track_lost(1, T0 + 8)
    assert [a.kind for a in ends] == ["end"] and ends[0].key == acts[0][1].key


def test_run_fires_on_running_not_walking():
    eng, _ = engine("run")                                     # speed_heights_per_sec > 1.5 в tech_area
    sim = Sim(scene())
    walk = run(eng, sim, 5, lambda t: [(1, 100 + 160 * t, 650, 0)])          # 0,8 роста/с
    assert not [a for _, a in walk if a.kind == "create"]
    eng, _ = engine("run")
    sim = Sim(scene())
    sprint = run(eng, sim, 4, lambda t: [(1, 100 + 400 * t, 650, 0)])       # 2 роста/с
    creates = [(t, a) for t, a in sprint if a.kind == "create"]
    assert len(creates) == 1 and creates[0][0] <= 3.0
    assert creates[0][1].extra["features"]["speed_heights_per_sec"] > 1.5


def test_fall_alarm_fast_then_motionless_update():
    eng, _ = engine("fall")
    sim = Sim(scene())

    def person(t):
        if t < 3:
            return [(1, 800, 650, 0)]                          # стоит
        if t < 3.5:
            return [(1, 800, 650, 90 * (t - 3) / 0.5)]         # падает за 0,5 с
        return [(1, 800, 650, 90)]                             # лежит неподвижно

    acts = run(eng, sim, 20, person)
    kinds = [(t, a.kind, a.extra.get("stage")) for t, a in acts]
    creates = [k for k in kinds if k[1] == "create"]
    updates = [k for k in kinds if k[1] == "update"]
    assert len(creates) == 1 and creates[0][2] == "detected"
    assert creates[0][0] - 3.0 <= 3.0                          # тревога в пределах нескольких секунд после падения
    assert len(updates) == 1 and updates[0][2] == "motionless"
    assert 3.5 + 9 <= updates[0][0] <= 3.5 + 11                # неподвижность 10 с — вторым этапом
    assert not [k for k in kinds if k[1] == "end"]             # лежит — событие не закрываем


def test_fall_not_triggered_by_bending_or_sitting_slowly():
    eng, _ = engine("fall")
    sim = Sim(scene())
    acts = run(eng, sim, 10, lambda t: [(1, 800, 650, min(45, 10 * t))])     # медленный наклон до 45°
    assert not [a for _, a in acts if a.kind == "create"]


def test_vehicle_proximity_needs_moving_vehicle():
    eng, _ = engine("vehicle_proximity")
    sim = Sim(scene())
    person = lambda t: [(1, 800, 650, 0)]                      # noqa: E731
    parked = lambda t: [(7, np.array([900, 500, 1200, 680], np.float32), 0.9, "truck", 0.0)]   # noqa: E731
    assert not [a for _, a in run(eng, sim, 4, person, parked) if a.kind == "create"]
    eng, _ = engine("vehicle_proximity")
    sim = Sim(scene())
    moving = lambda t: [(7, np.array([900, 500, 1200, 680], np.float32), 0.9, "truck", 120.0)]   # noqa: E731
    acts = [(t, a) for t, a in run(eng, sim, 4, person, moving) if a.kind == "create"]
    assert len(acts) == 1 and acts[0][0] <= 1.5                # критичный сценарий — быстро
    assert acts[0][1].other_label == "truck #7" and acts[0][1].other_bbox


def test_phone_near_wrist_while_moving():
    eng, _ = engine("phone")                                   # телефон у кисти И (зона equipment ИЛИ движется)
    sim = Sim(scene())
    walk = lambda t: [(1, 200 + 100 * t, 650, 0)]              # noqa: E731  0,5 роста/с
    # правая кисть стоящего человека: x = cx + 0.13H, y = feet - 0.55H
    near = lambda t: [([200 + 100 * t + 20, 530, 200 + 100 * t + 40, 560], 0.6)]   # noqa: E731
    far = lambda t: [([1300, 100, 1320, 130], 0.6)]            # noqa: E731
    assert not [a for _, a in run(eng, sim, 8, walk, None, far) if a.kind == "create"]
    eng, _ = engine("phone")
    sim = Sim(scene())
    creates = [a for _, a in run(eng, sim, 8, walk, None, near) if a.kind == "create"]
    assert len(creates) == 1 and creates[0].other_label == "cell phone"


def test_climb_fence_cross_line():
    eng, skipped = engine("climb_fence")                      # behavior_v1 нет — остаётся ветка cross_line
    assert not skipped
    sim = Sim(scene())
    acts = run(eng, sim, 6, lambda t: [(1, 940 + 30 * t, 650, 0)])   # пересекает x = 1000 на 2-й секунде
    creates = [a for _, a in acts if a.kind == "create"]
    assert len(creates) == 1 and creates[0].extra["line"] == "Ограждение"
    eng, _ = engine("climb_fence")
    acts = run(eng, Sim(scene()), 6, lambda t: [(1, 920 + 10 * t, 650, 0)])   # подошёл, но не пересёк
    assert not [a for _, a in acts if a.kind == "create"]
