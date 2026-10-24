"""behavior_v1: окно скелета, подключение модели к карточкам, разметка окон и обучение → ONNX → конвейер."""
from pathlib import Path

import numpy as np
import pytest

from core.behavior import FEATURES, T, window_tensor
from core.engine import BehaviorCond, CompileEnv, RuleCond, compile_card, load_cards
from core.features import TrackHistory
from core.tools.behavior_data import label_window

from .helpers import Sim, skeleton
from .test_engine import engine, run, scene

ROOT = Path(__file__).resolve().parents[2]
CARDS = load_cards(ROOT / "patterns")


def track(seconds: float, fps: float = 10, speed_px: float = 0.0, angle=lambda t: 0.0) -> TrackHistory:
    h = TrackHistory(1, 20.0)
    for i in range(int(seconds * fps)):
        t = i / fps
        box, k = skeleton(500 + speed_px * t, 800, 200, angle(t))
        h.add(t, box, k, 0.9)
    return h


class StubModel:
    name = "behavior_v1"
    classes = ["normal", "run", "fall", "lift_back"]

    def __init__(self, cls: str = "normal", p: float = 0.9):
        self.cls, self.p = cls, p

    def probs(self, h):
        return {c: (self.p if c == self.cls else (1 - self.p) / 3) for c in self.classes}


def test_window_shape_and_short_track():
    assert window_tensor(track(1.0)) is None                   # меньше окна — не оцениваем
    x = window_tensor(track(3.0, speed_px=300))
    assert x.shape == (T, FEATURES)
    assert x[:, 58].min() == 1.0                               # все моменты есть
    assert x[-1, 51] == 0 and x[0, 51] < -2.0                  # низ рамки 2 с назад левее на 600 px = 3 роста


def test_window_gap_drops_window():
    h = TrackHistory(1, 20.0)
    for i in range(30):
        if 5 <= i < 25:                                        # трекер потерял человека на 2 с
            continue
        box, k = skeleton(500, 800, 200)
        h.add(i / 10, box, k, 0.9)
    assert window_tensor(h) is None


def test_model_replaces_fallback_only_for_its_classes():
    env = CompileEnv(scene(), behavior={"behavior_v1": StubModel()})
    assert isinstance(compile_card(CARDS["fall"], env).cond.items[0], BehaviorCond)
    # класса lift_back в модели нет — карточка подъёма спиной остаётся на правиле-заглушке
    small = StubModel()
    small.classes = ["normal", "run", "fall"]
    env = CompileEnv(scene(), behavior={"behavior_v1": small})
    assert isinstance(compile_card(CARDS["carry_wrong"], env).cond.items[0], RuleCond)


def test_run_card_fires_from_model():
    sim = Sim(scene())
    eng, skipped = engine("run", env_kw={"behavior": {"behavior_v1": StubModel("run", 0.95)}})
    assert "run" not in skipped
    acts = run(eng, sim, 5, lambda t: [(1, 800, 600, 0)])      # стоит на месте, но модель говорит «бег»
    created = [a for t, a in acts if a.kind == "create"]
    assert len(created) == 1 and created[0].card.id == "run"
    eng, _ = engine("run", env_kw={"behavior": {"behavior_v1": StubModel("run", 0.5)}})
    assert not [a for t, a in run(eng, Sim(scene()), 5, lambda t: [(1, 800, 600, 0)]) if a.kind == "create"]


def test_label_window():
    whole = {"whole": "run", "intervals": []}
    assert label_window(5.0, 2.0, whole, 0.5) == "run"
    fall = {"whole": None, "intervals": [(3.0, 4.0, "fall")]}
    assert label_window(3.3, 2.0, fall, 0.5) is None           # задело падение на 0,3 с — спорное окно, пропуск
    assert label_window(3.6, 2.0, fall, 0.5) == "fall"
    assert label_window(5.4, 2.0, fall, 0.5) == "fall"         # падение ещё в окне на 0,6 с
    assert label_window(2.9, 2.0, fall, 0.5) == "normal"
    assert label_window(7.0, 2.0, fall, 0.5) == "normal"


def test_train_export_and_load(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from core.behavior import BehaviorModel
    from core.tools.train_behavior import export, predict, train

    rng = np.random.default_rng(0)
    X, y = [], []
    for k in range(40):
        stand = window_tensor(track(2.5, speed_px=rng.uniform(0, 100)))
        fall = window_tensor(track(2.5, angle=lambda t, s=rng.uniform(1.0, 1.5): min(90.0, max(0.0, (t - s) * 180))))
        X += [stand, fall]
        y += [0, 2]
    X, y = np.array(X), np.array(y)
    classes = ["normal", "run", "fall", "lift_back"]
    model, _ = train(X, y, len(classes), epochs=15, seed=0, log=lambda m: None)
    assert (predict(model, X).argmax(1) == y).mean() > 0.9
    out = tmp_path / "behavior_v1.onnx"
    export(model, classes, out)
    bm = BehaviorModel(out)
    assert bm.classes == classes
    h = track(2.5, angle=lambda t: min(90.0, max(0.0, (t - 1.2) * 180)))
    p = bm.probs(h)
    assert max(p, key=p.get) == "fall"
    assert bm.probs(h) is p                                    # в пределах шага 0,5 с — из кэша
