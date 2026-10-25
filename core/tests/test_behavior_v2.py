"""behavior_v1 на открытых данных (признаки версии 2): окно скелета и загрузка модели конвейером."""
from pathlib import Path

import numpy as np
import pytest

from core.behavior import BehaviorModel
from core.behavior_v2 import CHANNELS, T, make_window, resample
from core.features import TrackHistory

MODEL = Path(__file__).resolve().parents[1] / "models" / "behavior_v1_open.onnx"
# стоящий человек ростом ~170 px: голова сверху, стопы внизу (COCO-17, относительно стоп)
STAND = np.array([[0, -170], [-5, -175], [5, -175], [-10, -172], [10, -172], [-25, -140], [25, -140], [-30, -100],
                  [30, -100], [-30, -65], [30, -65], [-15, -80], [15, -80], [-15, -40], [15, -40], [-15, 0], [15, 0]],
                 np.float32)


def standing_track(n=25, dx=0.0):
    h = TrackHistory(1, 15.0)
    for i in range(n):
        k = np.concatenate([STAND + [600 + dx * i, 500], np.full((17, 1), 0.9, np.float32)], 1)
        h.add(i * 0.1, np.array([570 + dx * i, 325, 630 + dx * i, 500], np.float32), k, 0.9)
    return h


def test_window_shape_and_scale_invariance():
    kp = np.repeat((STAND + [600, 500])[None], T, 0)
    sc = np.full((T, 17), 0.9, np.float32)
    a = make_window(kp, sc)
    b = make_window(kp * 2.0, sc)                    # тот же человек вдвое ближе к камере
    assert a.shape == (CHANNELS, T)
    assert np.allclose(a, b, atol=1e-5)


def test_window_skips_cut_person():
    kp = np.repeat((STAND + [600, 500])[None], T, 0)
    sc = np.full((T, 17), 0.9, np.float32)
    sc[:, 11:] = 0.0                                 # ног и бёдер не видно — обрезан краем кадра
    assert make_window(kp, sc) is None


def test_resample_repeats_first_frame_before_track_start():
    ts = np.arange(10) * 0.1
    kp = np.zeros((10, 17, 2), np.float32)
    sc = np.full((10, 17), 0.9, np.float32)
    _, sc2 = resample(ts, kp, sc, end=0.9)
    assert (sc2 > 0).all()                           # до начала трека — копия первого кадра, а не «дыра»


@pytest.mark.skipif(not MODEL.exists(), reason="нет core/models/behavior_v1_open.onnx")
def test_open_model_loads_and_standing_is_normal():
    m = BehaviorModel(MODEL)
    assert m.version == 2 and set(m.classes) == {"normal", "fall", "run"}
    p = m.probs(standing_track())
    assert p is not None and abs(sum(p.values()) - 1) < 1e-4
    assert max(p, key=p.get) == "normal"
