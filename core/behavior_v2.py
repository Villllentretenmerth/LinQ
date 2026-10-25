"""Окно скелета для behavior_v1 второй версии признаков (модель на открытых данных: NTU60, HMDB51, KTH).

Один и тот же код строит окно и при обучении (ml/behavior/train.py), и в конвейере (core/behavior.py,
FEATURE_VERSION 2): модель видит на камере ровно то, на чём училась.

Окно — 2 с при 10 к/с, 17 точек COCO. Координаты точек — относительно медианного центра бёдер за окно и в ростах
человека (рост — 90-й перцентиль высоты скелета за окно, то есть «пока стоял»), поэтому падение видно как
смещение и сжатие скелета, а бег — как ход ног и смещение. Скорость точек — разность соседних кадров.
Калибровка пола в признаки не входит: в открытых наборах её нет.

Вход модели: (1, 85, 20) — по 17 точкам x, y, уверенность, скорость x, скорость y; 20 моментов времени.
"""
import numpy as np

T = 20                # кадров в окне
FPS = 10.0            # частота окна
KP_CONF = 0.3
N_KP = 17
CHANNELS = 5 * N_KP   # x, y, score, dx, dy
L_HIP, R_HIP = 11, 12
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]   # лево <-> право для отражения


def make_window(kp: np.ndarray, score: np.ndarray) -> np.ndarray | None:
    """kp (T, 17, 2) в пикселях, score (T, 17) -> (85, T) float32 или None, если человека почти не видно."""
    kp = np.asarray(kp, np.float32)
    sc = np.asarray(score, np.float32)
    vis = sc >= KP_CONF
    if vis.sum(1).mean() < 6:                       # в среднем меньше 6 точек — не судим
        return None
    hips = (vis[:, L_HIP] | vis[:, R_HIP]).mean()
    legs = (vis[:, 13] | vis[:, 14] | vis[:, 15] | vis[:, 16]).mean()
    if hips < 0.6 or legs < 0.5:                    # человек обрезан краем кадра или закрыт — по позе не судим
        return None
    hs = []
    for t in range(kp.shape[0]):
        if vis[t].sum() >= 4:
            ys = kp[t, vis[t], 1]
            hs.append(float(ys.max() - ys.min()))
    if not hs:
        return None
    H = max(float(np.percentile(hs, 90)), 1.0)       # рост: верхняя часть распределения (до падения стоял)
    hip_ok = vis[:, L_HIP] & vis[:, R_HIP]
    if hip_ok.any():
        origin = np.median(kp[hip_ok][:, [L_HIP, R_HIP]].mean(1), axis=0)
    else:
        origin = np.median(kp[vis], axis=0)
    xy = (kp - origin) / H
    xy[~vis] = 0.0
    xy = np.clip(xy, -3, 3)
    vel = np.zeros_like(xy)
    both = vis[1:] & vis[:-1]
    vel[1:][both] = (xy[1:] - xy[:-1])[both]
    feat = np.concatenate([xy[..., 0], xy[..., 1], np.where(vis, sc, 0.0), vel[..., 0], vel[..., 1]], axis=1)
    return feat.T.astype(np.float32)              # (85, T)


def resample(ts: np.ndarray, kp: np.ndarray, sc: np.ndarray, end: float, n: int = T, fps: float = FPS):
    """Взять n кадров с шагом 1/fps, заканчивая в end: для каждого — ближайший по времени сэмпл трека."""
    grid = end - (n - 1 - np.arange(n)) / fps
    idx = np.clip(np.searchsorted(ts, grid), 0, len(ts) - 1)
    prev = np.clip(idx - 1, 0, len(ts) - 1)
    idx = np.where(np.abs(ts[prev] - grid) < np.abs(ts[idx] - grid), prev, idx)
    far = np.abs(ts[idx] - grid) > 1.5 / fps      # дырка в треке — точки этого кадра считаем невидимыми
    far &= grid >= ts[0]                          # до начала трека — повторяем первый кадр, как при обучении
    kp2, sc2 = kp[idx].copy(), sc[idx].copy()
    sc2[far] = 0.0
    return kp2, sc2


def window_from_history(h) -> np.ndarray | None:
    """Окно (85, T) по истории трека (core.features.TrackHistory), заканчивающееся на последнем кадре."""
    if not h.samples:
        return None
    now = h.last_ts
    smp = [s for s in h.since(now - (T - 1) / FPS - 0.05) if s.kpts is not None]
    if len(smp) < 0.7 * T:
        return None
    ts = np.array([s.ts for s in smp])
    kp = np.stack([s.kpts[:, :2] for s in smp])
    sc = np.stack([s.kpts[:, 2] for s in smp])
    kp, sc = resample(ts, kp, sc, now)
    return make_window(kp, sc)
