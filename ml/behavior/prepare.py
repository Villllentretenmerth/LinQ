"""Окна скелета для behavior_v1 из открытых наборов (формат PYSKL / MMAction2: *_2d.pkl, 17 точек COCO, 30 к/с).

Только numpy: можно запускать там, где лежат большие pkl, а наружу отдавать компактный npz.

    python ml/behavior/prepare.py --hmdb hmdb51_2d.pkl --ntu ntu60_2d.pkl --out windows.npz

Классы: normal / fall / run.
- fall: HMDB51 fall_floor, NTU60 A43 «падение». Окно берётся, только если в нём есть само падение: центр бёдер
  опустился на ≥ 0,25 роста за 0,5 с, и после падения в окне есть ещё ≥ 0,3 с.
- run: HMDB51 run (все окна ролика).
- normal: бытовые действия HMDB51 (ходьба, стоит, сел, поднял предмет, …) и все одиночные действия NTU60,
  кроме падения: «сел», «встал», «поднял с пола» — как раз то, с чем путается падение.
Спорт и трюки HMDB51 (сальто, колесо, прыжки, удары) не берём: они похожи на падение, но не нормальное поведение цеха.
"""
import argparse
import pickle
from pathlib import Path
import re
from collections import Counter

import numpy as np

T, STEP = 20, 3          # 20 кадров по 10 к/с из 30 к/с
L_HIP, R_HIP = 11, 12
HMDB51 = ["brush_hair", "cartwheel", "catch", "chew", "clap", "climb", "climb_stairs", "dive", "draw_sword", "dribble",
          "drink", "eat", "fall_floor", "fencing", "flic_flac", "golf", "handstand", "hit", "hug", "jump", "kick",
          "kick_ball", "kiss", "laugh", "pick", "pour", "pullup", "punch", "push", "pushup", "ride_bike", "ride_horse",
          "run", "shake_hands", "shoot_ball", "shoot_bow", "shoot_gun", "sit", "situp", "smile", "smoke", "somersault",
          "stand", "swing_baseball", "sword", "sword_exercise", "talk", "throw", "turn", "walk", "wave"]
HMDB_NORMAL = {"brush_hair", "catch", "chew", "clap", "climb_stairs", "drink", "eat", "hug", "kiss", "laugh", "pick",
               "pour", "push", "shake_hands", "sit", "smile", "smoke", "stand", "talk", "throw", "turn", "walk", "wave",
               "draw_sword", "shoot_gun"}
NTU_FALL = 42            # A43 falling down (метки с 0)
NTU_TWO_PERSON = set(range(49, 60))


def load(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    if isinstance(d, dict) and "annotations" in d:
        return d["annotations"], d.get("split") or {}
    return d, {}


def main_person(a):
    kp = np.asarray(a["keypoint"], np.float32)              # (M, F, 17, 2)
    sc = np.asarray(a["keypoint_score"], np.float32)        # (M, F, 17)
    if kp.ndim != 4 or kp.shape[0] == 0:
        return None, None
    m = int(np.argmax(sc.mean(axis=(1, 2))))
    return kp[m], sc[m]


def hip_y_norm(kp, sc):
    vis = sc >= 0.3
    hs = [np.ptp(kp[t, vis[t], 1]) for t in range(len(kp)) if vis[t].sum() >= 4]
    H = max(float(np.percentile(hs, 90)), 1.0) if hs else 1.0
    ok = vis[:, L_HIP] & vis[:, R_HIP]
    y = np.where(ok, kp[:, [L_HIP, R_HIP], 1].mean(1), np.nan)
    if np.isnan(y).all():
        return None
    idx = np.arange(len(y))
    y = np.interp(idx, idx[~np.isnan(y)], y[~np.isnan(y)])
    return y / H


def windows_of(kp, sc, step, label, rng, max_n=None, stride=5, fall_mask=None):
    """Все окна ролика (после прореживания до 10 к/с). Короткий ролик добивается в начале первым кадром."""
    kp, sc = kp[::step], sc[::step]
    if len(kp) < T:
        pad = T - len(kp)
        kp = np.concatenate([np.repeat(kp[:1], pad, 0), kp])
        sc = np.concatenate([np.repeat(sc[:1], pad, 0), sc])
    ends = list(range(T, len(kp) + 1, stride)) or [len(kp)]
    if label == "fall":
        y = hip_y_norm(kp, sc)
        if y is None or len(y) < 8:
            return []
        drop = y[5:] - y[:-5]                                 # опускание за 0,5 с
        f = int(np.argmax(drop))
        if fall_mask is not None and np.asarray(fall_mask)[::step].any():
            fall_at = int(np.argmax(np.asarray(fall_mask)[::step]))   # начало падения по разметке набора
            fall_at += max(0, T - len(fall_mask[::step]))
        elif drop[f] < 0.25:
            return []
        else:
            fall_at = f + 5
        ends = [e for e in range(T, len(kp) + 1) if fall_at + 3 <= e <= fall_at + 15]
        ends = ends[::2]
    if max_n and len(ends) > max_n:
        ends = sorted(rng.choice(ends, max_n, replace=False))
    return [(kp[e - T:e], sc[e - T:e]) for e in ends]


def collect(anns, split, name_of, test_keys, rng, src, steps=None, normal_max=3, stride=None):
    X, S, Y, G, TEST = [], [], [], [], []
    stats = Counter()
    for a in anns:
        lab = name_of(a)
        if lab is None:
            continue
        kp, sc = main_person(a)
        if kp is None:
            continue
        st = steps or ((2, 3, 4) if lab != "normal" else (3,))   # для редких классов — темп ×0,75…×1,5
        for step in st:
            max_n = normal_max if lab == "normal" else None
            st_w = stride or (10 if lab == "normal" else 4)
            for w_kp, w_sc in windows_of(kp, sc, step, lab, rng, max_n=max_n, stride=st_w,
                                         fall_mask=a.get("fall_mask")):
                X.append(w_kp.astype(np.float16))
                S.append(w_sc.astype(np.float16))
                Y.append(lab)
                G.append(f"{src}:{a['frame_dir']}")
                TEST.append(a["frame_dir"] in test_keys)
                stats[(lab, step)] += 1
    return X, S, Y, G, TEST, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hmdb", default=None)
    ap.add_argument("--ntu", default=None)
    ap.add_argument("--custom", action="append", default=[], help="pkl из extract_pose.py (уже 10 к/с), можно несколько")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ntu-normal-frac", type=float, default=0.35, help="какую долю роликов NTU «норма» взять")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    allX, allS, allY, allG, allT = [], [], [], [], []

    if a.hmdb:
        anns, split = load(a.hmdb)
        test = set(split.get("test1", []))

        def name_hmdb(x):
            c = HMDB51[int(x["label"])] if int(x["label"]) < len(HMDB51) else None
            if c == "fall_floor":
                return "fall"
            if c == "run":
                return "run"
            return "normal" if c in HMDB_NORMAL else None
        # проверка порядка классов по имени ролика
        bad = sum(1 for x in anns[:500] if not re.search(r"_" + re.escape(HMDB51[int(x["label"])]) + r"_", "_" + x["frame_dir"] + "_"))
        print(f"HMDB51: {len(anns)} роликов; имена классов не совпали в {bad} из 500")
        r = collect(anns, split, name_hmdb, test, rng, "hmdb")
        for L, v in zip((allX, allS, allY, allG, allT), r[:5]):
            L += v
        print("HMDB51 окна:", dict(r[5]))

    if a.ntu:
        anns, split = load(a.ntu)
        test = set(split.get("xsub_val", []))

        def name_ntu(x):
            l = int(x["label"])
            if l == NTU_FALL:
                return "fall"
            if l in NTU_TWO_PERSON:
                return None
            return "normal" if rng.random() < a.ntu_normal_frac else None
        print(f"NTU60: {len(anns)} роликов")
        r = collect(anns, split, name_ntu, test, rng, "ntu")
        for L, v in zip((allX, allS, allY, allG, allT), r[:5]):
            L += v
        print("NTU60 окна:", dict(r[5]))

    for path in a.custom:
        anns, split = load(path)
        test = set(split.get("test", []))
        print(f"{path}: {len(anns)} роликов")
        r = collect(anns, split, lambda x: x.get("label_name"), test, rng, Path(path).stem, steps=(1,),
                    normal_max=None, stride=5)   # свои ролики: все окна с шагом 0,5 с, «норма» без ограничения
        for L, v in zip((allX, allS, allY, allG, allT), r[:5]):
            L += v
        print("  окна:", dict(r[5]))

    classes = ["normal", "fall", "run"]
    y = np.array([classes.index(v) for v in allY], np.int64)
    np.savez_compressed(a.out, kp=np.stack(allX), score=np.stack(allS), y=y, group=np.array(allG),
                        test=np.array(allT), classes=np.array(classes))
    print("сохранено:", a.out, {c: int((y == i).sum()) for i, c in enumerate(classes)},
          "тест:", int(np.sum(allT)))


if __name__ == "__main__":
    main()
