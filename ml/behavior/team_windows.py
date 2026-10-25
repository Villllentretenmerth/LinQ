"""Окна из роликов команды (наш ракурс) для дообучения behavior_v1 — только с перекрёстной проверкой:
модель, которая оценивается на ролике, этот ролик в обучении не видела (--fold).

    python ml/behavior/team_windows.py --cache out/cache --labels clips/labels.csv --out team.npz

Скелет главного человека — из кэша выходов YOLO-pose (eval_clips --cache). Разметка — labels.csv:
- падение: окна, которые кончаются через 0,3–2 с после начала падения -> fall; окна до падения -> normal;
  «лежит» после падения не берём (это уже не само падение, его ловит этап «неподвижен»);
- бег: окна, на 70% лежащие внутри интервала бега -> run; дальше секунды от интервала -> normal;
- остальные ролики (зона, ловушки, подъём, телефон) -> normal.
Фолды: 3 группы, в каждой по ролику с падением и с бегом.
"""
import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
T, FPS = 20, 10.0
FOLDS = {0: ["v05", "v08", "v02", "v11", "v14", "v17"], 1: ["v06", "v09", "v03", "v12", "v15", "v18"],
         2: ["v07", "v10", "v04", "v13", "v16", "v01"]}


def main_person_seq(poses):
    kp, sc, prev = [], [], None
    for r in poses:
        if len(r.conf):
            area = (r.xyxy[:, 2] - r.xyxy[:, 0]) * (r.xyxy[:, 3] - r.xyxy[:, 1])
            s = r.conf * area
            if prev is not None:
                inter = [max(0, min(b[2], prev[2]) - max(b[0], prev[0])) * max(0, min(b[3], prev[3]) - max(b[1], prev[1]))
                         for b in r.xyxy]
                s = s * (0.2 + np.array(inter) / np.maximum(area, 1))
            j = int(np.argmax(s))
            kp.append(r.kpts[j, :, :2])
            sc.append(r.kpts[j, :, 2])
            prev = r.xyxy[j]
        else:
            kp.append(np.zeros((17, 2), np.float32))
            sc.append(np.zeros(17, np.float32))
    return np.array(kp, np.float32), np.array(sc, np.float32)


def num(x):
    x = (x or "").strip().replace(",", ".")
    return float(x) if x else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lab = {r["file"].replace(".mp4", ""): r for r in csv.DictReader(open(a.labels, encoding="utf-8-sig"), delimiter=";")}
    fold_of = {v: f for f, vs in FOLDS.items() for v in vs}
    X, S, Y, G = [], [], [], []
    for v, r in sorted(lab.items()):
        poses = pickle.load(open(Path(a.cache) / f"{v}_pose.pkl", "rb"))
        kp, sc = main_person_seq(poses)
        typ, st, en = r["type"].strip().lower(), num(r["start"]), num(r["end"])
        for e in range(T, len(kp) + 1, 2):
            t_end, t_start = e / FPS, (e - T) / FPS
            y = 0
            if typ == "падение":
                if st + 0.3 <= t_end <= st + 2.0:
                    y = 1
                elif t_end > st + 2.0 or t_end > st - 0.5:
                    continue
            elif typ == "бег":
                inside = max(0.0, min(t_end, en) - max(t_start, st)) / (T / FPS)
                if inside >= 0.7:
                    y = 2
                elif t_end > st - 1.0 and t_start < en + 1.0:
                    continue
            X.append(kp[e - T:e].astype(np.float16))
            S.append(sc[e - T:e].astype(np.float16))
            Y.append(y)
            G.append(f"team:{v}:fold{fold_of.get(v, 0)}")
    y = np.array(Y)
    np.savez_compressed(a.out, kp=np.stack(X), score=np.stack(S), y=y, group=np.array(G),
                        test=np.zeros(len(y), bool), classes=np.array(["normal", "fall", "run"]))
    print("окна команды:", {c: int((y == k).sum()) for k, c in enumerate(["normal", "fall", "run"])})


if __name__ == "__main__":
    main()
