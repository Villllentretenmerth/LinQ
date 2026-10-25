"""Обучение behavior_v1 (признаки версии 2, core/behavior_v2.py) на окнах из prepare.py и выгрузка в ONNX.

    python ml/behavior/train.py --data win_hmdb.npz win_kth.npz win_ntu.npz --out core/models/behavior_v1_open.onnx --epochs 15

Модель — свёртки по времени над 85 каналами (17 точек × x, y, уверенность, скорость x, y), около 60 тыс. весов.
Аугментации под нашу камеру: сверху под углом (сжатие по вертикали), отражение, поворот, пропуски точек, шум.
Проверка — отложенные роли/субъекты наборов (split test1 / xsub_val); честная проверка — наши ролики через
core.tools.eval_clips (они в обучение не попадают).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.behavior_v2 import CHANNELS, FLIP, FPS, T, make_window  # noqa: E402


class Net(nn.Module):
    def __init__(self, n_cls: int, c_in: int = CHANNELS, w: int = 64, p_drop: float = 0.3):
        super().__init__()

        def block(ci, co, d):
            return nn.Sequential(nn.Conv1d(ci, co, 3, padding=d, dilation=d), nn.BatchNorm1d(co), nn.ReLU())
        self.body = nn.Sequential(block(c_in, w, 1), block(w, w, 2), block(w, w, 4), block(w, w, 1))
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(2 * w, n_cls))

    def forward(self, x):
        h = self.body(x)
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))


def augment(kp, sc, rng):
    kp = kp.astype(np.float32).copy()
    sc = sc.astype(np.float32).copy()
    c = np.nanmean(kp[sc >= 0.3], axis=0) if (sc >= 0.3).any() else np.zeros(2, np.float32)
    kp -= c
    if rng.random() < 0.5:                                   # отражение лево-право
        kp[..., 0] *= -1
        kp, sc = kp[:, FLIP], sc[:, FLIP]
    a = np.deg2rad(rng.uniform(-12, 12))                     # наклон камеры
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    kp = kp @ R.T
    kp[..., 1] *= rng.uniform(0.6, 1.05)                     # камера сверху: тело сжато по вертикали
    kp *= rng.uniform(0.7, 1.4)
    kp += rng.normal(0, 0.01 * max(1.0, float(np.ptp(kp[..., 1]))), kp.shape)   # дрожание точек
    drop = rng.random(sc.shape) < rng.uniform(0, 0.15)       # пропавшие точки
    sc = np.where(drop, 0.0, np.clip(sc + rng.normal(0, 0.05, sc.shape), 0, 1))
    return kp + c, sc


def featurize(kp, sc, rng=None):
    if rng is not None:
        kp, sc = augment(kp, sc, rng)
    x = make_window(kp, sc)
    return x


def build(data, idx, rng=None):
    X, keep = [], []
    for i in idx:
        x = featurize(data["kp"][i], data["score"][i], rng)
        if x is not None:
            X.append(x)
            keep.append(i)
    return np.stack(X), np.array(keep)


def report(y, p, classes):
    out = {}
    for k, c in enumerate(classes):
        tp = int(((p == k) & (y == k)).sum())
        out[c] = {"n": int((y == k).sum()), "recall": tp / max(1, (y == k).sum()), "precision": tp / max(1, (p == k).sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+", help="один или несколько npz из prepare.py")
    ap.add_argument("--out", default=str(ROOT / "core" / "models" / "behavior_v1_open.onnx"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-normal", type=int, default=30000, help="ограничить «норму», чтобы эпоха шла быстрее")
    ap.add_argument("--exclude", default=None, help="не учить на окнах, где в group есть эта подстрока (фолд проверки)")
    ap.add_argument("--team-weight", type=float, default=1.0, help="во сколько раз чаще брать окна нашего ракурса (team:)")
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    parts = [dict(np.load(p, allow_pickle=False)) for p in a.data]
    d = {k: np.concatenate([p[k] for p in parts]) for k in ("kp", "score", "y", "group", "test")}
    d["classes"] = parts[0]["classes"]
    classes = [str(c) for c in d["classes"]]
    y_all, test = d["y"], d["test"].astype(bool)
    grp = d["group"].astype(str)
    keep = np.ones(len(y_all), bool) if not a.exclude else ~np.char.find(grp, a.exclude).__ge__(0)
    tr = np.where(~test & keep)[0]
    te = np.where(test)[0]
    norm = tr[y_all[tr] == 0]
    if len(norm) > a.max_normal:
        tr = np.concatenate([rng.choice(norm, a.max_normal, replace=False), tr[y_all[tr] != 0]])
    print("train:", {c: int((y_all[tr] == k).sum()) for k, c in enumerate(classes)},
          "test:", {c: int((y_all[te] == k).sum()) for k, c in enumerate(classes)})
    Xte, kte = build(d, te)
    yte = y_all[kte]

    net = Net(len(classes))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * (len(tr) // a.batch + 1))
    counts = np.bincount(y_all[tr], minlength=len(classes)).astype(np.float32)
    w = torch.tensor((counts.sum() / (len(classes) * np.maximum(counts, 1))) ** 0.5)   # мягкая балансировка
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    p_samp = (1.0 / counts[y_all[tr]])
    team = np.char.startswith(grp[tr], "team:")
    p_samp = p_samp * np.where(team, a.team_weight, 1.0)
    p_samp /= p_samp.sum()
    best, best_state = -1, None
    for ep in range(a.epochs):
        t0 = time.time()
        net.train()
        base = np.where(team, a.team_weight, 1.0)
        order = rng.choice(tr, len(tr), p=0.5 * p_samp + 0.5 * base / base.sum())   # пересэмплинг редких классов и нашего ракурса
        tot = 0.0
        for b in range(0, len(order), a.batch):
            Xb, kb = build(d, order[b:b + a.batch], rng)
            xb, yb = torch.from_numpy(Xb), torch.from_numpy(y_all[kb])
            loss = lossf(net(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(kb)
        net.eval()
        with torch.no_grad():
            p = net(torch.from_numpy(Xte)).argmax(1).numpy()
        r = report(yte, p, classes)
        score = np.mean([r[c]["recall"] for c in classes])          # сбалансированная точность
        print(f"эпоха {ep + 1}: loss {tot / len(order):.3f}, test " +
              ", ".join(f"{c} R={r[c]['recall']:.2f} P={r[c]['precision']:.2f}" for c in classes) +
              f" ({time.time() - t0:.0f} с)")
        if score > best:
            best, best_state = score, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        p = net(torch.from_numpy(Xte)).argmax(1).numpy()
    r = report(yte, p, classes)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, torch.zeros(1, CHANNELS, T), str(out), input_names=["x"], output_names=["logits"],
                      opset_version=13, dynamo=False)
    # выход — вероятности, в метаданных — классы и версия признаков: так модель читает core/behavior.py
    import onnx
    from onnx import helper
    m = onnx.load(str(out))
    m.graph.node.append(helper.make_node("Softmax", ["logits"], ["probs"], axis=1, name="softmax_out"))
    del m.graph.output[:]
    m.graph.output.append(helper.make_tensor_value_info("probs", onnx.TensorProto.FLOAT, [1, len(classes)]))
    for k, v in {"name": "behavior_v1", "classes": json.dumps(classes), "input_shape": json.dumps([CHANNELS, T]),
                 "feature_version": "2", "trained_on": ", ".join(Path(p).name for p in a.data)}.items():
        prop = m.metadata_props.add()
        prop.key, prop.value = k, v
    onnx.checker.check_model(m)
    onnx.save(m, str(out))
    meta = {"name": "behavior_v1", "classes": classes, "T": T, "fps": FPS, "channels": CHANNELS,
            "trained_on": [Path(p).name for p in a.data], "test_split": r,
            "params": int(sum(p.numel() for p in net.parameters()))}
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("модель:", out, "весов:", meta["params"], "\nтест наборов:", json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
