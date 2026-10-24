"""Обучение behavior_v1 по окнам скелетов (core.tools.behavior_data) и выгрузка в ONNX для конвейера.

Модель — маленькая 1D-CNN по времени (свёртки с расширением 1, 2, 4 → среднее и максимум по окну → классы).
Рекуррентных слоёв нет: свёртки со статическим входом [1, T, F] квантуются и компилируются в программу TPU
тем же путём, что и YOLO.

Проверка честная: окна одной группы (человек или сцена) целиком либо в обучении, либо в проверке.

    # проверка на роликах команды, обучение на чужих датасетах
    python -m core.tools.train_behavior data/behavior/windows.npz --val "^ours-" --out core/models/behavior_v1.onnx
    # после проверки — финальная модель на всех данных с тем же числом эпох
    python -m core.tools.train_behavior data/behavior/windows.npz --val "^ours-" --final --out core/models/behavior_v1.onnx

Рядом с моделью пишется отчёт <out>.report.md: матрица ошибок и точность/полнота по классам на проверке.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(paths: list[str]):
    from ..behavior import FEATURE_VERSION

    X, y, g, classes = [], [], [], None
    for p in paths:
        d = np.load(p, allow_pickle=False)
        if int(d["feature_version"]) != FEATURE_VERSION:
            raise SystemExit(f"{p}: окна версии {int(d['feature_version'])}, нужна {FEATURE_VERSION} — пересоберите")
        c = [str(x) for x in d["classes"]]
        if classes is not None and c != classes:
            raise SystemExit(f"{p}: другие классы {c}")
        classes = c
        X.append(d["X"]), y.append(d["y"]), g.append(d["group"].astype(str))
    return np.concatenate(X), np.concatenate(y), np.concatenate(g), classes


def split(groups: np.ndarray, val_re: str | None, val_frac: float, seed: int) -> np.ndarray:
    """Маска проверки: группы по регулярному выражению или случайная доля групп."""
    uniq = np.unique(groups)
    if val_re:
        val = {u for u in uniq if re.search(val_re, u)}
    else:
        rng = np.random.default_rng(seed)
        val = set(rng.choice(uniq, max(1, int(round(len(uniq) * val_frac))), replace=False)) if len(uniq) > 1 else set()
    return np.isin(groups, list(val))


def build_model(features: int, n_cls: int, width: int = 64, dropout: float = 0.2):
    import torch
    import torch.nn as nn

    def block(cin, cout, dil):
        return nn.Sequential(nn.Conv1d(cin, cout, 3, padding=dil, dilation=dil), nn.BatchNorm1d(cout), nn.ReLU())

    class BehaviorNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(block(features, width, 1), block(width, width, 2), block(width, width, 4))
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * width, n_cls))

        def forward(self, x):                      # x: [B, T, F]
            h = self.body(x.transpose(1, 2))       # [B, C, T]
            return self.head(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1))

    return BehaviorNet()


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Отражение лево/право, потеря точек (перекрытия), масштаб и шум координат."""
    from ..behavior import FLIP_PAIRS, X_COLUMNS

    x = x.copy()
    b = len(x)
    flip = rng.random(b) < 0.5
    if flip.any():
        f = x[flip]
        for i, j in FLIP_PAIRS:
            a, c = f[:, :, 3 * i:3 * i + 3].copy(), f[:, :, 3 * j:3 * j + 3].copy()
            f[:, :, 3 * i:3 * i + 3], f[:, :, 3 * j:3 * j + 3] = c, a
        f[:, :, X_COLUMNS] *= -1
        x[flip] = f
    drop = rng.random((b, 1, 17)) < 0.08 * rng.random((b, 1, 1))    # у части окон теряется до 8 % точек
    drop = np.broadcast_to(drop, (b, x.shape[1], 17))
    for k in range(3):
        x[:, :, k:51:3][drop] = 0.0
    coords = list(range(0, 51, 3)) + list(range(1, 51, 3)) + list(range(51, 55))
    x[:, :, coords] *= rng.uniform(0.9, 1.1, (b, 1, 1)).astype(np.float32)
    x[:, :, coords] += rng.normal(0, 0.01, (b, x.shape[1], len(coords))).astype(np.float32)
    return x


def metrics(y: np.ndarray, p: np.ndarray, n: int) -> dict:
    cm = np.zeros((n, n), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    prec = [cm[i, i] / cm[:, i].sum() if cm[:, i].sum() else None for i in range(n)]
    rec = [cm[i, i] / cm[i].sum() if cm[i].sum() else None for i in range(n)]
    f1 = [2 * a * b / (a + b) if a and b else 0.0 for a, b in zip(prec, rec)]
    present = [i for i in range(n) if cm[i].sum()]
    return {"cm": cm, "precision": prec, "recall": rec, "macro_f1": float(np.mean([f1[i] for i in present])) if present else 0.0}


def train(X, y, n_cls, epochs, seed, Xv=None, yv=None, log=print):
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(X.shape[2], n_cls)
    counts = np.bincount(y, minlength=n_cls).astype(np.float32)
    w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
    w = w / w[counts > 0].mean()
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * max(1, int(np.ceil(len(X) / 256))))
    best = (-1.0, None, 0)
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(X))
        total = 0.0
        for k in range(0, len(X), 256):
            idx = order[k:k + 256]
            xb = torch.from_numpy(augment(X[idx], rng))
            yb = torch.from_numpy(y[idx])
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        msg = f"эпоха {ep:3d}: потеря {total / len(X):.4f}"
        if Xv is not None and len(Xv):
            m = metrics(yv, predict(model, Xv).argmax(1), n_cls)
            msg += f", проверка macro-F1 {m['macro_f1']:.3f}"
            if m["macro_f1"] > best[0]:
                best = (m["macro_f1"], {k: v.clone() for k, v in model.state_dict().items()}, ep)
        log(msg)
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, best[2] or epochs


def predict(model, X: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        out = [torch.softmax(model(torch.from_numpy(X[k:k + 1024])), 1).numpy() for k in range(0, len(X), 1024)]
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def export(model, classes: list[str], out: Path) -> None:
    import onnx
    import torch

    from ..behavior import FEATURE_VERSION, FEATURES, T

    class WithSoftmax(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return torch.softmax(self.m(x), dim=1)

    model.eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(WithSoftmax(model), torch.zeros(1, T, FEATURES), str(out), input_names=["x"],
                      output_names=["probs"], opset_version=13, dynamo=False)
    m = onnx.load(str(out))
    for k, v in {"name": "behavior_v1", "classes": json.dumps(classes), "input_shape": json.dumps([T, FEATURES]),
                 "feature_version": str(FEATURE_VERSION)}.items():
        e = m.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(m, str(out))


def report(path: Path, classes: list[str], m: dict, n_train: int, n_val: int, val_groups: list[str], epochs: int) -> str:
    lines = [f"# behavior_v1: проверка", "",
             f"Окон в обучении: {n_train}, в проверке: {n_val} (группы: {', '.join(val_groups) or '—'}). Эпох: {epochs}.",
             f"Macro-F1 на проверке: **{m['macro_f1']:.3f}**.", "",
             "| Класс | Окон | Precision | Recall |", "|---|---|---|---|"]
    for i, c in enumerate(classes):
        p, r = m["precision"][i], m["recall"][i]
        lines.append(f"| {c} | {int(m['cm'][i].sum())} | {'—' if p is None else f'{p:.0%}'} | {'—' if r is None else f'{r:.0%}'} |")
    lines += ["", "Матрица ошибок (строки — правда, столбцы — ответ модели):", "",
              "| | " + " | ".join(classes) + " |", "|---" * (len(classes) + 1) + "|"]
    for i, c in enumerate(classes):
        lines.append(f"| **{c}** | " + " | ".join(str(int(v)) for v in m["cm"][i]) + " |")
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+", help="npz из core.tools.behavior_data build")
    ap.add_argument("--val", default=None, help="регулярка групп для проверки (например ^ours-)")
    ap.add_argument("--val-frac", type=float, default=0.2, help="доля случайных групп, если --val не задан")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--final", action="store_true", help="после проверки дообучить на всех окнах и выгрузить её")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "core" / "models" / "behavior_v1.onnx"))
    a = ap.parse_args(argv)

    X, y, g, classes = load(a.data)
    n = len(classes)
    vmask = split(g, a.val, a.val_frac, a.seed)
    print(f"окон {len(y)}: " + ", ".join(f"{c} {int((y == i).sum())}" for i, c in enumerate(classes)))
    print(f"обучение {int((~vmask).sum())}, проверка {int(vmask.sum())} ({len(np.unique(g[vmask]))} групп)")
    if not (~vmask).any():
        raise SystemExit("все окна попали в проверку — уточните --val")
    model, best_ep = train(X[~vmask], y[~vmask], n, a.epochs, a.seed, X[vmask], y[vmask])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if vmask.any():
        m = metrics(y[vmask], predict(model, X[vmask]).argmax(1), n)
        print("\n" + report(out.with_suffix(".report.md"), classes, m, int((~vmask).sum()), int(vmask.sum()),
                            sorted(np.unique(g[vmask])), best_ep))
    if a.final:
        print(f"финальная модель: все {len(y)} окон, {best_ep} эпох")
        model, _ = train(X, y, n, best_ep, a.seed)
    export(model, classes, out)
    print(f"модель: {out}")


if __name__ == "__main__":
    main()
