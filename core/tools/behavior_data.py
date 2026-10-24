"""Окна скелетов для обучения behavior_v1: видео + разметка → out/behavior/<имя>.npz.

Шаг 1 — манифест: какие ролики и где в них какое действие (один формат для всех датасетов):

    file;start;end;class;group
    D:/data/kth/person01_running_d1_uncomp.avi;;;run;kth-person01     # пустое время — весь ролик
    D:/data/urfall/fall-01-cam0.mp4;2,3;3,4;fall;urfall-01             # интервал действия, с
    D:/data/urfall/adl-01-cam0.mp4;;;normal;urfall-adl-01

Необязательный столбец crop — x1,y1,x2,y2 части кадра (в UR Fall кадр — глубина и цвет рядом, берём цвет).
group — кто в кадре (человек или сцена): при проверке группа целиком уходит или в обучение, или в проверку,
иначе модель «узнаёт» человека, а не действие. Манифест собирается из готовых датасетов:

    python -m core.tools.behavior_data manifest --kth D:/data/kth --out data/behavior/kth.csv
    python -m core.tools.behavior_data manifest --urfall D:/data/urfall --out data/behavior/urfall.csv
    python -m core.tools.behavior_data manifest --ours eval/labels.csv --videos clips --out data/behavior/ours.csv

Шаг 2 — окна: ролик идёт через ту же yolo-pose и тот же трекер, что в конвейере, с шагом 1/fps; раз в
STRIDE_SEC для каждого трека строится окно (`core.behavior.window_tensor`) и получает метку:

    окно перекрывает интервал действия не меньше чем на min(--min-overlap, половина интервала) → класс действия;
    перекрывает меньше → окно пропускается (непонятно, что в нём);
    не перекрывает → normal.

В роликах с интервалами действие относится к главному треку (самому длинному); остальные треки там
пропускаются. В роликах без интервалов (весь ролик — один класс) берутся все треки.

    python -m core.tools.behavior_data build data/behavior/kth.csv data/behavior/urfall.csv \\
        --out data/behavior/windows.npz --cache out/cache/behavior --workers 4

Выходы позы кэшируются по ролику: повторная сборка (другие метки, другой min-overlap) идёт без нейросети.
"""
import argparse
import csv
import hashlib
import logging
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("core.behavior_data")

CLASSES = ["normal", "run", "fall", "lift_back"]
OURS = {"бег": "run", "падение": "fall", "подъём спиной": "lift_back", "подъем спиной": "lift_back",
        "зона": "normal", "ловушка": "normal", "нет": "normal", "": "normal"}


def _num(x):
    x = (x or "").strip().replace(",", ".")
    return float(x) if x else None


# ---------------------------------------------------------------- манифесты из датасетов

def manifest_kth(root: Path, jogging: str) -> list[dict]:
    """KTH Actions: person01_running_d1_uncomp.avi. running → run, jogging → --jogging, walking → normal."""
    cls = {"running": "run", "jogging": jogging, "walking": "normal"}
    rows = []
    for f in sorted(root.rglob("*.avi")):
        m = re.match(r"(person\d+)_([a-z]+)_d\d", f.name)
        if m and m.group(2) in cls:
            rows.append({"file": str(f), "start": None, "end": None, "class": cls[m.group(2)], "group": f"kth-{m.group(1)}"})
    return rows


def manifest_urfall(root: Path, fps: float = 30.0, cam1: bool = False) -> list[dict]:
    """UR Fall: fall-NN-camX.mp4, adl-NN-cam0.mp4 и urfall-cam0-falls.csv (кадры: -1 не лежит, 0 падает, 1 лежит).
    Интервал падения — кадры с меткой 0 (разметка по камере 0). Камера 1 висит под потолком и снимает сверху:
    поза так почти не находится и ракурс не похож на камеру наблюдения — по умолчанию не берём.
    Кадр mp4 — две картинки 320×240 рядом: слева глубина, справа цвет; берём правую."""
    falling: dict[str, list[int]] = {}
    lab = next(iter(root.rglob("urfall-cam0-falls.csv")), None)
    if lab is None:
        log.warning("нет urfall-cam0-falls.csv в %s — ролики падений пропущены", root)
    else:
        with open(lab, encoding="utf-8") as fh:
            for r in csv.reader(fh):
                if len(r) >= 3 and r[0].startswith("fall-") and r[2].strip() == "0":
                    falling.setdefault(r[0].strip(), []).append(int(float(r[1])))
    rows = []
    for f in sorted(root.rglob("*.mp4")):
        m = re.match(r"(fall|adl)-(\d+)-cam(\d)\.mp4", f.name)
        if not m or (m.group(3) == "1" and not cam1):
            continue
        seq = f"{m.group(1)}-{m.group(2)}"
        rgb = "320,0,640,240"
        if m.group(1) == "adl":
            rows.append({"file": str(f), "start": None, "end": None, "class": "normal", "group": f"urfall-{seq}",
                         "crop": rgb})
        elif seq in falling:
            fr = falling[seq]
            rows.append({"file": str(f), "start": (min(fr) - 1) / fps, "end": max(fr) / fps, "class": "fall",
                         "group": f"urfall-{seq}", "crop": rgb})
    return rows


def manifest_ours(labels: Path, videos: Path, fall_sec: float) -> list[dict]:
    """Ролики команды (eval/labels.csv). У падения в разметке интервал до конца лежания — для поведения берём
    первые fall_sec секунд (само падение); «зона» и ловушки — normal."""
    rows = []
    with open(labels, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            t = (r.get("type") or "").strip().lower()
            cls = OURS.get(t)
            if cls is None:
                log.warning("%s: тип «%s» не относится к поведению — пропущен", r["file"], t)
                continue
            s, e = _num(r.get("start")), _num(r.get("end"))
            if cls == "normal":
                s = e = None
            elif cls == "fall" and s is not None:
                e = min(e, s + fall_sec) if e is not None else s + fall_sec
            rows.append({"file": str(videos / r["file"].strip()), "start": s, "end": e, "class": cls,
                         "group": f"ours-{Path(r['file']).stem}"})
    return rows


def write_manifest(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["file", "start", "end", "class", "group", "crop"])
        for r in rows:
            w.writerow([r["file"], "" if r["start"] is None else f"{r['start']:.2f}",
                        "" if r["end"] is None else f"{r['end']:.2f}", r["class"], r["group"], r.get("crop") or ""])
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    print(f"{out}: роликов {len(rows)}, по классам {counts}")


def read_manifests(paths: list[str]) -> dict[str, dict]:
    """Ролики: file -> {group, whole (класс всего ролика или None), intervals [(start, end, class)]}."""
    videos: dict[str, dict] = {}
    for p in paths:
        with open(p, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh, delimiter=";"):
                cls = r["class"].strip()
                if cls not in CLASSES:
                    raise SystemExit(f"{p}: класс «{cls}» не из {CLASSES}")
                crop = (r.get("crop") or "").strip()
                v = videos.setdefault(r["file"].strip(), {"group": r["group"].strip(), "whole": None, "intervals": [],
                                                          "crop": [int(x) for x in crop.split(",")] if crop else None})
                s, e = _num(r.get("start")), _num(r.get("end"))
                if s is None:
                    v["whole"] = cls
                else:
                    v["intervals"].append((s, e if e is not None else s + 2.0, cls))
    return videos


# ---------------------------------------------------------------- окна

def label_window(t_end: float, win: float, video: dict, min_overlap: float) -> str | None:
    if video["whole"] and not video["intervals"]:
        return video["whole"]
    t0 = t_end - win
    for s, e, cls in video["intervals"]:
        ov = min(t_end, e) - max(t0, s)
        if ov <= 0:
            continue
        need = min(min_overlap, 0.5 * (e - s))
        return cls if cls == "normal" or ov >= need - 1e-6 else None
    return "normal"


def _poses(models, path: str, fps: float, cache_dir: Path | None, crop: list[int] | None = None) -> dict:
    """{"frames": [(время в ролике, PoseResult)], "wh": (ширина, высота)}. crop — x1,y1,x2,y2 части кадра."""
    from ..ingest import offline_frames

    cache = None
    if cache_dir is not None:
        key = hashlib.md5(f"{Path(path).resolve()}|{fps}{f'|{crop}' if crop else ''}".encode()).hexdigest()[:12]
        cache = cache_dir / f"{Path(path).stem}_{key}.pkl"
        if cache.exists():
            return pickle.load(open(cache, "rb"))
    out, t0, shape = [], None, None
    for fr in offline_frames(path, fps):
        t0 = fr.ts if t0 is None else t0
        img = fr.image if not crop else np.ascontiguousarray(fr.image[crop[1]:crop[3], crop[0]:crop[2]])
        shape = img.shape
        out.append((fr.ts - t0, models.pose(img)))
    res = {"frames": out, "wh": None if shape is None else (shape[1], shape[0])}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(res, open(cache, "wb"))
    return res


def windows_for_video(models, path: str, video: dict, fps: float, cache_dir: Path | None,
                      min_overlap: float) -> tuple[list[np.ndarray], list[str], list[float]]:
    from ..behavior import STRIDE_SEC, WINDOW_SEC, window_tensor
    from ..tracking import PersonTracker

    res = _poses(models, path, fps, cache_dir, video.get("crop"))
    tracker = PersonTracker(fps)
    root: dict[int, int] = {}            # трек -> первый трек цепочки (после смены ID при падении)
    frames: dict[int, int] = {}          # длина цепочки в кадрах
    next_t: dict[int, float] = {}
    recs = []                            # (цепочка, окно, метка, время)
    for t, pose in res["frames"]:
        persons, _ = tracker.update(pose, t)
        for p in persons:
            tid = p.track_id
            if tid not in root:
                donor = tracker.stitched.get(tid)
                root[tid] = root.get(donor, donor) if donor is not None else tid
            r = root[tid]
            frames[r] = frames.get(r, 0) + 1
            if t + 1e-6 < next_t.get(tid, -1.0):
                continue
            next_t[tid] = t + STRIDE_SEC
            x = window_tensor(p.hist)
            if x is None:
                continue
            lab = label_window(t, WINDOW_SEC, video, min_overlap)
            if lab is not None:
                recs.append((r, x, lab, t))
    if video["intervals"] and frames:
        main = max(frames, key=frames.get)
        recs = [x for x in recs if x[0] == main]
    return [x[1] for x in recs], [x[2] for x in recs], [x[3] for x in recs]


_worker_models = None


def _worker(args):
    global _worker_models
    config, threads, path, video, fps, cache_dir, min_overlap = args
    if _worker_models is None:
        from ..pipeline import Models

        cfg = yaml.safe_load(open(config, encoding="utf-8"))
        cfg.setdefault("models", {})["threads"] = threads
        cfg["models"].pop("det", None)              # для окон нужна только поза
        cfg["models"].pop("behavior", None)
        _worker_models = Models(cfg, ROOT)
    try:
        return path, windows_for_video(_worker_models, path, video, fps, cache_dir, min_overlap), None
    except Exception as e:                     # битый ролик не роняет всю сборку
        return path, None, f"{type(e).__name__}: {e}"


def build(a) -> None:
    from ..behavior import FEATURE_VERSION, FEATURES, T

    videos = read_manifests(a.manifests)
    missing = [f for f in videos if not Path(f).exists()]
    for f in missing:
        log.warning("нет файла %s", f)
        del videos[f]
    cache_dir = Path(a.cache) if a.cache else None
    jobs = [(a.config, a.threads, f, v, a.fps, cache_dir, a.min_overlap) for f, v in videos.items()]
    X, y, g, files, ts = [], [], [], [], []
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(_worker, j) for j in jobs]):
            path, out, err = fut.result()
            done += 1
            if err:
                print(f"[{done}/{len(jobs)}] {Path(path).name}: ОШИБКА {err}", file=sys.stderr)
                continue
            xs, ls, tt = out
            X += xs
            y += [CLASSES.index(l) for l in ls]
            g += [videos[path]["group"]] * len(xs)
            files += [path] * len(xs)
            ts += tt
            print(f"[{done}/{len(jobs)}] {Path(path).name}: окон {len(xs)}"
                  + "".join(f", {c} {ls.count(c)}" for c in CLASSES if ls.count(c)), flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=np.array(X, np.float32).reshape(-1, T, FEATURES), y=np.array(y, np.int64),
                        group=np.array(g), files=np.array(files), t=np.array(ts, np.float32),
                        classes=np.array(CLASSES), feature_version=FEATURE_VERSION)
    yy = np.array(y)
    print(f"\n{out}: окон {len(y)}, по классам "
          + ", ".join(f"{c} {int((yy == i).sum())}" for i, c in enumerate(CLASSES)) + f", групп {len(set(g))}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest", help="манифест из датасета")
    m.add_argument("--kth", help="папка KTH Actions (running, jogging, walking)")
    m.add_argument("--jogging", default="run", choices=["run", "normal"], help="куда относить бег трусцой")
    m.add_argument("--urfall", help="папка UR Fall (mp4 + urfall-cam0-falls.csv)")
    m.add_argument("--ours", help="eval/labels.csv команды")
    m.add_argument("--videos", default="clips", help="папка роликов для --ours")
    m.add_argument("--fall-sec", type=float, default=2.0, help="сколько секунд от начала падения считать падением")
    m.add_argument("--out", required=True)
    b = sub.add_parser("build", help="окна скелетов по манифестам")
    b.add_argument("manifests", nargs="+")
    b.add_argument("--out", default="data/behavior/windows.npz")
    b.add_argument("--config", default=str(ROOT / "core" / "cameras.yaml"), help="откуда взять модель позы")
    b.add_argument("--cache", default="out/cache/behavior", help="кэш выходов позы ('' — без кэша)")
    b.add_argument("--fps", type=float, default=10.0, help="шаг обработки, как в конвейере")
    b.add_argument("--min-overlap", type=float, default=0.5, help="с: сколько окно должно перекрыть действие")
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--threads", type=int, default=4, help="потоков ONNX Runtime на процесс")
    a = ap.parse_args(argv)
    logging.basicConfig(level="WARNING", format="%(levelname)-7s %(name)s: %(message)s")
    if a.cmd == "manifest":
        if a.kth:
            rows = manifest_kth(Path(a.kth), a.jogging)
        elif a.urfall:
            rows = manifest_urfall(Path(a.urfall))
        elif a.ours:
            rows = manifest_ours(Path(a.ours), Path(a.videos), a.fall_sec)
        else:
            raise SystemExit("укажите --kth, --urfall или --ours")
        write_manifest(rows, Path(a.out))
    else:
        build(a)


if __name__ == "__main__":
    main()
