"""Скелеты из видео открытых наборов той же YOLO-pose, что работает на камере (10 к/с, 17 точек COCO).

Результат — pkl в формате prepare.py (как PYSKL, но уже 10 к/с и с именем класса):

    python ml/behavior/extract_pose.py --dataset kth --videos datasets/kth --out datasets/kth_pose.pkl
    python ml/behavior/extract_pose.py --dataset caucafall --videos datasets/CAUCAFall --out datasets/cauca_pose.pkl
    python ml/behavior/extract_pose.py --dataset urfd --videos datasets/urfd --out datasets/urfd_pose.pkl

Классы по наборам:
- kth: running, jogging -> run; walking (и остальное) -> normal. Тест — люди 22–25.
- caucafall: папки «Fall …» -> fall (кадры падения — по разметке набора, если найдена), остальное -> normal. Тест — субъекты 9–10.
- urfd: fall-NN -> fall, adl-NN -> normal. Тест — каждый пятый ролик.
"""
import argparse
import pickle
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.infer import PoseModel, make_backend  # noqa: E402

VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg"}


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def video_pose(path: Path, model: PoseModel, fps: float):
    """Главный человек ролика по кадрам: продолжаем того, кто ближе к прошлому, иначе — самого крупного."""
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1.0, src_fps / fps)
    kps, scs, frames, n, nxt, prev = [], [], [], 0, 0.0, None
    while True:
        ok = cap.grab()
        if not ok:
            break
        if n + 1e-6 >= nxt:
            ok, img = cap.retrieve()
            if ok:
                r = model(img)
                kp, sc = np.zeros((17, 2), np.float32), np.zeros(17, np.float32)
                if len(r.conf):
                    area = (r.xyxy[:, 2] - r.xyxy[:, 0]) * (r.xyxy[:, 3] - r.xyxy[:, 1])
                    score = r.conf * area
                    if prev is not None:
                        score = score * (0.2 + np.array([iou(b, prev) for b in r.xyxy]))
                    j = int(np.argmax(score))
                    if r.conf[j] >= 0.3:
                        kp, sc = r.kpts[j, :, :2], r.kpts[j, :, 2]
                        prev = r.xyxy[j]
                kps.append(kp)
                scs.append(sc)
                frames.append(n)
            nxt += step
        n += 1
    cap.release()
    return np.array(kps, np.float32), np.array(scs, np.float32), np.array(frames), src_fps


def label_kth(p: Path):
    m = re.match(r"person(\d+)_(\w+?)_d\d", p.stem)
    if not m:
        return None
    act, person = m.group(2), int(m.group(1))
    return ("run" if act in ("running", "jogging") else "normal"), person >= 22, f"kth:{person}"


def label_urfd(p: Path):
    m = re.match(r"(fall|adl)-(\d+)", p.stem)
    if not m:
        return None
    return ("fall" if m.group(1) == "fall" else "normal"), int(m.group(2)) % 5 == 0, f"urfd:{p.stem}"


def label_cauca(p: Path):
    parts = [x.lower() for x in p.parts]
    subj = next((int(re.search(r"(\d+)", x).group(1)) for x in parts if x.startswith("subject") and re.search(r"\d+", x)), 0)
    act = p.parent.name.lower()
    return ("fall" if "fall" in act else "normal"), subj >= 9, f"cauca:{subj}:{act}"


def cauca_fall_frames(p: Path, frames: np.ndarray):
    """Покадровая разметка CAUCAFall (если рядом с видео есть .txt на каждый кадр). Формат уточним по файлам;
    пока не узнан — None, и момент падения prepare.py найдёт по скелету (бёдра резко опустились)."""
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["kth", "urfd", "caucafall"])
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=str(ROOT / "core" / "models" / "yolov8n-pose.onnx"))
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0, help="только первые N роликов (проверка)")
    ap.add_argument("--max-sec", type=float, default=0, help="остановиться через N секунд (продолжить следующим запуском)")
    a = ap.parse_args()
    model = PoseModel(make_backend("onnx", a.model, True), conf=0.25)
    labeler = {"kth": label_kth, "urfd": label_urfd, "caucafall": label_cauca}[a.dataset]
    vids = sorted(p for p in Path(a.videos).rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if a.dataset == "urfd":
        vids = [v for v in vids if "cam1" not in v.name]          # потолочная камера — не наш ракурс
    if a.limit:
        vids = vids[:a.limit]
    anns, split = [], {"train": [], "test": []}
    part_dir = Path(a.out).with_suffix(".parts")            # по файлу на ролик: прерванный запуск продолжается
    part_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done_all = True
    for i, v in enumerate(vids):
        lab = labeler(v)
        if lab is None:
            continue
        name, is_test, group = lab
        rel = str(v.relative_to(a.videos))
        part = part_dir / (re.sub(r"[^\w.-]", "_", rel) + ".pkl")
        if part.exists():
            ann = pickle.load(open(part, "rb"))
        else:
            if a.max_sec and time.time() - t0 > a.max_sec:
                done_all = False
                break
            kp, sc, frames, src_fps = video_pose(v, model, a.fps)
            ann = {"frame_dir": rel, "label_name": name, "group": group, "fps": a.fps,
                   "keypoint": kp[None], "keypoint_score": sc[None], "total_frames": len(kp)}
            if a.dataset == "caucafall" and name == "fall":
                m = cauca_fall_frames(v, frames)
                if m is not None and m.any():
                    ann["fall_mask"] = m
            pickle.dump(ann, open(part, "wb"))
            el = time.time() - t0
            print(f"[{i + 1}/{len(vids)}] {v.name}: {name}, {len(kp)} кадров, видно человека в "
                  f"{np.mean(sc.max(1) > 0) if len(sc) else 0:.0%}; {el / 60:.1f} мин", flush=True)
        if ann["total_frames"] < 5:
            continue
        anns.append(ann)
        split["test" if is_test else "train"].append(ann["frame_dir"])
    if not done_all:
        print(f"ПРОДОЛЖИТЬ: готово {len(list(part_dir.glob('*.pkl')))} из {len(vids)} роликов — запустите ещё раз")
        return
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump({"annotations": anns, "split": split}, f)
    print("сохранено:", a.out, "роликов:", len(anns))


if __name__ == "__main__":
    main()
