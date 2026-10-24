"""Калибровочные кадры для квантования голов YOLO в int8 (SDK LinQ) — из своих роликов, папок кадров и картинок.

Кадры берутся равномерно по всем источникам и готовятся тем же препроцессингом, что в конвейере
(core.infer.preprocess: letterbox 640, RGB, /255). Результат — один массив (N, 3, 640, 640) float32:
его читает core/tools/compile_tpu.py. Пример SDK калибрует на случайном шуме — для детектора так нельзя:
пороги int8 подстраиваются под реальные активации, иначе съезжают рамки и точки скелета.

    python -m core.tools.make_calib D:/video/*.mp4 core/demo_frames --n 100 --out calib.npy

Сколько кадров: 64–128 из разных сцен (свет, дальние и ближние люди, техника). Больше — не всегда лучше:
калибровка по максимуму раздувает пороги на выбросах (см. --samplewise-percentile в compile_tpu.py).
"""
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

from core.infer import preprocess
from core.ingest import list_images

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def expand(sources: list[str]) -> list[Path]:
    out = []
    for s in sources:
        hits = glob.glob(s) or [s]                   # Windows-шелл сам маски не раскрывает
        out += [Path(h) for h in sorted(hits)]
    return out


def frames_of(path: Path, k: int):
    """k кадров равномерно по видео / папке / одна картинка."""
    if path.is_dir():
        files = list_images(str(path))
        for i in np.linspace(0, len(files) - 1, min(k, len(files))).round().astype(int) if files else []:
            img = cv2.imread(str(files[i]))
            if img is not None:
                yield img
        return
    if path.suffix.lower() in IMG_EXT:
        img = cv2.imread(str(path))
        if img is not None:
            yield img
        return
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        raise SystemExit(f"не открылось видео: {path}")
    for i in np.linspace(0, total - 1, min(k, total)).round().astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if ok:
            yield img
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", help="видео, папки кадров, картинки (можно маски *.mp4)")
    ap.add_argument("--n", type=int, default=100, help="сколько кадров всего (по умолчанию 100)")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--out", default="calib.npy")
    ap.add_argument("--preview", help="папка, куда сохранить выбранные кадры jpg — проверить глазами")
    a = ap.parse_args()

    paths = expand(a.sources)
    if not paths:
        raise SystemExit("нет источников")
    singles = sum(1 for p in paths if p.is_file() and p.suffix.lower() in IMG_EXT)
    per = max(1, (a.n - singles) // max(1, len(paths) - singles))   # видео и папкам — поровну
    tensors = []
    for p in paths:
        for img in frames_of(p, per):
            if len(tensors) >= a.n:
                break
            if a.preview:
                Path(a.preview).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(a.preview) / f"calib_{len(tensors):03d}.jpg"), img)
            tensors.append(preprocess(img, a.size)[0])
        print(f"{p}: всего кадров {len(tensors)}")
    if not tensors:
        raise SystemExit("ни одного кадра не прочитано")
    calib = np.concatenate(tensors, axis=0).astype(np.float32)
    np.save(a.out, calib)
    print(f"{a.out}: {calib.shape}, {calib.nbytes / 2**20:.0f} МБ, диапазон {calib.min():.2f}..{calib.max():.2f}")


if __name__ == "__main__":
    main()
