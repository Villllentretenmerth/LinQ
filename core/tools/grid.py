"""Первый кадр источника с сеткой координат — чтобы снять пиксели зон и линий для cameras.yaml.

    python -m core.tools.grid D:/video/shift.mp4            # -> shift_grid.jpg рядом с текущей папкой
    python -m core.tools.grid core/demo_frames --step 50
"""
import argparse
from pathlib import Path

import cv2

from core.ingest import list_images


def first_frame(source: str):
    p = Path(source)
    if p.is_dir():
        files = list_images(source)
        return cv2.imread(str(files[0])) if files else None
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    img = first_frame(args.source)
    if img is None:
        raise SystemExit(f"не удалось прочитать кадр: {args.source}")
    h, w = img.shape[:2]
    for x in range(0, w, args.step):
        cv2.line(img, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.putText(img, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    for y in range(0, h, args.step):
        cv2.line(img, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(img, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    out = args.out or Path(f"{Path(args.source).stem or 'frame'}_grid.jpg")
    cv2.imwrite(str(out), img)
    print(f"кадр {w}×{h} с сеткой: {out}")


if __name__ == "__main__":
    main()
