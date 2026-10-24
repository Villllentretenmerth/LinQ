"""Выгрузка предобученных YOLOv8 (Ultralytics) в ONNX — для конвейера на CPU и для LinQ TPU.

    pip install ultralytics onnx onnxslim        # нужен только здесь, конвейеру не нужен
    python core/export_models.py
    # -> core/models/yolov8n-pose.onnx, yolov8n.onnx          — целые модели (CPU, backend: onnx)
    #    core/models/yolov8n-pose_head.onnx, yolov8n_head.onnx — головы без декодирования (отдать организаторам)

Формат под TPU: статический вход 640×640, batch 1, без NMS в графе. yolov8n_pose есть в списке моделей,
поддерживаемых SDK LinQ (yolo11 — нет). Голова отрезается core/tools/split_onnx.py и сразу проверяется.
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ["yolov8n-pose.pt", "yolov8n.pt"]   # люди + скелет; COCO-детектор (car, truck, cell phone)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT, help="веса Ultralytics (скачаются сами)")
    ap.add_argument("--out", type=Path, default=ROOT / "core" / "models")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=13, help="13 поддерживают почти все компиляторы")
    args = ap.parse_args()

    from ultralytics import YOLO

    sys.path.insert(0, str(ROOT))
    from core.tools.split_onnx import split, verify

    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)                     # .pt скачиваются в текущую папку — не мусорим в репозитории
        try:
            for name in args.models:
                path = YOLO(name).export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True,
                                         opset=args.opset, batch=1, nms=False)
                target = args.out / Path(path).name
                shutil.move(path, target)
                head = target.with_name(target.stem + "_head.onnx")
                split(target, head)
                print(f"готово: {target}\n        {head} (разрез проверен, расхождение {verify(target, head):.1e})")
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    main()
