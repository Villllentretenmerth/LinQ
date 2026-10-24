"""Отрезать «голову» YOLOv8 (Ultralytics, ONNX) для LinQ TPU; декодирование остаётся на CPU.

В документации SDK LinQ yolov8n_pose компилируется не целиком: граф обрезается перед декодированием
головы, а оно переносится в постпроцессинг. Мы режем по выходам последнего Conv каждой ветки головы
Detect/Pose (cv2 — рамки, cv3 — классы, cv4 — ключевые точки, по три масштаба). На TPU остаются только
Conv, SiLU, Concat, Split, MaxPool, Resize, Add; DFL softmax, сигмоиды и перевод в пиксели делает
core.infer.YoloHeadDecoder на numpy, поэтому int8 не портит координаты и onnxruntime на узле не нужен.

    pip install onnx onnxruntime
    python -m core.tools.split_onnx core/models/yolov8n-pose.onnx --image кадр.jpg
    # -> core/models/yolov8n-pose_head.onnx — его отдаём организаторам на компиляцию

Проверка: голова на ONNX Runtime + YoloHeadDecoder должны дать тот же выход, что и целая модель.
"""
import argparse
import re
from pathlib import Path

import numpy as np

HEAD_CONV = re.compile(r"^/model\.(\d+)/cv(\d)\.(\d)/cv\2\.\3\.2/Conv$")


def find_cuts(model) -> list[str]:
    """Выходы последних Conv веток головы, по масштабу, затем по ветке: [cv2.0, cv3.0, cv4.0, cv2.1, ...]."""
    found = []
    for n in model.graph.node:
        m = HEAD_CONV.match(n.name)
        if n.op_type == "Conv" and m:
            found.append((int(m.group(1)), int(m.group(3)), int(m.group(2)), n.output[0]))
    if not found:
        raise SystemExit("не нашёл свёрток головы /model.N/cvX.Y/cvX.Y.2/Conv — это выгрузка Ultralytics YOLOv8?")
    last = max(f[0] for f in found)               # индекс модуля головы (22 у yolov8n)
    return [f[3] for f in sorted(f for f in found if f[0] == last)]


def split(src: Path, head_path: Path) -> list[str]:
    import onnx
    from onnx.utils import Extractor

    model = onnx.shape_inference.infer_shapes(onnx.load(str(src)))
    cuts = find_cuts(model)
    head = Extractor(model).extract_model([i.name for i in model.graph.input], cuts)
    for p in model.metadata_props:                # имена классов и пр. из выгрузки Ultralytics
        head.metadata_props.add(key=p.key, value=p.value)
    onnx.save(head, str(head_path))
    return cuts


def verify(src: Path, head_path: Path, image: str | None = None) -> float:
    from core.infer import OnnxBackend, OnnxHeadBackend, preprocess

    full = OnnxBackend(src)
    head = OnnxHeadBackend(head_path, kpt=None)
    if image:
        import cv2
        x, _ = preprocess(cv2.imread(image), full.input_shape[-1])
    else:
        x = np.random.default_rng(0).random(full.input_shape, dtype=np.float32)
    ref = full.run(x)[0]
    raw = head.run(x)
    head.decode.kpt = any(o.shape[1] == 51 for o in raw)
    return float(np.max(np.abs(ref - head.finish(raw)[0])))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--image", help="кадр для проверки (иначе случайный тензор)")
    args = ap.parse_args()
    head = args.onnx.with_name(args.onnx.stem + "_head.onnx")
    cuts = split(args.onnx, head)
    print(f"голова для TPU: {head}\nвыходы ({len(cuts)}):")
    for c in cuts:
        print("  ", c)
    diff = verify(args.onnx, head, args.image)
    print(f"проверка: max |целая − (голова + декодер на CPU)| = {diff:.1e} px")
    if diff > 1e-2:
        raise SystemExit("расхождение слишком большое — разрез неверный")


if __name__ == "__main__":
    main()
