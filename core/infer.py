"""Адаптер инференса с единым интерфейсом для CPU (ONNX Runtime) и LinQ TPU.

Бэкенд знает только тензоры: на вход — препроцессированный тензор NCHW (1, 3, 640, 640), на выход — список
выходных тензоров сети. Препроцессинг (letterbox, RGB, /255) и постпроцессинг (декодирование головы, порог,
NMS, перевод рамок и ключевых точек в координаты кадра) — на CPU, одинаково для всех бэкендов:

    onnx       — целая модель на ONNX Runtime (разработка);
    onnx_head  — голова (*_head.onnx) на ONNX Runtime + хвост на numpy: тот же путь данных, что с TPU;
    linq       — голова, скомпилированная в программу TPU, через pytpu + тот же хвост на numpy.

Модели — YOLOv8 из Ultralytics в ONNX без NMS в графе (статический вход 640, batch 1). yolov8n_pose есть
в списке поддерживаемых моделей SDK LinQ. Выход целой модели (и хвоста):
    yolo-pose: (1, 56, 8400) = cx, cy, w, h, score, 17 × (x, y, conf)
    yolo-det:  (1, 4 + C, 8400) = cx, cy, w, h, C оценок классов (COCO: C = 80)
"""
import ast
import importlib
import threading
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


# ---------------------------------------------------------------- бэкенды

class Backend(Protocol):
    name: str
    input_shape: tuple[int, int, int, int]
    class_names: list[str] | None

    def run(self, tensor: np.ndarray) -> list[np.ndarray]: ...


class OnnxBackend:
    """ONNX Runtime на CPU — для разработки и для замера «до TPU»."""

    def __init__(self, path: str | Path, threads: int | None = None, providers: list[str] | None = None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.name = f"onnx:{Path(path).name}"
        self.sess = ort.InferenceSession(str(path), so, providers=providers or ["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        self.input_shape = tuple(shape)
        self.class_names = _names_from_metadata(self.sess.get_modelmeta().custom_metadata_map)

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        return self.sess.run(None, {self.input_name: tensor})


class YoloHeadDecoder:
    """Хвост YOLOv8 на numpy: сырые выходы последних свёрток головы (то, что считает TPU) → тот же тензор,
    что у целой модели: (1, 4 + nc [+ 51], 8400). Выходы узнаются по форме, а не по имени: 64 канала — рамки
    (DFL, reg_max = 16), 51 — ключевые точки, остальное — классы; масштаб — по размеру карты (80/40/20 → шаг 8/16/32)."""

    def __init__(self, kpt: bool, imgsz: int = 640, reg_max: int = 16):
        self.kpt, self.imgsz, self.reg_max = kpt, imgsz, reg_max
        self._grid: dict = {}

    def _anchors(self, sizes: list[tuple[int, int, int]]):
        key = tuple(sizes)
        if key not in self._grid:
            pts, st = [], []
            for s, h, w in sizes:
                sy, sx = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
                pts.append(np.stack([sx.ravel(), sy.ravel()]))
                st.append(np.full(h * w, s, np.float32))
            self._grid[key] = (np.concatenate(pts, 1).astype(np.float32), np.concatenate(st))
        return self._grid[key]

    def __call__(self, outs: list[np.ndarray]) -> np.ndarray:
        by: dict[int, dict] = {}
        for o in outs:
            o = np.asarray(o, np.float32)
            o = o[None] if o.ndim == 3 else o
            _, c, h, w = o.shape
            kind = "box" if c == 4 * self.reg_max else ("kpt" if self.kpt and c == 51 else "cls")
            d = by.setdefault(int(round(self.imgsz / h)), {"hw": (h, w)})
            d[kind] = o.reshape(c, h * w)
        strides = sorted(by)
        anchors, st = self._anchors([(s, *by[s]["hw"]) for s in strides])
        box = np.concatenate([by[s]["box"] for s in strides], 1)
        cls = np.concatenate([by[s]["cls"] for s in strides], 1)
        a = box.shape[1]
        b = box.reshape(4, self.reg_max, a)
        b = np.exp(b - b.max(1, keepdims=True))
        dist = (b / b.sum(1, keepdims=True) * np.arange(self.reg_max, dtype=np.float32)[None, :, None]).sum(1)
        x1y1, x2y2 = anchors - dist[:2], anchors + dist[2:]
        parts = [np.concatenate([(x1y1 + x2y2) / 2, x2y2 - x1y1]) * st, 1 / (1 + np.exp(-cls))]
        if self.kpt:
            k = np.concatenate([by[s]["kpt"] for s in strides], 1).reshape(17, 3, a)
            xy = (k[:, :2] * 2.0 + (anchors[None] - 0.5)) * st
            parts.append(np.concatenate([xy, 1 / (1 + np.exp(-k[:, 2:3]))], 1).reshape(51, a))
        return np.concatenate(parts, 0)[None].astype(np.float32)


class OnnxHeadBackend:
    """Голова на ONNX Runtime + хвост на numpy — тот же путь данных, что с TPU, но на CPU (проверка разреза)."""

    def __init__(self, path: str | Path, kpt: bool, threads: int | None = None):
        self.head = OnnxBackend(path, threads)
        self.name = f"onnx-head:{Path(path).name}"
        self.input_shape, self.class_names = self.head.input_shape, self.head.class_names
        self.decode = YoloHeadDecoder(kpt, self.input_shape[-1])

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        return self.head.run(tensor)

    def finish(self, outs: list[np.ndarray]) -> list[np.ndarray]:
        return [self.decode(outs)]


_TPU_DEVICE = None   # одно устройство на процесс, на нём — программы всех моделей


def _tpu_device(index: int = 0):
    global _TPU_DEVICE
    if _TPU_DEVICE is None:
        import pytpu as tpu

        devices = tpu.Device.list_devices()
        if not devices:
            raise RuntimeError("TPU не найден: проверьте ls /dev/tpu*")
        _TPU_DEVICE = tpu.Device.open(devices[index]).__enter__()
    return _TPU_DEVICE


class LinqBackend:
    """LinQ TPU через pytpu (как в usage_example_python_api): программа .tpu — голова YOLO, скомпилированная
    SDK из *_head.onnx (core/tools/split_onnx.py → core/tools/compile_tpu.py); хвост — YoloHeadDecoder на CPU.

    run() — только вызов TPU (это время уходит в метрики как загрузка TPU), finish() — декодирование на CPU.
    Можно подменить вызов своей функцией: module: "my_linq:make_infer" → make_infer(program, **opts)
    возвращает infer(tensor) -> список/словарь выходов."""

    def __init__(self, program: str, kpt: bool = False, module: str | None = None, infer_fn: Callable | None = None,
                 input_name: str = "images", input_shape=(1, 3, 640, 640), class_names: list[str] | None = None,
                 device_index: int = 0, **opts):
        self.name = f"linq:{Path(program).name}"
        self.input_name = input_name
        self.input_shape = tuple(input_shape)
        self.class_names = class_names
        self.decode = YoloHeadDecoder(kpt, self.input_shape[-1])
        if infer_fn is None and module:
            mod_name, _, fn_name = module.partition(":")
            infer_fn = getattr(importlib.import_module(mod_name), fn_name or "make_infer")(program, **opts)
        if infer_fn is None:
            prog = _tpu_device(device_index).load(str(program)).__enter__()
            session = prog.inference().__enter__()
            call = getattr(session, "sync", None) or session.run
            infer_fn = lambda t: call({self.input_name: t})      # noqa: E731
        self.infer_fn = infer_fn
        self._lock = threading.Lock()                            # камеры в разных потоках, сессия TPU одна

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        with self._lock:
            outs = self.infer_fn(np.ascontiguousarray(tensor, dtype=np.float32))
        if isinstance(outs, dict):
            outs = list(outs.values())
        return [np.asarray(o, dtype=np.float32) for o in (outs if isinstance(outs, (list, tuple)) else [outs])]

    def finish(self, outs: list[np.ndarray]) -> list[np.ndarray]:
        return [self.decode(outs)]


def _names_from_metadata(meta: dict) -> list[str] | None:
    """Ultralytics кладёт имена классов в метаданные ONNX: "{0: 'person', 1: 'bicycle', ...}"."""
    raw = meta.get("names") if meta else None
    if not raw:
        return None
    try:
        d = ast.literal_eval(raw)
        return [d[i] for i in sorted(d)]
    except Exception:
        return None


def make_backend(kind: str, path: str, kpt: bool, linq_opts: dict | None = None,
                 threads: int | None = None) -> Backend:
    if kind == "onnx":
        return OnnxBackend(path, threads=threads)
    if kind == "onnx_head":
        return OnnxHeadBackend(path, kpt, threads=threads)
    if kind == "linq":
        return LinqBackend(path, kpt, **(linq_opts or {}))
    raise ValueError(f"Неизвестный бэкенд: {kind} (onnx | onnx_head | linq)")


def _infer(backend, tensor: np.ndarray, timer, name: str) -> np.ndarray:
    with (timer(name) if timer else nullcontext()):   # в метрики — только время ускорителя
        outs = backend.run(tensor)
    finish = getattr(backend, "finish", None)            # хвост головы на CPU — вне таймера TPU
    return (finish(outs) if finish else outs)[0]


# ---------------------------------------------------------------- пре- и постпроцессинг

@dataclass
class Letterbox:
    ratio: float
    pad: tuple[float, float]   # dx, dy

    def to_frame(self, xy: np.ndarray) -> np.ndarray:
        """Координаты входа сети -> координаты кадра (последняя ось — x, y)."""
        out = xy.astype(np.float32).copy()
        out[..., 0] = (out[..., 0] - self.pad[0]) / self.ratio
        out[..., 1] = (out[..., 1] - self.pad[1]) / self.ratio
        return out


def preprocess(img_bgr: np.ndarray, size: int = 640) -> tuple[np.ndarray, Letterbox]:
    """Letterbox как в Ultralytics: вписать с сохранением пропорций, добить серым 114 по центру."""
    h, w = img_bgr.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dx, dy = (size - nw) / 2, (size - nh) / 2
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img_bgr
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = int(round(dy - 0.1)), int(round(dx - 0.1))
    canvas[top:top + nh, left:left + nw] = resized
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor), Letterbox(r, (left, top))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Классический NMS на CPU. boxes: (N, 4) xyxy. Возвращает индексы оставшихся по убыванию оценки."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=int)


def _cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] - b[:, 2] / 2
    out[:, 1] = b[:, 1] - b[:, 3] / 2
    out[:, 2] = b[:, 0] + b[:, 2] / 2
    out[:, 3] = b[:, 1] + b[:, 3] / 2
    return out


def _clip_boxes(b: np.ndarray, w: int, h: int) -> np.ndarray:
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    return b


@dataclass
class PoseResult:
    xyxy: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))     # (N, 4) в пикселях кадра
    conf: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))          # (N,)
    kpts: np.ndarray = field(default_factory=lambda: np.zeros((0, 17, 3), np.float32))  # (N, 17, x/y/conf)


@dataclass
class DetResult:
    xyxy: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    conf: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    cls: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    names: list[str] = field(default_factory=list)   # имя класса для каждой рамки

    def subset(self, wanted: set[str]) -> "DetResult":
        m = np.array([n in wanted for n in self.names], dtype=bool)
        return DetResult(self.xyxy[m], self.conf[m], self.cls[m], [n for n, k in zip(self.names, m) if k])


def postprocess_pose(out: np.ndarray, lb: Letterbox, frame_wh: tuple[int, int],
                     conf_thr: float = 0.25, iou_thr: float = 0.45, max_det: int = 100) -> PoseResult:
    p = np.asarray(out, dtype=np.float32)
    p = p[0] if p.ndim == 3 else p
    if p.shape[0] < p.shape[1]:          # (56, 8400) -> (8400, 56)
        p = p.T
    p = p[p[:, 4] >= conf_thr]
    if not len(p):
        return PoseResult()
    boxes = _cxcywh_to_xyxy(p[:, :4])
    keep = nms(boxes, p[:, 4], iou_thr)[:max_det]
    p, boxes = p[keep], boxes[keep]
    kpts = p[:, 5:5 + 51].reshape(-1, 17, 3).copy()
    kpts[..., :2] = lb.to_frame(kpts[..., :2])
    boxes = _clip_boxes(lb.to_frame(boxes.reshape(-1, 2, 2)).reshape(-1, 4), *frame_wh)
    return PoseResult(boxes, p[:, 4].copy(), kpts)


def postprocess_det(out: np.ndarray, lb: Letterbox, frame_wh: tuple[int, int], class_names: list[str],
                    wanted: set[str] | None = None, conf_thr: float = 0.3, iou_thr: float = 0.45,
                    max_det: int = 100) -> DetResult:
    p = np.asarray(out, dtype=np.float32)
    p = p[0] if p.ndim == 3 else p
    if p.shape[0] < p.shape[1]:          # (84, 8400) -> (8400, 84)
        p = p.T
    scores_all = p[:, 4:]
    if wanted is not None:               # считаем только нужные классы — остальные не мешают NMS
        idx = [i for i, n in enumerate(class_names) if n in wanted]
        mask = np.zeros(scores_all.shape[1], bool)
        mask[idx] = True
        scores_all = np.where(mask[None], scores_all, 0)
    cls = scores_all.argmax(1)
    conf = scores_all[np.arange(len(cls)), cls]
    m = conf >= conf_thr
    if not m.any():
        return DetResult()
    boxes, conf, cls = _cxcywh_to_xyxy(p[m, :4]), conf[m], cls[m]
    keep = nms(boxes + cls[:, None] * 4096.0, conf, iou_thr)[:max_det]   # NMS по классам
    boxes = _clip_boxes(lb.to_frame(boxes[keep].reshape(-1, 2, 2)).reshape(-1, 4), *frame_wh)
    cls = cls[keep]
    return DetResult(boxes, conf[keep], cls, [class_names[c] if c < len(class_names) else str(c) for c in cls])


# ---------------------------------------------------------------- модели конвейера

class PoseModel:
    """yolo-pose: люди + 17 ключевых точек COCO."""

    name = "yolo-pose"

    def __init__(self, backend: Backend, conf: float = 0.25, iou: float = 0.45):
        self.backend, self.conf, self.iou = backend, conf, iou
        self.size = backend.input_shape[-1]

    def __call__(self, img: np.ndarray, timer: Callable | None = None) -> PoseResult:
        tensor, lb = preprocess(img, self.size)
        out = _infer(self.backend, tensor, timer, self.name)
        return postprocess_pose(out, lb, (img.shape[1], img.shape[0]), self.conf, self.iou)


class DetModel:
    """yolo-det: техника, телефон и т. п. Пока своей модели нет — COCO-детектор (car, truck, cell phone)."""

    name = "yolo-det"

    def __init__(self, backend: Backend, wanted: set[str] | None = None, conf: float = 0.3, iou: float = 0.45,
                 class_conf: dict[str, float] | None = None):
        self.backend, self.conf, self.iou = backend, conf, iou
        self.class_names = backend.class_names or COCO80
        self.wanted = wanted
        self.class_conf = class_conf or {}
        self.size = backend.input_shape[-1]

    @property
    def classes(self) -> set[str]:
        return set(self.class_names)

    def __call__(self, img: np.ndarray, timer: Callable | None = None) -> DetResult:
        tensor, lb = preprocess(img, self.size)
        out = _infer(self.backend, tensor, timer, self.name)
        thr = min([self.conf, *self.class_conf.values()])
        res = postprocess_det(out, lb, (img.shape[1], img.shape[0]), self.class_names, self.wanted, thr, self.iou)
        if self.class_conf:             # у мелких объектов (телефон) свой порог
            keep = np.array([c >= self.class_conf.get(n, self.conf) for c, n in zip(res.conf, res.names)], bool)
            res = DetResult(res.xyxy[keep], res.conf[keep], res.cls[keep], [n for n, k in zip(res.names, keep) if k])
        return res
