"""Кадр-доказательство: зоны, рамка нарушителя с подписью, второй участник (техника, телефон), время."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from .geometry import Scene

RED, ORANGE, GRAY, ZONE = (40, 40, 230), (0, 140, 255), (170, 170, 170), (0, 200, 255)
_FONT_CANDIDATES = ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                    "/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"]
_font_cache: dict[int, object] = {}


def _font(size: int):
    """TTF с кириллицей, если есть Pillow и шрифт; иначе None (подписи латиницей через OpenCV)."""
    if size in _font_cache:
        return _font_cache[size]
    font = None
    try:
        from PIL import ImageFont
        for path in _FONT_CANDIDATES:
            if Path(path).exists():
                font = ImageFont.truetype(path, size)
                break
    except Exception:
        font = None
    _font_cache[size] = font
    return font


def _texts(img: np.ndarray, items: list[tuple[str, tuple[int, int], tuple[int, int, int]]], size: int) -> np.ndarray:
    """Подписи на плашках. items: (текст, левый верхний угол, цвет плашки BGR)."""
    font = _font(size)
    if font is None:
        for text, (x, y), color in items:
            t = text.encode("ascii", "replace").decode()
            (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, size / 30, 2)
            cv2.rectangle(img, (x, y), (x + tw + 8, y + th + 10), color, -1)
            cv2.putText(img, t, (x + 4, y + th + 4), cv2.FONT_HERSHEY_SIMPLEX, size / 30, (255, 255, 255), 2, cv2.LINE_AA)
        return img
    from PIL import Image, ImageDraw
    pil = Image.fromarray(img[:, :, ::-1])
    draw = ImageDraw.Draw(pil)
    for text, (x, y), color in items:
        l, t, r, b = draw.textbbox((x + 5, y + 3), text, font=font)
        draw.rectangle((x, y, r + 5, b + 5), fill=color[::-1])
        draw.text((x + 5, y + 3), text, font=font, fill=(255, 255, 255))
    return np.ascontiguousarray(np.asarray(pil)[:, :, ::-1])


def draw_evidence(frame: np.ndarray, scene: Scene, bbox, label: str, color=RED, other_bbox=None,
                  other_label: str | None = None, persons: list | None = None, ts: float | None = None,
                  camera: str = "", tz: str = "Europe/Moscow") -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    lw = max(2, int(round(min(h, w) / 300)))
    fs = max(14, int(round(min(h, w) / 32)))

    overlay = img.copy()
    for z in scene.zones:
        cv2.fillPoly(overlay, [z.polygon.astype(np.int32)], ZONE)
    img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
    for z in scene.zones:
        cv2.polylines(img, [z.polygon.astype(np.int32)], True, ZONE, lw)
    for ln in scene.lines:
        cv2.line(img, tuple(map(int, ln.p1)), tuple(map(int, ln.p2)), (255, 0, 255), lw + 1)
    for p in persons or []:
        x1, y1, x2, y2 = map(int, p)
        cv2.rectangle(img, (x1, y1), (x2, y2), GRAY, 1)

    items = []
    for z in scene.zones:
        x, y = map(int, z.polygon.min(axis=0))
        items.append((z.name, (x, y), (60, 60, 60)))
    if other_bbox is not None:
        x1, y1, x2, y2 = map(int, other_bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), ORANGE, lw)
        if other_label:
            items.append((other_label, (x1, max(0, y1 - fs - 12)), ORANGE))
        if bbox is not None:
            a = (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))
            b = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.line(img, a, b, ORANGE, lw)
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, lw + 1)
        items.append((label, (x1, max(0, y1 - fs - 12)), color))
    if ts is not None:
        t = datetime.fromtimestamp(ts, ZoneInfo(tz)).strftime("%Y-%m-%d %H:%M:%S")
        items.append((f"{camera}  {t}", (8, h - fs - 16), (30, 30, 30)))
    return _texts(img, items, fs)


def to_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("не удалось закодировать JPEG")
    return buf.tobytes()
