"""Источники кадров для конвейера: RTSP-камера, видеофайл или папка кадров.

Для RTSP — читатель «только последний кадр»: отдельный поток постоянно забирает кадры из камеры
и хранит лишь самый свежий, поэтому задержка не накапливается, даже если обработка медленнее потока.
Старые кадры выбрасываются (их число уходит в метрики узла как dropped).

Видеофайл и папка кадров в обычном режиме ведут себя как камера: поток отдаёт кадры с темпом исходного FPS
(для папки — с FPS из конфига), обработка берёт последний. В офлайн-режиме (`--offline`) кадры читаются
подряд без пропусков и без пауз, а время кадра считается по его номеру — так результат не зависит от
скорости машины, а «секунды» в правилах остаются секундами видео.

Все источники отдают `Frame`: `ts` — время кадра в потоке (по нему считаются окна, скорость и неподвижность),
`recv_ts` — когда кадр получен конвейером (по нему журнал считает задержку конвейера).
"""
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Frame:
    image: np.ndarray
    ts: float        # время кадра в потоке, с (unix)
    recv_ts: float   # время получения кадра конвейером, с (unix)
    seq: int         # номер кадра у источника, с 1
    skipped: int = 0  # сколько кадров источника выброшено с прошлой выдачи


class LatestFrameReader:
    """RTSP (или любой поток OpenCV): фоновый поток держит только самый свежий кадр."""

    reconnect_after_sec = 3.0

    def __init__(self, url: str):
        self.url = url
        self.cap = self._open()
        self.frame, self.ts, self.seq, self.dropped = None, 0.0, 0, 0
        self.eof = False                     # у живой камеры конца нет; для файла см. VideoFileReader
        self._last_read_seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True, name=f"reader:{url}").start()

    def _open(self):
        src = int(self.url) if str(self.url).isdigit() else self.url   # «0» — веб-камера
        cap = cv2.VideoCapture(src) if isinstance(src, int) else cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _publish(self, f, ts: float) -> None:
        with self._lock:
            self.frame, self.ts = f, ts
            self.seq += 1

    def _loop(self):
        failed_since = None
        while not self._stop.is_set():
            ok, f = self.cap.read()
            if not ok:
                failed_since = failed_since or time.monotonic()
                if time.monotonic() - failed_since > self.reconnect_after_sec:   # камера отвалилась — переподключаемся
                    self.cap.release()
                    self.cap = self._open()
                    failed_since = None
                time.sleep(0.05)
                continue
            failed_since = None
            self._publish(f, time.time())   # метка приёма — для замера задержки конвейера

    def read(self) -> Frame | None:
        """Самый свежий кадр и сколько кадров пропущено с прошлого чтения (None, пока кадров нет)."""
        with self._lock:
            if self.frame is None:
                return None
            skipped = max(0, self.seq - self._last_read_seq - 1)
            self._last_read_seq = self.seq
            self.dropped += skipped
            return Frame(self.frame, self.ts, self.ts, self.seq, skipped)

    def close(self):
        self._stop.set()
        self.cap.release()


class VideoFileReader(LatestFrameReader):
    """Видеофайл как камера: кадры появляются с темпом исходного FPS, обработка берёт последний."""

    def __init__(self, path: str, loop: bool = False):
        self.loop = loop
        super().__init__(path)
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0

    def _open(self):
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            raise FileNotFoundError(f"Не удалось открыть видео: {self.url}")
        return cap

    def _loop(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        t0, n = time.time(), 0
        while not self._stop.is_set():
            ok, f = self.cap.read()
            if not ok:
                if self.loop and n:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.eof = True
                return
            ts = t0 + n / fps
            n += 1
            time.sleep(max(0.0, ts - time.time()))   # темп исходного видео
            self._publish(f, ts)


class ImageFolderReader(LatestFrameReader):
    """Папка кадров (имитация стоп-кадров) как камера: кадр за кадром с заданным FPS, по имени файла."""

    def __init__(self, path: str, fps: float, loop: bool = False):
        self.files = list_images(path)
        if not self.files:
            raise FileNotFoundError(f"В папке нет изображений: {path}")
        self.src_fps, self.loop = fps, loop
        super().__init__(path)

    def _open(self):
        return None

    def _loop(self):
        t0, n = time.time(), 0
        while not self._stop.is_set():
            if n >= len(self.files) and not self.loop:
                self.eof = True
                return
            f = cv2.imread(str(self.files[n % len(self.files)]))
            ts = t0 + n / self.src_fps
            n += 1
            if f is None:
                continue
            time.sleep(max(0.0, ts - time.time()))
            self._publish(f, ts)

    def close(self):
        self._stop.set()


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def list_images(path: str) -> list[Path]:
    """Изображения папки, отсортированные по имени (frame2 раньше frame10)."""
    return sorted((p for p in Path(path).iterdir() if p.suffix.lower() in IMAGE_EXT), key=_natural_key)


def fixed_rate(reader: LatestFrameReader, fps: float) -> Iterator[Frame]:
    """Отдаёт кадры с ровным шагом по времени — так фильтр Калмана в ByteTrack прогнозирует корректно
    (sv.ByteTrack(frame_rate=fps)). Отдаются только новые кадры: если источник не прислал кадр с прошлого
    шага, шаг пропускается, а не повторяет старый кадр. Заканчивается, когда у файла/папки кончились кадры."""
    step = 1.0 / fps
    nxt = time.monotonic()
    last_seq = 0
    while True:
        nxt += step
        now = time.monotonic()
        if nxt < now - step:     # обработка отстала (тяжёлый кадр) — не догоняем пачкой, а идём дальше ровно
            nxt = now
        fr = reader.read()
        if fr is not None and fr.seq != last_seq:
            last_seq = fr.seq
            yield fr
        elif reader.eof:
            return
        time.sleep(max(0.0, nxt - time.monotonic()))


def offline_frames(source: str, fps: float) -> Iterator[Frame]:
    """Покадровое чтение файла или папки без пропусков: берём каждый кадр, ближайший к шагу 1/fps по времени
    видео. Время кадра — по номеру, отсчёт от момента запуска."""
    t0 = time.time()
    p = Path(source)
    if p.is_dir():
        for i, f in enumerate(list_images(source)):
            img = cv2.imread(str(f))
            if img is not None:
                yield Frame(img, t0 + i / fps, time.time(), i + 1)
        return
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    take_every = max(1.0, src_fps / fps)      # 25 FPS видео при обработке 10 FPS — каждый 2,5-й кадр
    n, nxt = 0, 0.0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                return
            if n + 1e-6 >= nxt:
                ok, img = cap.retrieve()
                if ok:
                    yield Frame(img, t0 + n / src_fps, time.time(), n + 1)
                nxt += take_every
            n += 1
    finally:
        cap.release()


def is_stream(source: str) -> bool:
    return bool(re.match(r"^(rtsp|rtsps|rtmp|http|https|udp|tcp)://", str(source))) or str(source).isdigit()


def open_source(source: str, fps: float, offline: bool = False, loop: bool = False) -> tuple[Iterator[Frame], LatestFrameReader | None]:
    """Итератор кадров для конвейера и сам читатель (чтобы закрыть и взять dropped); в офлайне читателя нет."""
    source = str(source)
    if offline and not is_stream(source):
        return offline_frames(source, fps), None
    if is_stream(source):
        reader = LatestFrameReader(source)
    elif Path(source).is_dir():
        reader = ImageFolderReader(source, fps, loop=loop)
    else:
        reader = VideoFileReader(source, loop=loop)
    return fixed_rate(reader, fps), reader
