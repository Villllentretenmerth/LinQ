"""Bounded JPEG history, event windows, and background browser-compatible MP4 encoding.

Capture calls add_frame BEFORE publishing the latest frame to inference. Recording therefore
continues while inference is busy. Windows are relative to the confirming source frame,
not the wall clock at which a slow model finishes. Audio is not captured.
"""
import logging
import math
import queue
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def clip_window(evidence):
    value = (evidence or {}).get("clip_sec")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start, end = map(float, value)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(start) and math.isfinite(end) and -120 <= start <= 0 <= end <= 120):
        return None
    return -start, end


@dataclass(frozen=True)
class Sample:
    ts: float
    jpeg: bytes


@dataclass
class Job:
    key: str
    start: float
    end: float
    anchor: float
    samples: list = field(default_factory=list)
    size: int = 0
    truncated: bool = False


def encode_mp4(path, samples, fps):
    """Preserve source time, duplicating the previous image across capture gaps."""
    import imageio_ffmpeg
    first = cv2.imdecode(np.frombuffer(samples[0].jpeg, np.uint8), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError("Не удалось прочитать кадр клипа")
    h, w = first.shape[:2]
    w, h = max(2, w // 2 * 2), max(2, h // 2 * 2)
    writer = imageio_ffmpeg.write_frames(
        str(path), (w, h), fps=fps, codec="libx264", pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p", macro_block_size=2, ffmpeg_log_level="error",
        output_params=["-preset", "veryfast", "-crf", "23", "-movflags", "+faststart"],
    )
    writer.send(None)
    index, decoded_index, rgb = 0, -1, None
    count = max(1, round((samples[-1].ts - samples[0].ts) * fps) + 1)
    try:
        for n in range(count):
            ts = samples[0].ts + n / fps
            while index + 1 < len(samples) and samples[index + 1].ts <= ts + 1e-6:
                index += 1
            if index != decoded_index:
                img = cv2.imdecode(np.frombuffer(samples[index].jpeg, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Повреждённый кадр клипа")
                rgb = cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2RGB)
                decoded_index = index
            writer.send(rgb)
    finally:
        writer.close()
    # Verify with the same bundled decoder, unaffected by RTSP-specific OpenCV flags.
    check = imageio_ffmpeg.read_frames(str(path))
    try:
        next(check)  # stream metadata
        next(check)  # at least one decoded image
    finally:
        check.close()



class ClipRecorder:
    def __init__(self, directory, on_ready, on_update, *, fps=10, buffer_sec=30,
                 max_width=960, max_buffer_mb=64, max_clip_mb=32, max_pending=8,
                 encoder=encode_mp4):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.on_ready, self.on_update, self.encoder = on_ready, on_update, encoder
        self.fps, self.buffer_sec = float(fps), float(buffer_sec)
        self.max_width = int(max_width)
        self.max_bytes, self.max_clip_bytes = int(max_buffer_mb * 1024**2), int(max_clip_mb * 1024**2)
        if not (0 < self.fps <= 60 and 0 < self.buffer_sec <= 300 and self.max_width >= 2
                and self.max_bytes > 0 and self.max_clip_bytes > 0 and max_pending > 0):
            raise ValueError("Некорректные ограничения буфера клипов")
        self.history, self.bytes = deque(), 0
        self.pending = {}
        self._lock = threading.Lock()
        self._last = None
        self._closed = False
        # Slots cover collecting + queued + currently encoding jobs, bounding all retained frames.
        self._slots = threading.BoundedSemaphore(max_pending)
        self._jobs = queue.Queue(maxsize=max_pending)
        self._worker = threading.Thread(target=self._work, daemon=True, name="clip-encoder")
        self._worker.start()

    def add_frame(self, image, ts):
        with self._lock:
            if self._closed or (self._last is not None and ts < self._last + 1 / self.fps - 1e-6):
                return
            h, w = image.shape[:2]
            if w > self.max_width:
                image = cv2.resize(image, (self.max_width, max(2, round(h * self.max_width / w))))
            ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                return
            sample = Sample(float(ts), jpg.tobytes())
            self._last = ts
            self.history.append(sample)
            self.bytes += len(sample.jpeg)
            while self.history and (self.history[0].ts < ts - self.buffer_sec or self.bytes > self.max_bytes):
                self.bytes -= len(self.history.popleft().jpeg)
            for key, job in list(self.pending.items()):
                if job.start <= ts <= job.end + 1e-6:
                    self._append(job, sample)
                if ts >= job.end - 1e-6:
                    del self.pending[key]
                    self._jobs.put_nowait(job)

    def _append(self, job, sample):
        if job.size + len(sample.jpeg) > self.max_clip_bytes:
            job.truncated = True
            return
        job.samples.append(sample)
        job.size += len(sample.jpeg)

    def request(self, key, anchor, before, after):
        if not (0 <= before <= 120 and 0 <= after <= 120):
            raise ValueError("Окно клипа должно быть в пределах 120 секунд до/после")
        with self._lock:
            if self._closed or not self._slots.acquire(blocking=False):
                self.on_update(key, {"clip": {"state": "failed", "error": "Очередь записи клипов заполнена или закрыта"}})
                return False
            job = Job(key, anchor - before, anchor + after, anchor)
            for sample in self.history:
                if job.start - 1e-6 <= sample.ts <= job.end + 1e-6:
                    self._append(job, sample)
            if self._last is not None and self._last >= job.end - 1e-6:
                self._jobs.put_nowait(job)
            else:
                self.pending[key] = job
            return True

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for job in self.pending.values():
                job.truncated = True
                self._jobs.put_nowait(job)
            self.pending.clear()
            self.history.clear()
            self.bytes = 0
        self._jobs.put(None)
        self._worker.join()  # uploads must be enqueued before JournalSink closes

    def _work(self):
        while True:
            job = self._jobs.get()
            if job is None:
                return
            path = self.directory / f"clip-{uuid.uuid4().hex}.mp4"
            temp = path.with_suffix(".part.mp4")
            try:
                if not job.samples:
                    raise ValueError("Для этого события в буфере не осталось кадров")
                self.on_update(job.key, {"clip": {"state": "encoding"}})
                self.encoder(temp, job.samples, self.fps)
                temp.replace(path)
                first, last = job.samples[0].ts, job.samples[-1].ts
                tol = 1.5 / self.fps
                meta = {"state": "ready", "fps": self.fps, "frames": len(job.samples),
                        "duration_sec": round(last - first + 1 / self.fps, 3),
                        "before_sec": round(max(0, job.anchor - first), 3),
                        "after_sec": round(max(0, last - job.anchor), 3),
                        "partial": job.truncated or first > job.start + tol or last < job.end - tol}
                self.on_ready(job.key, path, meta)
            except Exception as exc:
                temp.unlink(missing_ok=True)
                log.exception("Не удалось сохранить клип %s", job.key)
                self.on_update(job.key, {"clip": {"state": "failed", "error": str(exc)}})
            finally:
                self._slots.release()
