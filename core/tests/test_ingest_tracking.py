"""Источники кадров, сшивка треков при падении и декодер головы YOLOv8 (если модели выгружены)."""
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.infer import PoseResult
from core.ingest import Frame, fixed_rate, list_images, offline_frames
from core.tracking import PersonTracker

from .helpers import skeleton

MODELS = Path(__file__).resolve().parents[1] / "models"


def test_image_folder_sorted_by_name(tmp_path):
    for name in ["frame10.jpg", "frame2.jpg", "frame1.png", "notes.txt"]:
        p = tmp_path / name
        if p.suffix == ".txt":
            p.write_text("x")
        else:
            cv2.imwrite(str(p), np.full((8, 8, 3), 10, np.uint8))
    assert [p.name for p in list_images(str(tmp_path))] == ["frame1.png", "frame2.jpg", "frame10.jpg"]
    frames = list(offline_frames(str(tmp_path), fps=5))
    assert [f.seq for f in frames] == [1, 2, 3]
    assert frames[1].ts - frames[0].ts == pytest.approx(0.2)    # время кадра — по номеру и FPS


class FakeReader:
    """Источник, который присылает новый кадр раз в 0,1 с, а читают его чаще."""

    def __init__(self, n: int):
        self.seq, self.eof, self.n = 0, False, n
        self._lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        for _ in range(self.n):
            time.sleep(0.1)
            with self._lock:
                self.seq += 1
        self.eof = True

    def read(self):
        with self._lock:
            return Frame(np.zeros((2, 2, 3), np.uint8), time.time(), time.time(), self.seq) if self.seq else None

    def close(self):
        pass


def test_fixed_rate_yields_only_new_frames():
    seqs = [f.seq for f in fixed_rate(FakeReader(5), fps=50)]   # читаем в 5 раз чаще, чем приходят кадры
    assert seqs == sorted(set(seqs)) and seqs[-1] == 5          # ни одного повтора, конец по eof


def test_tracker_stitches_history_when_fall_changes_id():
    tr = PersonTracker(fps=10, lost_sec=2.0)
    ts = 0.0
    for i in range(20):                                          # стоит 2 с
        box, k = skeleton(800, 650, 200, 0)
        persons, _ = tr.update(PoseResult(box[None], np.array([0.9], np.float32), k[None]), ts)
        ts += 0.1
    first_id = persons[0].track_id
    box, k = skeleton(800, 650, 200, 90)                         # следующий кадр: уже лежит (IoU мал)
    for _ in range(3):
        persons, _ = tr.update(PoseResult(box[None], np.array([0.9], np.float32), k[None]), ts)
        ts += 0.1
    p = persons[0]
    if p.track_id != first_id:                                   # ByteTrack сменил ID — история должна перейти
        assert tr.stitched[p.track_id] == first_id
    assert p.hist.first_ts == 0.0
    assert p.hist.features()["center_drop_heights"] > 0.4       # правило падения видит, что человек стоял


@pytest.mark.skipif(not (MODELS / "yolov8n-pose_head.onnx").exists(), reason="модели не выгружены")
def test_head_decoder_matches_full_model():
    from core.infer import OnnxBackend, OnnxHeadBackend

    x = np.random.default_rng(1).random((1, 3, 640, 640), dtype=np.float32)
    for name, kpt in [("yolov8n-pose", True), ("yolov8n", False)]:
        full = OnnxBackend(MODELS / f"{name}.onnx").run(x)[0]
        head = OnnxHeadBackend(MODELS / f"{name}_head.onnx", kpt)
        assert np.abs(full - head.finish(head.run(x))[0]).max() < 1e-2
