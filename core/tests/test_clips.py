"""Event videos: independent capture, delayed inference, EOF, bounded memory and H.264."""
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.clips import ClipRecorder, clip_window
from core.ingest import LatestFrameReader


def frame(i=0):
    return np.full((48, 64, 3), i, np.uint8)


def recorder(tmp_path, **opts):
    ready, updates = [], []
    r = ClipRecorder(tmp_path, lambda *x: ready.append(x), lambda *x: updates.append(x), **opts)
    return r, ready, updates


def fake_encoder(path, samples, fps):
    path.write_bytes(b'test')


def test_delayed_detection_keeps_frames_skipped_by_inference(tmp_path):
    captured = []
    def encode(path, samples, fps):
        captured.append([s.ts for s in samples])
        path.write_bytes(b'test')
    r, ready, _ = recorder(tmp_path, fps=10, encoder=encode)
    # Exercise the same publication path as RTSP: only the last frame is available to inference.
    reader = LatestFrameReader.__new__(LatestFrameReader)
    reader._lock, reader._stop = threading.Lock(), threading.Event()
    reader.on_frame, reader.seq = r.add_frame, 0
    reader._last_read_seq, reader.dropped = 0, 0
    for i in range(61):
        reader._publish(frame(i), i / 10)
    latest = reader.read()
    assert latest.ts == 6 and latest.skipped == 60
    # Inference has just confirmed the event on source frame 3, while capture reached 6.
    r.request('fall', anchor=3, before=2, after=2)
    r.close()
    assert captured[0] == pytest.approx([i / 10 for i in range(10, 51)])
    assert ready[0][2]['partial'] is False


def test_overlapping_events_and_eof_keep_available_tail(tmp_path):
    windows = []
    def encode(path, samples, fps):
        windows.append([s.ts for s in samples]); path.write_bytes(b'x')
    r, ready, _ = recorder(tmp_path, fps=2, encoder=encode)
    for i in range(5):
        r.add_frame(frame(i), i / 2)
    r.request('a', 2, 1, 1)
    r.request('b', 2, 1, 4)
    for i in range(5, 8):
        r.add_frame(frame(i), i / 2)
    r.close()
    by_key = {key: meta for key, _, meta in ready}
    assert by_key['a']['partial'] is False
    assert by_key['b']['partial'] is True
    assert windows[0] == [1, 1.5, 2, 2.5, 3]
    assert windows[1][-1] == 3.5


def test_capacity_memory_limits_and_encoder_failure(tmp_path):
    r, ready, updates = recorder(tmp_path, fps=2, max_pending=1, max_buffer_mb=.002,
                                 encoder=fake_encoder)
    for i in range(20):
        r.add_frame(frame(i), i / 2)
    assert r.bytes <= .002 * 1024**2
    assert r.request('a', 9.5, 5, 10)
    assert not r.request('b', 9.5, 1, 2)
    r.close()
    assert ready[0][2]['partial']
    assert any(k == 'b' and e['clip']['state'] == 'failed' for k, e in updates)
    def broken(*args):
        raise RuntimeError('encoder unavailable')
    r, ready, updates = recorder(tmp_path, encoder=broken)
    r.add_frame(frame(), 0)
    r.request('bad', 0, 0, 0)
    r.close()
    assert not ready
    assert updates[-1][1]['clip']['state'] == 'failed'


def test_h264_clip_duration_and_timestamp_gaps(tmp_path):
    r, ready, updates = recorder(tmp_path, fps=10)
    for ts in [0, .1, .2, 1.0]:  # gap must not make the clip play four times faster
        r.add_frame(frame(round(ts * 200)), ts)
    r.request('video', 1, 1, 0)
    r.close()
    assert ready, updates
    import imageio_ffmpeg
    video = imageio_ffmpeg.read_frames(str(ready[0][1]))
    meta = next(video)
    assert meta['fps'] == pytest.approx(10)
    assert meta['size'] == (64, 48)
    assert len(list(video)) == 11
    data = ready[0][1].read_bytes()
    assert b'avc1' in data  # H.264, unlike OpenCV's browser-incompatible mp4v fallback
    assert data.index(b'moov') < data.index(b'mdat')


def test_clip_window_validation():
    assert clip_window({'clip_sec': [-10, 5]}) == (10, 5)
    for value in ([1, 2], [-1, -2], [float('nan'), 1], [-10000, 1], 'bad'):
        assert clip_window({'clip_sec': value}) is None
