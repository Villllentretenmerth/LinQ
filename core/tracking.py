"""Трекинг людей и техники: ByteTrack из supervision (фильтр Калмана внутри) + история каждого трека.

frame_rate трекера = шаг обработки (FPS конвейера), а не FPS камеры: фильтр Калмана прогнозирует на один
обработанный кадр вперёд. Потерянный трек живёт lost_sec секунд — потом его события закрываются.
"""
import inspect
from collections import deque

import numpy as np
import supervision as sv

from .engine import Person, Vehicle
from .features import TrackHistory
from .infer import DetResult, PoseResult


def make_bytetrack(fps: float, lost_sec: float = 2.0, activation: float = 0.25, matching: float = 0.8):
    frame_rate = max(1, int(round(fps)))
    # в ByteTrack max_time_lost = frame_rate / 30 * lost_track_buffer, т. е. буфер задан «в кадрах при 30 FPS»
    buffer = max(1, int(round(lost_sec * 30)))
    if hasattr(sv, "ByteTrack"):
        return sv.ByteTrack(track_activation_threshold=activation, lost_track_buffer=buffer,
                            minimum_matching_threshold=matching, frame_rate=frame_rate)
    from trackers import ByteTrackTracker   # supervision >= 0.27: ByteTrack переехал в пакет trackers

    params = inspect.signature(ByteTrackTracker).parameters
    kw = {"track_activation_threshold": activation, "lost_track_buffer": buffer,
          "minimum_matching_threshold": matching, "frame_rate": frame_rate}
    return ByteTrackTracker(**{k: v for k, v in kw.items() if k in params})


def _update(tracker, det: sv.Detections) -> sv.Detections:
    if hasattr(tracker, "update_with_detections"):
        return tracker.update_with_detections(det)
    out = tracker.update(det)
    return out[out.tracker_id != -1]


class PersonTracker:
    """Треки людей с историей. При падении рамка за полсекунды из вертикальной становится горизонтальной,
    IoU с прогнозом Калмана мал, и ByteTrack заводит новый ID. Чтобы правило падения видело, что человек
    только что стоял, новый трек наследует историю трека, пропавшего рядом не дольше stitch_sec назад."""

    def __init__(self, fps: float, lost_sec: float = 2.0, keep_sec: float = 15.0, stitch_sec: float = 1.5,
                 stitch_dist_h: float = 1.0, floor=None):
        self.floor = floor
        self.bt = make_bytetrack(fps, lost_sec)
        self.lost_sec, self.keep_sec = lost_sec, keep_sec
        self.stitch_sec, self.stitch_dist_h = stitch_sec, stitch_dist_h
        self.tracks: dict[int, TrackHistory] = {}
        self.stitched: dict[int, int] = {}       # новый ID -> ID, чью историю он продолжил
        self.donated: set[int] = set()           # треки, уже отдавшие историю

    def _donor(self, box: np.ndarray, ts: float, taken: set[int]) -> TrackHistory | None:
        bottom = np.array([(box[0] + box[2]) / 2, box[3]])
        best, best_d = None, None
        for tid, h in self.tracks.items():
            if tid in taken or tid in self.donated or h.last_ts >= ts or ts - h.last_ts > self.stitch_sec:
                continue
            x1, _, x2, y2 = h.last.xyxy
            d = float(np.linalg.norm(bottom - np.array([(x1 + x2) / 2, y2]))) / h.ref_height()
            if d <= self.stitch_dist_h and (best_d is None or d < best_d):
                best, best_d = h, d
        return best

    def update(self, pose: PoseResult, ts: float) -> tuple[list[Person], list[int]]:
        """Люди на кадре с историей треков + ID треков, которые потеряны дольше lost_sec."""
        det = sv.Detections(xyxy=pose.xyxy.astype(np.float32), confidence=pose.conf.astype(np.float32),
                            class_id=np.zeros(len(pose.conf), dtype=int),
                            data={"idx": np.arange(len(pose.conf))})
        tracked = _update(self.bt, det)
        present = {int(t) for t in tracked.tracker_id}
        taken = set(present)
        persons = []
        for conf, tid, idx in zip(tracked.confidence, tracked.tracker_id, tracked.data.get("idx", [])):
            tid = int(tid)
            h = self.tracks.get(tid)
            if h is None:
                h = self.tracks[tid] = TrackHistory(tid, self.keep_sec, self.floor)
                donor = self._donor(pose.xyxy[idx], ts, taken)
                if donor is not None:
                    h.inherit(donor)
                    taken.add(donor.id)
                    self.donated.add(donor.id)
                    self.stitched[tid] = donor.id
            h.add(ts, pose.xyxy[idx], pose.kpts[idx], float(conf))
            persons.append(Person(tid, pose.xyxy[idx], float(conf), h))
        lost = [tid for tid, h in self.tracks.items() if ts - h.last_ts > self.lost_sec]
        for tid in lost:
            del self.tracks[tid]
            self.stitched.pop(tid, None)
            self.donated.discard(tid)
        return persons, lost


class VehicleTracker:
    """Техника из детектора: ID трека и скорость центра рамки (пикселей в секунду) за последнюю секунду."""

    def __init__(self, fps: float, lost_sec: float = 2.0):
        self.bt = make_bytetrack(fps, lost_sec)
        self.lost_sec = lost_sec
        self.hist: dict[int, deque] = {}

    def update(self, det: DetResult, ts: float) -> list[Vehicle]:
        d = sv.Detections(xyxy=det.xyxy.astype(np.float32), confidence=det.conf.astype(np.float32),
                          class_id=det.cls.astype(int), data={"idx": np.arange(len(det.conf))})
        tracked = _update(self.bt, d)
        out = []
        for conf, tid, idx in zip(tracked.confidence, tracked.tracker_id, tracked.data.get("idx", [])):
            tid = int(tid)
            box = det.xyxy[idx]
            c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            q = self.hist.setdefault(tid, deque())
            q.append((ts, c))
            while q and q[0][0] < ts - 1.0:
                q.popleft()
            speed = None
            if len(q) >= 2 and ts - q[0][0] >= 0.3:
                speed = float(np.linalg.norm(c - q[0][1]) / (ts - q[0][0]))
            out.append(Vehicle(tid, box, float(conf), det.names[idx], speed))
        for tid in [t for t, q in self.hist.items() if ts - q[-1][0] > self.lost_sec]:
            del self.hist[tid]
        return out
