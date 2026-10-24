"""Клиент журнала для движка: отправить подтверждённое событие, приложить кадр, закрыть событие.

Пример:
    from journal_client import JournalClient
    j = JournalClient("http://localhost:8000")
    ev = j.create_event(pattern_id="fall", pattern_name="Падение", categories=["behavior"],
                        severity="critical", camera_id="cam-03", area="Механический цех",
                        track_id=17, started_at=t0, confirmed_at=t1, frame_ts=t1, confidence=0.91)
    j.upload_snapshot(ev["id"], "frame.jpg")
    ...
    j.end_event(ev["id"], ended_at=t2)
"""
import uuid
from datetime import datetime, timezone

import requests


def _iso(dt: datetime | float | None) -> str | None:
    if dt is None:
        return None
    if isinstance(dt, (int, float)):  # time.time() из конвейера
        dt = datetime.fromtimestamp(dt, tz=timezone.utc)
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).isoformat()


class JournalClient:
    def __init__(self, base_url: str, timeout: float = 3.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()

    def create_event(self, *, started_at, confirmed_at, frame_ts=None, event_uid: str | None = None, **fields) -> dict:
        """event_uid — ключ идемпотентности: при повторной отправке (сеть моргнула) дубля не будет."""
        payload = {
            "event_uid": event_uid or uuid.uuid4().hex,
            "started_at": _iso(started_at),
            "confirmed_at": _iso(confirmed_at),
            "frame_ts": _iso(frame_ts),
            **fields,
        }
        r = self.s.post(f"{self.base}/api/events", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def upload_snapshot(self, event_id: int, path: str) -> dict:
        return self._upload(event_id, path, "snapshot")

    def upload_clip(self, event_id: int, path: str) -> dict:
        return self._upload(event_id, path, "clip")

    def _upload(self, event_id: int, path: str, kind: str) -> dict:
        with open(path, "rb") as fh:
            r = self.s.post(f"{self.base}/api/events/{event_id}/media", params={"kind": kind},
                            files={"file": (path.rsplit("/", 1)[-1], fh)}, timeout=30)
        r.raise_for_status()
        return r.json()

    def update_event(self, event_id: int, extra: dict) -> dict:
        r = self.s.patch(f"{self.base}/api/events/{event_id}", json={"extra": extra}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def end_event(self, event_id: int, ended_at) -> dict:
        r = self.s.post(f"{self.base}/api/events/{event_id}/end", json={"ended_at": _iso(ended_at)},
                        timeout=self.timeout)
        r.raise_for_status()
        return r.json()
