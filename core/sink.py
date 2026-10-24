"""Отправка событий в журнал без задержки конвейера: очередь + фоновый поток с повторами.

Использует journal/client/journal_client.py: create_event (идемпотентно по event_uid — повтор после сбоя сети
не создаст дубль) → upload_snapshot (кадр с рамкой и подписью) → end_event, когда трек ушёл.
Второй этап (например, «лежит без движения 10 с» после падения) дописывается в extra через PATCH
/api/events/{id}, если журнал его поддерживает; иначе — только в локальный events.jsonl.
Без адреса журнала (сухой прогон) события пишутся только в лог и в папку доказательств.
"""
import json
import logging
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "journal" / "client"))
_dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True   # не оставлять __pycache__ в journal/
from journal_client import JournalClient, _iso  # noqa: E402
from metrics_client import MetricsReporter  # noqa: E402,F401  (реэкспорт для run.py)
sys.dont_write_bytecode = _dwb

__all__ = ["JournalSink", "MetricsReporter"]
log = logging.getLogger("core.sink")


class JournalSink:
    def __init__(self, url: str | None, evidence_dir: Path, retries: int = 5):
        self.url = url.rstrip("/") if url else None
        self.client = JournalClient(self.url) if self.url else None
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.retries = retries
        self.ids: dict[str, int] = {}          # локальный ключ события -> id в журнале
        self.patch_supported = True
        self.sent = self.failed = 0
        self._q: queue.Queue = queue.Queue()
        self._jsonl = open(self.evidence_dir / "events.jsonl", "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._loop, daemon=True, name="journal-sink")
        self._worker.start()

    # ---------- вызывается из конвейера (не блокирует) ----------

    def register_camera(self, cam_id: str, name: str, area: str, source: str) -> None:
        self._q.put(("camera", cam_id, {"name": name, "area": area, "source": source}))

    def create(self, key: str, payload: dict, snapshot: Path | None) -> None:
        payload.setdefault("event_uid", uuid.uuid4().hex)
        self._log_local("create", key, payload)
        self._q.put(("create", key, {"payload": payload, "snapshot": snapshot}))

    def update(self, key: str, extra: dict) -> None:
        self._log_local("update", key, {"extra": extra})
        self._q.put(("update", key, {"extra": extra}))

    def end(self, key: str, ended_at: float) -> None:
        self._log_local("end", key, {"ended_at": _iso(ended_at)})
        self._q.put(("end", key, {"ended_at": ended_at}))

    def close(self, timeout: float = 20.0) -> None:
        self._q.put(None)
        self._worker.join(timeout)
        self._jsonl.close()

    # ---------- фоновый поток ----------

    def _log_local(self, kind: str, key: str, data: dict) -> None:
        rec = {"kind": kind, "key": key, "at": _iso(time.time()), **data}
        with self._lock:
            self._jsonl.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            self._jsonl.flush()

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            if self.client is None:
                continue
            kind, key, data = item
            try:
                self._with_retries(lambda: self._handle(kind, key, data))
            except Exception as e:
                self.failed += 1
                log.error("журнал: %s %s не отправлено: %s", kind, key, e)

    def _with_retries(self, fn):
        delay = 0.5
        for attempt in range(self.retries):
            try:
                return fn()
            except requests.HTTPError as e:      # 4xx — ошибка данных, повтор не поможет
                if e.response is not None and e.response.status_code < 500:
                    raise RuntimeError(f"{e} {e.response.text[:300]}") from None
                err = e
            except requests.RequestException as e:
                err = e
            if attempt < self.retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 8)
        raise err

    def _handle(self, kind: str, key: str, data: dict) -> None:
        c = self.client
        if kind == "camera":
            r = c.s.put(f"{self.url}/api/cameras/{key}", json=data, timeout=c.timeout)
            r.raise_for_status()
            return
        if kind == "create":
            if key not in self.ids:
                ev = c.create_event(**data["payload"])
                self.ids[key] = ev["id"]
                self.sent += 1
                log.info("журнал: событие #%s создано (%s)", ev["id"], key)
            if data.get("snapshot"):
                c.upload_snapshot(self.ids[key], Path(data["snapshot"]).as_posix())
                data["snapshot"] = None          # при повторе не грузим второй раз
            return
        event_id = self.ids.get(key)
        if event_id is None:
            log.warning("журнал: %s для %s пропущено — событие не было создано", kind, key)
            return
        if kind == "update":
            if not self.patch_supported:
                return
            r = c.s.patch(f"{self.url}/api/events/{event_id}", json={"extra": data["extra"]}, timeout=c.timeout)
            if r.status_code in (404, 405):
                self.patch_supported = False
                log.warning("журнал не поддерживает PATCH /api/events/{id}: второй этап события (%s) записан "
                            "только в events.jsonl", data["extra"].get("stage"))
                return
            r.raise_for_status()
            log.info("журнал: событие #%s дополнено (%s)", event_id, data["extra"].get("stage"))
            return
        if kind == "end":
            c.end_event(event_id, ended_at=data["ended_at"])
            return
