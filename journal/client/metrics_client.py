"""Сборщик метрик для конвейера на узле LinQ. Раз в window_sec отправляет в журнал окно метрик:
сколько кадров обработано по каждому потоку, сколько времени TPU был занят каждой моделью, сколько кадров
выброшено читателем «последний кадр».

    from metrics_client import MetricsReporter
    rep = MetricsReporter("http://journal:8000", node_id="linq-node-1", node_name="Узел LinQ №1", tpu_cores=1)

    # в цикле обработки
    with rep.timer("yolo-pose"):
        out = linq_infer(pose_program, tensor)
    rep.frame("cam-01")

    # если LinQ API умеет отдавать собственную загрузку ускорителя:
    rep.set_native_util(0.73)

Если журнал недоступен (например, из ВМ Jupyter нет маршрута до ноутбука), окна копятся в памяти
(до max_backlog) и уходят, когда связь появится; ошибки не роняют конвейер.
"""
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone

import requests


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


class MetricsReporter:
    def __init__(self, journal_url: str, node_id: str, node_name: str | None = None, host: str | None = None,
                 tpu_cores: int = 1, window_sec: float = 5.0, tpu_models: set[str] | None = None,
                 source: str = "pipeline", autostart: bool = True, max_backlog: int = 720):
        self.url = journal_url.rstrip("/") + "/api/metrics"
        self.node = {"node_id": node_id, "node_name": node_name, "host": host, "tpu_cores": tpu_cores}
        self.window_sec = window_sec
        self.tpu_models = tpu_models          # None — все замеренные модели считаются работающими на TPU
        self.source = source
        self.max_backlog = max_backlog
        self.backlog: list[dict] = []
        self.last_error: str | None = None
        self._lock = threading.Lock()
        self._reset()
        self._stop = threading.Event()
        if autostart:
            threading.Thread(target=self._loop, daemon=True, name="metrics-reporter").start()

    # ---------- запись событий конвейера ----------

    def _reset(self) -> None:
        self._t0 = time.monotonic()
        self._frames: dict[str, int] = defaultdict(int)
        self._dropped = 0
        self._lat: dict[str, list[float]] = defaultdict(list)
        self._native_util: float | None = None

    def frame(self, stream_id: str, n: int = 1) -> None:
        with self._lock:
            self._frames[stream_id] += n

    def dropped(self, n: int = 1) -> None:
        with self._lock:
            self._dropped += n

    def infer(self, model: str, ms: float) -> None:
        with self._lock:
            self._lat[model].append(ms)

    @contextmanager
    def timer(self, model: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.infer(model, (time.perf_counter() - t0) * 1000)

    def set_native_util(self, util: float) -> None:
        with self._lock:
            self._native_util = max(0.0, min(1.0, util))

    # ---------- отправка ----------

    def snapshot(self) -> dict:
        """Забрать окно метрик и начать новое."""
        with self._lock:
            window = max(1e-3, time.monotonic() - self._t0)
            frames, dropped, lat, util = dict(self._frames), self._dropped, dict(self._lat), self._native_util
            self._reset()
        models = {m: {"calls": len(v), "mean_ms": sum(v) / len(v), "p50_ms": _pct(v, 0.5), "p95_ms": _pct(v, 0.95)}
                  for m, v in lat.items() if v}
        busy = sum(sum(v) for m, v in lat.items() if self.tpu_models is None or m in self.tpu_models)
        payload = {
            **self.node, "ts": datetime.now(timezone.utc).isoformat(), "window_sec": window, "source": self.source,
            "streams": len(frames), "frames": sum(frames.values()), "dropped": dropped,
            "per_stream": {s: n / window for s, n in frames.items()}, "tpu_busy_ms": busy, "models": models,
        }
        if util is not None:
            payload["tpu_util"] = util
        try:
            import psutil  # необязательно
            payload["cpu_util"] = psutil.cpu_percent(interval=None) / 100
        except Exception:
            pass
        return payload

    def flush(self) -> None:
        self.backlog.append(self.snapshot())
        self.backlog = self.backlog[-self.max_backlog:]
        while self.backlog:
            try:
                r = requests.post(self.url, json=self.backlog[0], timeout=3)
                r.raise_for_status()
                self.backlog.pop(0)
                self.last_error = None
            except Exception as e:  # журнал недоступен — попробуем в следующем окне
                self.last_error = str(e)
                break

    def _loop(self) -> None:
        while not self._stop.wait(self.window_sec):
            self.flush()

    def stop(self) -> None:
        self._stop.set()
        self.flush()
