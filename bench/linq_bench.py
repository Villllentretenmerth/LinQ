"""Замер скорости моделей на LinQ TPU прямо в Jupyter хакатона.

Ничего не знает про API LinQ: вы передаёте функцию, которая делает один инференс
(ту же строку, что в examples/example.ipynb), а замер считает задержки и пропускную способность.

    from linq_bench import Bench
    b = Bench(node_id="linq-hackathon", node_name="ВМ хакатона LinQ", tpu_cores=1)
    b.measure("resnet50", lambda: run_resnet(x))                         # пример организаторов
    b.measure("yolo-pose", lambda: run_pose(frame), calls_per_frame=1)   # на каждый кадр
    b.measure("behavior_v1", lambda: run_beh(win), calls_per_frame=0.9)  # ~раз в 0,5 с на трек
    b.table()                       # таблица в ноутбуке
    b.save("bench_result.json")     # скачать и импортировать в журнал: python -m scripts.import_metrics bench_result.json
    b.push("http://<адрес журнала>:8000")   # если из ВМ есть маршрут до журнала

calls_per_frame — сколько раз модель вызывается на один кадр в среднем (каскад: дорогие модели —
только для кандидатов). Из этого считается «цена кадра» в мс TPU и сколько камер выдержит узел.
"""
import json
import statistics
import time
from datetime import datetime, timezone


class Bench:
    def __init__(self, node_id: str = "linq-jupyter", node_name: str | None = None, tpu_cores: int = 1):
        self.node_id, self.node_name, self.tpu_cores = node_id, node_name or node_id, tpu_cores
        self.results: dict[str, dict] = {}

    def measure(self, name: str, fn, n: int = 100, warmup: int = 10, calls_per_frame: float = 1.0,
                tpu_time=None) -> dict:
        """tpu_time(результат fn) -> мс на самом ускорителе. Облачный TPU хакатона отвечает через брокер, и полный
        круг (rtt) включает сеть и очередь; загрузку TPU и запас по камерам честнее считать по времени воркера:
        b.measure("resnet50", lambda: tpu.infer(x), tpu_time=lambda r: r[1]["tpu_ms"])"""
        for _ in range(warmup):            # прогрев: первые вызовы всегда медленнее
            fn()
        rtt, dev = [], []
        for _ in range(n):
            t0 = time.perf_counter()
            res = fn()
            rtt.append((time.perf_counter() - t0) * 1000)
            if tpu_time is not None:
                dev.append(float(tpu_time(res)))

        def stats(times: list[float]) -> dict:
            times = sorted(times)
            q = lambda p: times[min(len(times) - 1, int(round(p * (len(times) - 1))))]  # noqa: E731
            return {"mean_ms": statistics.fmean(times), "p50_ms": q(0.5), "p95_ms": q(0.95),
                    "min_ms": times[0], "max_ms": times[-1]}

        r = {"calls": n, **stats(dev or rtt), "calls_per_frame": calls_per_frame,
             "time_source": "tpu" if dev else "wall"}
        r["fps"] = 1000 / r["mean_ms"]
        if dev:   # полный круг — отдельно: он ограничивает FPS одной камеры при вызовах по одному
            r.update({f"rtt_{k}": v for k, v in stats(rtt).items()})
        self.results[name] = r
        return r

    @property
    def ms_per_frame(self) -> float:
        """Время TPU на один кадр со всем каскадом моделей."""
        return sum(r["mean_ms"] * r["calls_per_frame"] for r in self.results.values())

    def streams_capacity(self, target_fps: float = 5, util_cap: float = 0.8) -> int:
        return int(util_cap * 1000 * self.tpu_cores / (self.ms_per_frame * target_fps)) if self.ms_per_frame else 0

    def table(self) -> None:
        print(f"{'модель':<16}{'p50, мс':>10}{'p95, мс':>10}{'среднее':>10}{'FPS':>8}{'вызовов/кадр':>14}{'круг p50, мс':>14}")
        for name, r in self.results.items():
            rtt = f"{r['rtt_p50_ms']:>14.2f}" if "rtt_p50_ms" in r else f"{'—':>14}"
            print(f"{name:<16}{r['p50_ms']:>10.2f}{r['p95_ms']:>10.2f}{r['mean_ms']:>10.2f}{r['fps']:>8.1f}"
                  f"{r['calls_per_frame']:>14.2f}{rtt}")
        if any(r.get("time_source") == "tpu" for r in self.results.values()):
            print("время — на воркере TPU; «круг» — от отправки задачи до ответа (сеть + очередь + TPU)")
        if self.results:
            print(f"\nЦена кадра (каскад): {self.ms_per_frame:.2f} мс TPU → до {1000 / self.ms_per_frame:.0f} кадров/с на узел")
            for fps in (5, 10, 25):
                print(f"  при {fps} FPS на камеру и загрузке ≤80%: {self.streams_capacity(fps)} камер")

    def to_metric(self, frames: int = 100) -> dict:
        """Формат окна метрик журнала (source='bench'): узел гоняет каскад без пауз, загрузка TPU = 100%."""
        busy = frames * self.ms_per_frame
        return {
            "node_id": self.node_id, "node_name": self.node_name, "tpu_cores": self.tpu_cores,
            "ts": datetime.now(timezone.utc).isoformat(), "window_sec": max(busy / 1000, 1e-3), "source": "bench",
            "streams": 1, "frames": frames, "per_stream": {"bench": 1000 / self.ms_per_frame if self.ms_per_frame else 0},
            "tpu_busy_ms": busy,
            "models": {k: {"calls": int(round(frames * v["calls_per_frame"])) or 1, "mean_ms": v["mean_ms"],
                           "p50_ms": v["p50_ms"], "p95_ms": v["p95_ms"]} for k, v in self.results.items()},
        }

    def save(self, path: str = "bench_result.json") -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"metric": self.to_metric(), "raw": self.results}, f, ensure_ascii=False, indent=2)
        return path

    def push(self, journal_url: str) -> dict:
        import requests
        r = requests.post(journal_url.rstrip("/") + "/api/metrics", json=self.to_metric(), timeout=5)
        r.raise_for_status()
        return r.json()
