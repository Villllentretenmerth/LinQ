"""Метрики узлов LinQ: FPS, загрузка TPU, задержки моделей и расчёт, сколько ещё камер выдержит узел.

Откуда приходят метрики:
  * конвейер — client/metrics_client.py, окно каждые несколько секунд (source="pipeline");
  * замер в Jupyter — bench/linq_bench.py, разовый результат (source="bench");
  * демо-режим — правдоподобные значения для показа интерфейса (source="demo", помечаются в UI).

Загрузка TPU считается по нашим замерам: время, когда TPU был занят инференсом, делённое на длину окна
(busy time / wall time). Если API LinQ отдаёт собственную загрузку — передайте её в tpu_util, она важнее.
"""
import asyncio
import math
import random
from datetime import datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import SessionLocal, as_utc, get_session, utcnow
from .models import Node, NodeMetric

router = APIRouter(prefix="/api/metrics", tags=["metrics"])
ONLINE_SEC = 30  # узел «в сети», если метрики приходили не позже этого


class ModelStat(BaseModel):
    calls: int = Field(ge=0)
    mean_ms: float = Field(ge=0)
    p50_ms: float | None = None
    p95_ms: float | None = None


class NodeMetricIn(BaseModel):
    node_id: str
    node_name: str | None = None
    host: str | None = None
    tpu_cores: int = Field(1, ge=1)
    ts: datetime | None = None                     # конец окна; по умолчанию — время приёма
    window_sec: float = Field(gt=0)
    source: Literal["pipeline", "bench", "demo"] = "pipeline"
    streams: int = Field(0, ge=0)
    frames: int = Field(0, ge=0)
    dropped: int = Field(0, ge=0)
    per_stream: dict[str, float] = {}              # FPS по каждому потоку
    fps_stream_min: float | None = None
    tpu_busy_ms: float = Field(0, ge=0)
    tpu_util: float | None = Field(None, ge=0, le=1)   # если LinQ отдаёт свою загрузку
    cpu_util: float | None = Field(None, ge=0, le=1)
    models: dict[str, ModelStat] = {}

    @field_validator("ts")
    @classmethod
    def _utc(cls, v):
        return as_utc(v)


def _sample_dict(m: NodeMetric) -> dict:
    return {
        "ts": as_utc(m.ts).isoformat(), "window_sec": m.window_sec, "source": m.source, "streams": m.streams,
        "frames": m.frames, "dropped": m.dropped, "fps_total": round(m.fps_total, 2),
        "fps_stream_min": None if m.fps_stream_min is None else round(m.fps_stream_min, 2),
        "tpu_busy_ms": round(m.tpu_busy_ms, 1), "tpu_util": round(m.tpu_util, 4),
        "cpu_util": None if m.cpu_util is None else round(m.cpu_util, 4),
        "models": m.models, "per_stream": m.per_stream,
    }


def capacity(m: NodeMetric | None, tpu_cores: int, target_fps: float, util_cap: float) -> dict | None:
    """Сколько потоков выдержит узел при целевом FPS на поток и запасе по загрузке TPU.
    Цена кадра = время TPU на кадр со всем каскадом моделей (busy / кадры) — берётся из замеров."""
    if m is None or m.frames <= 0 or m.tpu_busy_ms <= 0:
        return None
    ms_per_frame = m.tpu_busy_ms / m.frames
    max_streams = math.floor(util_cap * 1000 * tpu_cores / (ms_per_frame * target_fps))
    return {
        "tpu_ms_per_frame": round(ms_per_frame, 2), "target_fps": target_fps, "util_cap": util_cap,
        "max_streams": max_streams, "current_streams": m.streams, "headroom": max_streams - m.streams,
    }


def store_metric(session: Session, body: NodeMetricIn) -> NodeMetric:
    now = utcnow()
    node = session.get(Node, body.node_id)
    if node is None:
        node = Node(id=body.node_id, name=body.node_name or body.node_id, host=body.host, tpu_cores=body.tpu_cores)
        session.add(node)
    if body.node_name:
        node.name = body.node_name
    if body.host:
        node.host = body.host
    node.tpu_cores = body.tpu_cores
    node.last_seen, node.last_source = now, body.source

    fps_min = body.fps_stream_min
    if fps_min is None and body.per_stream:
        fps_min = min(body.per_stream.values())
    util = body.tpu_util
    if util is None:
        util = min(1.0, body.tpu_busy_ms / (body.window_sec * 1000 * body.tpu_cores))
    m = NodeMetric(
        node_id=body.node_id, ts=body.ts or now, window_sec=body.window_sec, source=body.source,
        streams=body.streams or len(body.per_stream), frames=body.frames, dropped=body.dropped,
        fps_total=body.frames / body.window_sec, fps_stream_min=fps_min, tpu_busy_ms=body.tpu_busy_ms,
        tpu_util=util, cpu_util=body.cpu_util,
        models={k: v.model_dump() for k, v in body.models.items()}, per_stream=body.per_stream,
    )
    session.add(m)
    session.commit()
    return m


@router.post("", status_code=201)
def push_metrics(body: NodeMetricIn, session: Session = Depends(get_session)):
    m = store_metric(session, body)
    return {"ok": True, "node_id": m.node_id, "fps_total": round(m.fps_total, 2), "tpu_util": round(m.tpu_util, 4)}


@router.get("/nodes")
def list_nodes(target_fps: float = Query(5, gt=0), util_cap: float = Query(0.8, gt=0, le=1),
               session: Session = Depends(get_session)):
    now = utcnow()
    out = []
    for node in session.scalars(select(Node).order_by(Node.id)):
        latest = session.scalars(
            select(NodeMetric).where(NodeMetric.node_id == node.id).order_by(NodeMetric.ts.desc()).limit(1)
        ).first()
        last_seen = as_utc(node.last_seen)
        out.append({
            "id": node.id, "name": node.name, "host": node.host, "tpu_cores": node.tpu_cores,
            "last_seen": last_seen.isoformat() if last_seen else None,
            "online": bool(last_seen and (now - last_seen).total_seconds() <= ONLINE_SEC),
            "source": node.last_source,
            "latest": _sample_dict(latest) if latest else None,
            "capacity": capacity(latest, node.tpu_cores, target_fps, util_cap),
        })
    return out


@router.get("/series")
def series(node_id: list[str] | None = Query(None), minutes: int = Query(60, ge=1, le=24 * 60),
           session: Session = Depends(get_session)):
    since = utcnow() - timedelta(minutes=minutes)
    stmt = select(NodeMetric).where(NodeMetric.ts >= since).order_by(NodeMetric.ts)
    if node_id:
        stmt = stmt.where(NodeMetric.node_id.in_(node_id))
    res: dict[str, list] = {}
    for m in session.scalars(stmt):
        res.setdefault(m.node_id, []).append({
            "t": as_utc(m.ts).isoformat(), "tpu_util": round(m.tpu_util, 4), "fps_total": round(m.fps_total, 2),
            "fps_stream_min": None if m.fps_stream_min is None else round(m.fps_stream_min, 2),
            "cpu_util": m.cpu_util, "streams": m.streams, "source": m.source,
        })
    return res


# ---------- демо: правдоподобные метрики для показа интерфейса ----------

DEMO_NODES = [
    # id, имя, потоков, целевой FPS на поток, загрузка CPU
    ("linq-node-1", "Узел LinQ №1 — склад отгрузки", 7, 10, 0.46),
    ("linq-node-2", "Узел LinQ №2 — цех и периметр", 4, 10, 0.31),
]
DEMO_MODELS = {  # среднее время и сколько вызовов на кадр
    "yolo-pose": (8.6, 1.0), "yolo-det": (4.1, 0.34), "behavior_v1": (0.7, 0.9), "crop_attr_v1": (2.4, 0.12),
}


def demo_sample(node: tuple, window_sec: float, ts: datetime, rnd: random.Random, load: float = 1.0) -> NodeMetricIn:
    node_id, name, streams, fps, cpu = node
    per_stream = {f"{node_id}-cam{i + 1}": round(max(1.0, rnd.gauss(fps * min(1.0, 1.05 / max(load, 0.6)), 0.35)), 2)
                  for i in range(streams)}
    frames = int(sum(per_stream.values()) * window_sec)
    models, busy = {}, 0.0
    for mname, (mean, per_frame) in DEMO_MODELS.items():
        calls = max(1, int(frames * per_frame))
        m = mean * rnd.uniform(0.94, 1.06) * load
        busy += calls * m
        models[mname] = {"calls": calls, "mean_ms": round(m, 2), "p50_ms": round(m * 0.97, 2), "p95_ms": round(m * 1.38, 2)}
    return NodeMetricIn(
        node_id=node_id, node_name=name, host="demo", tpu_cores=1, ts=ts, window_sec=window_sec, source="demo",
        streams=streams, frames=frames, dropped=rnd.randint(0, 3), per_stream=per_stream,
        tpu_busy_ms=busy, cpu_util=min(0.98, max(0.05, rnd.gauss(cpu, 0.04))), models=models,
    )


def seed_demo_metrics(session: Session, minutes: int = 120, seed: int = 94) -> int:
    rnd = random.Random(seed)
    now = utcnow().replace(second=0, microsecond=0)
    n = 0
    for node in DEMO_NODES:
        for k in range(minutes, 0, -1):
            ts = now - timedelta(minutes=k)
            # утренний пик: в начале смены нагрузка выше
            load = 1.0 + (0.12 if 50 <= k <= 70 else 0) + rnd.uniform(-0.03, 0.03)
            store_metric(session, demo_sample(node, 60.0, ts, rnd, load))
            n += 1
    return n


async def demo_ticker(interval: float = 5.0):
    """Пока включён DEMO_MODE и есть демо-узлы, раз в несколько секунд присылает свежее окно метрик."""
    rnd = random.Random()
    while True:
        await asyncio.sleep(interval)
        try:
            with SessionLocal() as s:
                demo_ids = {nid for nid, in s.execute(select(Node.id).where(Node.last_source == "demo"))}
                for node in DEMO_NODES:
                    if node[0] in demo_ids:
                        store_metric(s, demo_sample(node, interval, utcnow(), rnd, 1.0 + rnd.uniform(-0.04, 0.06)))
        except Exception:  # демо не должно ронять сервис
            pass
