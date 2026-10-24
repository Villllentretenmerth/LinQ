"""Сервис журнала событий: приём событий от движка, журнал с фильтрами, статусы оператора,
статистика для дашбордов, живая лента и выгрузка для проверяющих."""
import asyncio
import csv
import io
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import String, cast, func, select
from sqlalchemy.orm import Session, selectinload

from . import metrics, stats
from .db import Base, SessionLocal, as_utc, engine, get_session, utcnow
from .demo import seed_startup_events
from .live import manager
from .models import Camera, Event, StatusChange
from .schemas import EventDetail, EventEnd, EventIn, EventOut, EventPage, EventUpdate, StatusIn

MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "./media"))
WEB_DIR = Path(os.getenv("WEB_DIR", Path(__file__).resolve().parent.parent / "web"))
DEFAULT_TZ = os.getenv("PLANT_TZ", "Europe/Moscow")  # часовой пояс предприятия для смен и графиков
DEMO_MODE = os.getenv("DEMO_MODE", "1") == "1"       # кнопка «тестовое событие» в интерфейсе

DEMO_SEED = os.getenv("DEMO_SEED", "1") == "1"


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(engine)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    if DEMO_MODE and DEMO_SEED:
        with SessionLocal() as session:
            seed_startup_events(session)
    ticker = asyncio.create_task(metrics.demo_ticker()) if DEMO_MODE else None  # оживляет демо-узлы
    yield
    if ticker:
        ticker.cancel()


app = FastAPI(title="team94 — журнал событий ТБ", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")
app.include_router(metrics.router)


# ---------- вспомогательное ----------

def to_out(e: Event, detail: bool = False) -> EventOut:
    cls = EventDetail if detail else EventOut
    out = cls.model_validate(e)
    out.time_to_alert_ms = stats.time_to_alert_ms(e)
    out.pipeline_latency_ms = stats.pipeline_latency_ms(e)
    return out


class EventFilter:
    """Фильтры журнала: тип нарушения, категория, камера, участок, зона, статус, время."""

    def __init__(
        self,
        pattern_id: list[str] | None = Query(None),
        category: str | None = None,
        severity: str | None = None,
        camera_id: list[str] | None = Query(None),
        area: list[str] | None = Query(None),
        zone: str | None = None,
        status: list[str] | None = Query(None),
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ):
        self.pattern_id, self.category, self.severity = pattern_id, category, severity
        self.camera_id, self.area, self.zone, self.status = camera_id, area, zone, status
        self.date_from, self.date_to = as_utc(date_from), as_utc(date_to)

    def apply(self, stmt):
        if self.pattern_id:
            stmt = stmt.where(Event.pattern_id.in_(self.pattern_id))
        if self.category:  # категории хранятся JSON-списком; ищем по тексту — работает и в PG, и в SQLite
            stmt = stmt.where(cast(Event.categories, String).like(f'%"{self.category}"%'))
        if self.severity:
            stmt = stmt.where(Event.severity == self.severity)
        if self.camera_id:
            stmt = stmt.where(Event.camera_id.in_(self.camera_id))
        if self.area:
            stmt = stmt.where(Event.area.in_(self.area))
        if self.zone:
            stmt = stmt.where(Event.zone == self.zone)
        if self.status:
            stmt = stmt.where(Event.status.in_(self.status))
        if self.date_from:
            stmt = stmt.where(Event.started_at >= self.date_from)
        if self.date_to:
            stmt = stmt.where(Event.started_at < self.date_to)
        return stmt


def _filtered(session: Session, f: EventFilter) -> list[Event]:
    return list(session.scalars(f.apply(select(Event))).all())


def _tz(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except Exception:
        raise HTTPException(400, f"Неизвестный часовой пояс: {name}")


def _get_event(session: Session, event_id: int) -> Event:
    e = session.get(Event, event_id)
    if e is None:
        raise HTTPException(404, "Событие не найдено")
    return e


# ---------- служебное ----------

@app.get("/health")
def health():
    return {"status": "ok", "db": engine.dialect.name, "plant_tz": DEFAULT_TZ, "demo_mode": DEMO_MODE}


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui/")


# ---------- камеры ----------

@app.get("/api/cameras")
def list_cameras(session: Session = Depends(get_session)):
    return [{"id": c.id, "name": c.name, "area": c.area, "source": c.source}
            for c in session.scalars(select(Camera).order_by(Camera.id))]


@app.put("/api/cameras/{camera_id}")
def upsert_camera(camera_id: str, body: dict, session: Session = Depends(get_session)):
    cam = session.get(Camera, camera_id) or Camera(id=camera_id, name=camera_id, area="—")
    cam.name = body.get("name", cam.name)
    cam.area = body.get("area", cam.area)
    cam.source = body.get("source", cam.source)
    session.add(cam)
    session.commit()
    return {"id": cam.id, "name": cam.name, "area": cam.area, "source": cam.source}


# ---------- события: запись (вызывает движок) ----------

@app.post("/api/events", response_model=EventOut, status_code=201)
async def create_event(body: EventIn, session: Session = Depends(get_session)):
    # Идемпотентность: повторная отправка того же события не создаёт дубль в журнале
    existing = session.scalar(select(Event).where(Event.event_uid == body.event_uid))
    if existing:
        return to_out(existing)

    if session.get(Camera, body.camera_id) is None:
        session.add(Camera(id=body.camera_id, name=body.camera_name or body.camera_id, area=body.area))

    data = body.model_dump(exclude={"camera_name"})
    e = Event(**data, received_at=utcnow(), status="new")
    session.add(e)
    session.commit()
    session.refresh(e)
    out = to_out(e)
    await manager.broadcast({"type": "event_created", "event": out.model_dump(mode="json")})
    return out


def merge_extra(base: dict, update: dict) -> dict:
    merged = dict(base or {})
    for key, value in update.items():
        merged[key] = merge_extra(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    return merged


@app.patch("/api/events/{event_id}", response_model=EventOut)
async def update_event(event_id: int, body: EventUpdate, session: Session = Depends(get_session)):
    e = _get_event(session, event_id)
    e.extra = merge_extra(e.extra, body.extra)
    session.commit()
    out = to_out(e)
    await manager.broadcast({"type": "event_updated", "event": out.model_dump(mode="json")})
    return out


@app.post("/api/events/{event_id}/end", response_model=EventOut)
async def end_event(event_id: int, body: EventEnd, session: Session = Depends(get_session)):
    e = _get_event(session, event_id)
    e.ended_at = body.ended_at
    session.commit()
    out = to_out(e)
    await manager.broadcast({"type": "event_ended", "event": out.model_dump(mode="json")})
    return out


@app.post("/api/events/{event_id}/media", response_model=EventOut)
async def upload_media(event_id: int, kind: str = Query("snapshot", pattern="^(snapshot|clip)$"),
                       file: UploadFile = File(...), session: Session = Depends(get_session)):
    """Кадр или клип-доказательство. Если движок пишет файлы в общий том, можно передать путь в EventIn."""
    e = _get_event(session, event_id)
    ext = Path(file.filename or "").suffix.lower() or (".jpg" if kind == "snapshot" else ".mp4")
    folder = MEDIA_DIR / "events" / str(event_id)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{kind}{ext}"
    # Atomic replacement: a browser must never see a half-written clip.
    import uuid
    temporary = target.with_name(f".{uuid.uuid4().hex}.upload")
    try:
        with temporary.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
        await file.close()
    url = f"/media/events/{event_id}/{target.name}"
    if kind == "snapshot":
        e.snapshot_path = url
    else:
        e.clip_path = url
        e.extra = merge_extra(e.extra, {"clip": {"state": "ready"}})
    session.commit()
    out = to_out(e)
    await manager.broadcast({"type": "event_media_updated", "event": out.model_dump(mode="json")})
    return out


# ---------- события: реакция оператора ----------

@app.post("/api/events/{event_id}/status", response_model=EventDetail)
async def set_status(event_id: int, body: StatusIn, session: Session = Depends(get_session)):
    e = _get_event(session, event_id)
    if body.status == e.status:
        raise HTTPException(409, f"Событие уже в статусе {e.status}")
    session.add(StatusChange(event_id=e.id, from_status=e.status, to_status=body.status,
                             user=body.user, comment=body.comment, at=utcnow()))
    e.status, e.reviewed_by, e.reviewed_at = body.status, body.user, utcnow()
    if body.comment:
        e.comment = body.comment
    session.commit()
    session.refresh(e)
    out = to_out(e, detail=True)
    await manager.broadcast({"type": "event_updated", "event": out.model_dump(mode="json")})
    return out


# ---------- журнал: чтение ----------

@app.get("/api/events", response_model=EventPage)
def list_events(f: EventFilter = Depends(), limit: int = Query(50, ge=1, le=1000),
                offset: int = Query(0, ge=0), session: Session = Depends(get_session)):
    total = session.scalar(f.apply(select(func.count(Event.id))))
    rows = session.scalars(
        f.apply(select(Event)).order_by(Event.started_at.desc()).limit(limit).offset(offset)
    ).all()
    return EventPage(total=total or 0, items=[to_out(e) for e in rows])


FIELDS_RU = [
    ("id", "№"), ("started_at", "Начало"), ("confirmed_at", "Подтверждено"), ("ended_at", "Окончание"),
    ("pattern_name", "Нарушение"), ("severity", "Уровень"), ("area", "Участок"), ("camera_id", "Камера"),
    ("zone", "Зона"), ("track_id", "Трек"), ("confidence", "Уверенность"), ("status", "Статус"),
    ("reviewed_by", "Кто отреагировал"), ("reviewed_at", "Когда"), ("comment", "Комментарий"),
    ("snapshot_path", "Кадр"), ("clip_path", "Клип"),
]


@app.get("/api/events/export.csv")
def export_csv(f: EventFilter = Depends(), tz: str = DEFAULT_TZ, session: Session = Depends(get_session)):
    """Выгрузка журнала для проверяющих органов и расследований."""
    zone = _tz(tz)
    buf = io.StringIO()
    buf.write("﻿")  # BOM, чтобы Excel открыл кириллицу
    w = csv.writer(buf, delimiter=";")
    w.writerow([ru for _, ru in FIELDS_RU])
    for e in session.scalars(f.apply(select(Event)).order_by(Event.started_at)):
        row = []
        for key, _ in FIELDS_RU:
            v = getattr(e, key)
            if isinstance(v, datetime):
                v = as_utc(v).astimezone(zone).strftime("%Y-%m-%d %H:%M:%S")
            row.append("" if v is None else v)
        w.writerow(row)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": "attachment; filename=journal.csv"})


@app.get("/api/events/{event_id}", response_model=EventDetail)
def get_event(event_id: int, session: Session = Depends(get_session)):
    e = session.scalar(select(Event).where(Event.id == event_id).options(selectinload(Event.history)))
    if e is None:
        raise HTTPException(404, "Событие не найдено")
    return to_out(e, detail=True)


# ---------- статистика для дашбордов ----------

@app.get("/api/stats/summary")
def stats_summary(f: EventFilter = Depends(), tz: str = DEFAULT_TZ, session: Session = Depends(get_session)):
    return stats.summary(_filtered(session, f), _tz(tz))


@app.get("/api/stats/timeseries")
def stats_timeseries(f: EventFilter = Depends(), bucket: str = Query("day", pattern="^(hour|day)$"),
                     tz: str = DEFAULT_TZ, session: Session = Depends(get_session)):
    return stats.timeseries(_filtered(session, f), bucket, _tz(tz))


@app.get("/api/stats/heatmap")
def stats_heatmap(f: EventFilter = Depends(), rows: str = Query("area", pattern="^(area|camera|pattern)$"),
                  tz: str = DEFAULT_TZ, session: Session = Depends(get_session)):
    return stats.heatmap(_filtered(session, f), _tz(tz), rows)


@app.get("/api/stats/quality")
def stats_quality(f: EventFilter = Depends(), session: Session = Depends(get_session)):
    return stats.quality(_filtered(session, f))


# ---------- демо: тестовое событие для показа живой ленты ----------

@app.post("/api/dev/simulate", response_model=EventOut, status_code=201)
async def simulate_event(severity: str | None = Query(None, pattern="^(critical|warning)$"),
                         session: Session = Depends(get_session)):
    """Создаёт правдоподобное событие из уже известных паттернов и камер. Только при DEMO_MODE=1."""
    if not DEMO_MODE:
        raise HTTPException(404, "Демо-режим выключен")
    import random
    from datetime import timedelta

    recent = session.scalars(select(Event).order_by(Event.id.desc()).limit(300)).all()
    patterns = {e.pattern_id: (e.pattern_name, e.categories, e.severity, e.camera_id, e.area) for e in recent}
    choices = [(pid, *v) for pid, v in patterns.items() if severity is None or v[2] == severity]
    if not choices:
        choices = [("fall", "Падение, неподвижность, признаки потери сознания", ["behavior"], "critical",
                    "cam-03", "Механический цех")]
    pid, name, cats, sev, cam, area = random.choice(choices)
    now = utcnow()
    started = now - timedelta(seconds=random.uniform(1.5, 12))
    body = EventIn(pattern_id=pid, pattern_name=name, categories=list(cats), severity=sev, camera_id=cam,
                   area=area, track_id=random.randint(1, 999), started_at=started,
                   confirmed_at=now - timedelta(milliseconds=random.uniform(150, 600)),
                   frame_ts=now - timedelta(milliseconds=random.uniform(150, 600)),
                   confidence=round(random.uniform(0.7, 0.97), 3), extra={"demo": True, "simulated": True})
    return await create_event(body, session)


# ---------- живая лента ----------

@app.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()  # держим соединение; клиент может слать ping
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ---------- веб-интерфейс (дашборды) ----------

if WEB_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")
