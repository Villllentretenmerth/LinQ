"""Таблицы журнала: камеры, события, история смены статусов (аудит для доказательной базы)."""
from datetime import datetime

from sqlalchemy import JSON, BigInteger, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base, utcnow

# Жизненный цикл события в интерфейсе оператора
STATUSES = ("new", "confirmed", "false_alarm", "sent_to_master", "resolved")
# Какие статусы считаются реальным нарушением при расчёте precision
TRUE_STATUSES = ("confirmed", "sent_to_master", "resolved")

_BigId = BigInteger().with_variant(Integer, "sqlite")  # autoincrement в SQLite работает только с INTEGER


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    area: Mapped[str] = mapped_column(String(100), index=True)  # цех / участок
    source: Mapped[str | None] = mapped_column(String(500))      # RTSP, файл или папка кадров


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(_BigId, primary_key=True, autoincrement=True)
    event_uid: Mapped[str] = mapped_column(String(64), unique=True)  # ключ идемпотентности от движка

    pattern_id: Mapped[str] = mapped_column(String(64))
    pattern_name: Mapped[str] = mapped_column(String(200))
    categories: Mapped[list] = mapped_column(JSON, default=list)
    severity: Mapped[str] = mapped_column(String(16))  # critical | warning

    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"))
    area: Mapped[str] = mapped_column(String(100))        # дублируем для быстрых фильтров
    zone: Mapped[str | None] = mapped_column(String(100))
    track_id: Mapped[int | None] = mapped_column(Integer)

    # Временные метки: из них считаются время до тревоги и задержка конвейера
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))    # первый кадр-кандидат
    confirmed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # подтверждено, ушла тревога
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    frame_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # время кадра подтверждения
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    confidence: Mapped[float] = mapped_column(Float)
    bbox: Mapped[list | None] = mapped_column(JSON)       # [x1, y1, x2, y2] на кадре-доказательстве
    snapshot_path: Mapped[str | None] = mapped_column(String(500))
    clip_path: Mapped[str | None] = mapped_column(String(500))

    status: Mapped[str] = mapped_column(String(20), default="new", index=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(100))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    comment: Mapped[str | None] = mapped_column(Text)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)

    history: Mapped[list["StatusChange"]] = relationship(
        back_populates="event", cascade="all, delete-orphan", order_by="StatusChange.at"
    )

    __table_args__ = (
        Index("ix_events_started", "started_at"),
        Index("ix_events_camera_started", "camera_id", "started_at"),
        Index("ix_events_pattern_started", "pattern_id", "started_at"),
        Index("ix_events_area_started", "area", "started_at"),
    )


class Node(Base):
    """Вычислительный узел с ускорителями LinQ (ВМ хакатона, сервер на площадке, edge-узел)."""

    __tablename__ = "nodes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    host: Mapped[str | None] = mapped_column(String(200))
    tpu_cores: Mapped[int] = mapped_column(Integer, default=1)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_source: Mapped[str | None] = mapped_column(String(16))


class NodeMetric(Base):
    """Окно метрик узла: сколько кадров обработано, сколько времени занят TPU, задержки моделей."""

    __tablename__ = "node_metrics"

    id: Mapped[int] = mapped_column(_BigId, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.id"))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))   # конец окна
    window_sec: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(16))                 # pipeline | bench | demo
    streams: Mapped[int] = mapped_column(Integer)
    frames: Mapped[int] = mapped_column(Integer)
    dropped: Mapped[int] = mapped_column(Integer, default=0)        # кадры, выброшенные читателем «последний кадр»
    fps_total: Mapped[float] = mapped_column(Float)
    fps_stream_min: Mapped[float | None] = mapped_column(Float)
    tpu_busy_ms: Mapped[float] = mapped_column(Float)               # суммарное время инференса на TPU за окно
    tpu_util: Mapped[float] = mapped_column(Float)                  # 0..1
    cpu_util: Mapped[float | None] = mapped_column(Float)
    models: Mapped[dict] = mapped_column(JSON, default=dict)        # {"yolo-pose": {calls, mean_ms, p50_ms, p95_ms}}
    per_stream: Mapped[dict] = mapped_column(JSON, default=dict)    # {"cam-01": 9.8}

    __table_args__ = (Index("ix_node_metrics_node_ts", "node_id", "ts"),)


class StatusChange(Base):
    """Кто, когда и как отреагировал на событие — отметка о реакции ответственного лица."""

    __tablename__ = "event_status_history"

    id: Mapped[int] = mapped_column(_BigId, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), index=True)
    from_status: Mapped[str] = mapped_column(String(20))
    to_status: Mapped[str] = mapped_column(String(20))
    user: Mapped[str] = mapped_column(String(100))
    comment: Mapped[str | None] = mapped_column(Text)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    event: Mapped[Event] = relationship(back_populates="history")
