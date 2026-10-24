"""Форматы данных API."""
import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .db import as_utc

Severity = Literal["critical", "warning"]
Status = Literal["new", "confirmed", "false_alarm", "sent_to_master", "resolved"]


class EventIn(BaseModel):
    """Событие, которое движок отправляет после подтверждения нарушения."""

    event_uid: str = Field(default_factory=lambda: uuid.uuid4().hex)
    pattern_id: str
    pattern_name: str
    categories: list[str] = []
    severity: Severity
    camera_id: str
    camera_name: str | None = None
    area: str
    zone: str | None = None
    track_id: int | None = None
    started_at: datetime
    confirmed_at: datetime
    frame_ts: datetime | None = None
    confidence: float = Field(ge=0, le=1)
    bbox: list[float] | None = None
    snapshot_path: str | None = None
    clip_path: str | None = None
    extra: dict = {}

    @field_validator("started_at", "confirmed_at", "frame_ts")
    @classmethod
    def _utc(cls, v):
        return as_utc(v)


class EventEnd(BaseModel):
    ended_at: datetime

    @field_validator("ended_at")
    @classmethod
    def _utc(cls, v):
        return as_utc(v)


class StatusIn(BaseModel):
    status: Status
    user: str
    comment: str | None = None


class StatusChangeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    from_status: str
    to_status: str
    user: str
    comment: str | None
    at: datetime


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    event_uid: str
    pattern_id: str
    pattern_name: str
    categories: list[str]
    severity: str
    camera_id: str
    area: str
    zone: str | None
    track_id: int | None
    started_at: datetime
    confirmed_at: datetime
    ended_at: datetime | None
    frame_ts: datetime | None
    received_at: datetime
    confidence: float
    bbox: list[float] | None
    snapshot_path: str | None
    clip_path: str | None
    status: str
    reviewed_by: str | None
    reviewed_at: datetime | None
    comment: str | None
    extra: dict
    time_to_alert_ms: int | None = None       # confirmed_at - started_at
    pipeline_latency_ms: int | None = None    # received_at - frame_ts

    @field_validator("started_at", "confirmed_at", "ended_at", "frame_ts", "received_at", "reviewed_at")
    @classmethod
    def _utc(cls, v):
        return as_utc(v)


class EventDetail(EventOut):
    history: list[StatusChangeOut] = []


class EventPage(BaseModel):
    total: int
    items: list[EventOut]
