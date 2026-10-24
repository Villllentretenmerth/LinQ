"""Ten persistent sample events for the first operator-screen visit."""
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from .db import utcnow
from .models import Camera, Event

CASES = (
    ('fall', 'Падение человека', 'critical', 'Механический цех'),
    ('run', 'Бег в зоне движения техники', 'warning', 'Склад отгрузки'),
    ('danger_zone', 'Вход в опасную зону', 'critical', 'Механический цех'),
    ('vehicle_proximity', 'Опасное сближение с погрузчиком', 'critical', 'Склад отгрузки'),
    ('phone', 'Использование телефона при работе', 'warning', 'Механический цех'),
    ('climb_fence', 'Перелезание через ограждение', 'critical', 'Периметр'),
    ('no_helmet', 'Отсутствие защитной каски', 'warning', 'Склад отгрузки'),
    ('smoking', 'Курение в запрещённой зоне', 'warning', 'Сварочный участок'),
    ('fire_smoke', 'Дым или открытый огонь', 'critical', 'Сварочный участок'),
    ('height_no_harness', 'Работа на высоте без страховки', 'critical', 'Электроучасток'),
)


def seed_startup_events(session):
    """Stable IDs prevent duplicates and preserve operator decisions across restarts."""
    now = utcnow()
    created = 0
    for index, (pattern, name, severity, area) in enumerate(CASES, 1):
        uid = f'startup-demo-v1-{index:02d}'
        if session.scalar(select(Event.id).where(Event.event_uid == uid)) is not None:
            continue
        try:
            # A savepoint makes concurrent server startup safe without losing other events.
            with session.begin_nested():
                camera_id = f'demo-cam-{index:02d}'
                if session.get(Camera, camera_id) is None:
                    session.add(Camera(id=camera_id, name=f'Демо-камера {index}', area=area,
                                       source='demo'))
                    session.flush()
                confirmed = now - timedelta(minutes=10 - index)
                session.add(Event(
                    event_uid=uid, pattern_id=pattern, pattern_name=name, categories=[],
                    severity=severity, camera_id=camera_id, area=area, track_id=index,
                    started_at=confirmed - timedelta(seconds=2), confirmed_at=confirmed,
                    frame_ts=confirmed, received_at=confirmed, confidence=0.9, status='new',
                    extra={'demo': True, 'startup_demo': True,
                           'description': 'Учебный пример для проверки интерфейса. Не результат детектора.'},
                ))
                session.flush()
            created += 1
        except IntegrityError:
            # Only ignore a race which actually created this same sample.
            if session.scalar(select(Event.id).where(Event.event_uid == uid)) is None:
                raise
    session.commit()
    return created
