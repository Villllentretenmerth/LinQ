"""Persistent educational events with bundled, attributed dataset videos."""
import json
from datetime import timedelta
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from .db import utcnow
from .models import Camera, Event

MEDIA = Path(__file__).resolve().parents[1] / 'web' / 'dataset-media'


def seed_startup_events(session):
    manifest = MEDIA / 'manifest.json'
    if not manifest.exists():
        return 0
    cases = json.loads(manifest.read_text())
    # Retire only our obsolete, untouched placeholders. Real events and reviews are preserved.
    for old in session.scalars(select(Event).where(Event.event_uid.like('startup-demo-v1-%'))):
        if old.extra.get('startup_demo') and old.status == 'new' and not old.history and not old.clip_path:
            session.delete(old)
    now, created = utcnow(), 0
    for index, case in enumerate(cases):
        key = case['key']
        if not all((MEDIA / f'{key}.{ext}').is_file() for ext in ('mp4', 'jpg')):
            continue
        uid = f'dataset-demo-v1-{key}'
        if session.scalar(select(Event.id).where(Event.event_uid == uid)) is not None:
            continue
        try:
            with session.begin_nested():
                camera_id = f'{case.get("source_group", "dataset")}-{case["pattern_id"]}'
                if session.get(Camera, camera_id) is None:
                    session.add(Camera(id=camera_id, name=case['dataset'], area='Учебные видеозаписи', source=case['source_url']))
                    session.flush()
                confirmed = now - timedelta(seconds=(len(cases) - index) * 10)
                session.add(Event(
                    event_uid=uid, pattern_id=case['pattern_id'], pattern_name=case['name'],
                    categories=['behavior'], severity='critical' if case['pattern_id'] == 'fall' else 'warning',
                    camera_id=camera_id, area='Учебные видеозаписи',
                    started_at=confirmed, confirmed_at=confirmed, frame_ts=confirmed,
                    received_at=confirmed, confidence=1, status='new',
                    clip_path=f'/ui/dataset-media/{key}.mp4', snapshot_path=f'/ui/dataset-media/{key}.jpg',
                    extra={'demo': True, 'startup_demo': True, 'dataset': case,
                           'description': case['description'], 'clip': {'state': 'ready', 'duration_sec': case['duration_sec']}},
                ))
                session.flush()
            created += 1
        except IntegrityError:
            if session.scalar(select(Event.id).where(Event.event_uid == uid)) is None:
                raise
    session.commit()
    return created
