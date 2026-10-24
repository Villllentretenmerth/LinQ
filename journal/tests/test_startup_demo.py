"""Startup events remain reviewable and do not duplicate across restarts."""
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from app import main
from app.models import Event


def test_startup_samples_and_restart(tmp_path, monkeypatch):
    engine = create_engine(f'sqlite:///{tmp_path}/startup.db', connect_args={'check_same_thread': False})
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(main, 'engine', engine)
    monkeypatch.setattr(main, 'SessionLocal', sessions)
    monkeypatch.setattr(main, 'DEMO_MODE', True)
    monkeypatch.setattr(main, 'DEMO_SEED', True)
    def db():
        with sessions() as session:
            yield session
    main.app.dependency_overrides[main.get_session] = db
    try:
        with TestClient(main.app) as client:
            page = client.get('/api/events', params={'status': 'new'}).json()
            assert page['total'] == 10
            assert len({e['pattern_id'] for e in page['items']}) == 10
            assert all(e['extra']['demo'] for e in page['items'])
            eid = page['items'][0]['id']
            assert client.post(f'/api/events/{eid}/status', json={
                'status': 'false_alarm', 'user': 'operator', 'comment': 'Учебная проверка'
            }).status_code == 200
        with TestClient(main.app) as client:
            assert client.get('/api/events').json()['total'] == 10
            assert client.get('/api/events', params={'status': 'new'}).json()['total'] == 9
            detail = client.get(f'/api/events/{eid}').json()
            assert detail['status'] == 'false_alarm'
            assert detail['history'][0]['comment'] == 'Учебная проверка'
    finally:
        main.app.dependency_overrides.pop(main.get_session, None)
        engine.dispose()


def test_production_startup_has_no_samples(tmp_path, monkeypatch):
    engine = create_engine(f'sqlite:///{tmp_path}/production.db')
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(main, 'engine', engine)
    monkeypatch.setattr(main, 'SessionLocal', sessions)
    monkeypatch.setattr(main, 'DEMO_MODE', False)
    monkeypatch.setattr(main, 'DEMO_SEED', True)
    with TestClient(main.app):
        with sessions() as session:
            assert session.scalars(select(Event)).all() == []
    engine.dispose()
