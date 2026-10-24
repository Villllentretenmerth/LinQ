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
            assert page['total'] == 26
            assert len({e['clip_path'] for e in page['items']}) == 26
            assert all(e['extra']['demo'] for e in page['items'])
            for event in page['items']:
                assert client.get(event['snapshot_path']).status_code == 200
                response = client.get(event['clip_path'], headers={'Range': 'bytes=0-63'})
                assert response.status_code == 206
                assert b'ftyp' in response.content
            assert sum(e['extra']['dataset'].get('expected_violation') is False for e in page['items']) == 2
            assert client.get('/api/stats/summary').json()['total'] == 0
            assert client.get('/api/stats/summary?source=demo').json()['total'] == 26
            assert client.get('/api/stats/quality').json()['overall']['events'] == 0
            assert client.get('/api/stats/quality?source=demo').json()['overall']['events'] == 26
            assert client.get('/api/stats/timeseries').json() == []
            assert client.get('/api/stats/heatmap').json()['rows'] == []
            assert sum(x['total'] for x in client.get('/api/stats/timeseries?source=demo').json()) == 26
            eid = page['items'][0]['id']
            assert client.post(f'/api/events/{eid}/status', json={
                'status': 'false_alarm', 'user': 'operator', 'comment': 'Учебная проверка'
            }).status_code == 200
        with TestClient(main.app) as client:
            assert client.get('/api/events').json()['total'] == 26
            assert client.get('/api/events', params={'status': 'new'}).json()['total'] == 25
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
