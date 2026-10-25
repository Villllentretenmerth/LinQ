"""Upload queue, validation, zero-event completion and interrupted-job recovery."""
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from app import main, uploads


@pytest.fixture
def upload_client(tmp_path, monkeypatch):
    monkeypatch.setattr(uploads, 'UPLOAD_DIR', tmp_path / 'uploads')
    monkeypatch.setattr(uploads, 'config', lambda: {'models': {}, 'cameras': [{'id':'camera', 'name':'Camera',
        'patterns':['fall'], 'zones':[{'name':'z'}], 'fps':10}]})
    class Process:
        def __init__(self, argv, **kw):
            folder = Path(argv[-1])
            request = json.loads((folder / 'request.json').read_text())
            assert request['camera']['zones'] == []
            assert request['camera']['overrides']['fall']['schedule'] == 'always'
            (folder / 'result.json').write_text(json.dumps({'events':0,'event_ids':[], 'frames':20, 'patterns':['fall']}))
        def wait(self, **kw): return 0
        def terminate(self): pass
    monkeypatch.setattr(uploads.subprocess, 'Popen', Process)
    with TestClient(main.app) as client:
        yield client


def test_upload_no_events_and_validation(upload_client, monkeypatch):
    client = upload_client
    assert client.get('/api/uploads/options').json()['available']
    assert client.post('/api/uploads', files={'file':('x.txt',b'x')}).status_code == 415
    assert client.post('/api/uploads', files={'file':('x.mp4',b'')}).status_code == 422
    assert client.post('/api/uploads', data={'profile':'missing'}, files={'file':('x.mp4',b'x')}).status_code == 422
    monkeypatch.setattr(uploads, 'MAX_BYTES', 4)
    assert client.post('/api/uploads', files={'file':('x.mp4',b'12345')}).status_code == 413
    r = client.post('/api/uploads', files={'file':('../../file.mp4',b'data')})
    assert r.status_code == 202
    jid = r.json()['id']
    for _ in range(100):
        job = client.get(f'/api/uploads/{jid}').json()
        if job['state'] in ('done', 'failed'): break
        time.sleep(.01)
    assert job['state'] == 'done'
    assert job['filename'] == 'file.mp4'
    assert job['message'] == 'Нарушений не обнаружено.'
    assert job['result']['event_ids'] == []
    assert client.get('/api/uploads/bad').status_code == 404
    assert len(list(uploads.UPLOAD_DIR.iterdir())) == 1


def test_busy_queue_and_worker_failure(upload_client, monkeypatch):
    manager = uploads.manager
    for _ in range(3): manager.slots.acquire()
    try:
        assert upload_client.post('/api/uploads', files={'file':('x.mp4',b'x')}).status_code == 429
    finally:
        for _ in range(3): manager.slots.release()
    def fail(*a, **kw): raise RuntimeError('worker failed')
    monkeypatch.setattr(uploads.subprocess, 'Popen', fail)
    jid = upload_client.post('/api/uploads', files={'file':('x.mp4',b'x')}).json()['id']
    for _ in range(100):
        job = upload_client.get(f'/api/uploads/{jid}').json()
        if job['state'] == 'failed': break
        time.sleep(.01)
    assert job['state'] == 'failed'


def test_restart_marks_interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(uploads, 'UPLOAD_DIR', tmp_path)
    folder = tmp_path / ('a' * 32); folder.mkdir()
    (folder / 'status.json').write_text(json.dumps({'state':'processing'}))
    manager = uploads.Jobs()
    try: assert json.loads((folder / 'status.json').read_text())['state'] == 'failed'
    finally: manager.close()
