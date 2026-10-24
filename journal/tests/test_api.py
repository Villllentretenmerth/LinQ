"""Проверка API журнала. По умолчанию на временной SQLite.
Для проверки на PostgreSQL задайте TEST_DATABASE_URL — ОТДЕЛЬНУЮ тестовую базу: тесты пересоздают таблицы."""
import os
import tempfile
from datetime import datetime, timedelta, timezone

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = os.getenv("TEST_DATABASE_URL", f"sqlite:///{_tmp}/test.db")
os.environ["MEDIA_DIR"] = os.path.join(_tmp, "media")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db import Base, engine  # noqa: E402
from app.main import app  # noqa: E402

T0 = datetime(2026, 10, 24, 5, 0, tzinfo=timezone.utc)  # 08:00 по Москве


def ev(**kw):
    base = dict(pattern_id="fall", pattern_name="Падение", categories=["behavior"], severity="critical",
                camera_id="cam-03", camera_name="Цех — станки", area="Механический цех", zone="z1",
                track_id=7, started_at=T0.isoformat(), confirmed_at=(T0 + timedelta(seconds=11)).isoformat(),
                frame_ts=(T0 + timedelta(seconds=11)).isoformat(), confidence=0.9)
    base.update(kw)
    return base


@pytest.fixture(scope="module")
def client():
    Base.metadata.drop_all(engine)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("app.main.DEMO_SEED", False)
        with TestClient(app) as c:
            yield c
    Base.metadata.drop_all(engine)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_create_is_idempotent(client):
    r1 = client.post("/api/events", json=ev(event_uid="abc"))
    assert r1.status_code == 201
    body = r1.json()
    assert body["time_to_alert_ms"] == 11000
    assert body["pipeline_latency_ms"] is not None
    r2 = client.post("/api/events", json=ev(event_uid="abc"))
    assert r2.json()["id"] == body["id"]
    assert client.get("/api/events").json()["total"] == 1
    assert client.get("/api/cameras").json()[0]["area"] == "Механический цех"


def test_filters(client):
    client.post("/api/events", json=ev(event_uid="p1", pattern_id="phone", pattern_name="Телефон",
                                       categories=["attribute", "behavior"], severity="warning",
                                       camera_id="cam-01", area="Склад отгрузки",
                                       started_at=(T0 + timedelta(hours=15)).isoformat(),
                                       confirmed_at=(T0 + timedelta(hours=15, seconds=2)).isoformat()))
    assert client.get("/api/events", params={"category": "attribute"}).json()["total"] == 1
    assert client.get("/api/events", params={"category": "behavior"}).json()["total"] == 2
    assert client.get("/api/events", params={"camera_id": "cam-01"}).json()["total"] == 1
    assert client.get("/api/events", params={"area": "Механический цех"}).json()["total"] == 1
    assert client.get("/api/events", params={"severity": "critical"}).json()["total"] == 1
    assert client.get("/api/events", params={"zone": "z1"}).json()["total"] == 2
    r = client.get("/api/events", params={"date_from": (T0 + timedelta(hours=1)).isoformat()})
    assert r.json()["total"] == 1


def test_status_and_history(client):
    eid = client.get("/api/events", params={"pattern_id": "fall"}).json()["items"][0]["id"]
    r = client.post(f"/api/events/{eid}/status", json={"status": "confirmed", "user": "оператор-1"})
    assert r.status_code == 200 and r.json()["status"] == "confirmed"
    assert client.post(f"/api/events/{eid}/status",
                       json={"status": "confirmed", "user": "оператор-1"}).status_code == 409
    client.post(f"/api/events/{eid}/status",
                json={"status": "sent_to_master", "user": "оператор-1", "comment": "мастеру"})
    d = client.get(f"/api/events/{eid}").json()
    assert [h["to_status"] for h in d["history"]] == ["confirmed", "sent_to_master"]
    assert d["reviewed_by"] == "оператор-1"


def test_end_and_media(client):
    eid = client.get("/api/events", params={"pattern_id": "fall"}).json()["items"][0]["id"]
    r = client.post(f"/api/events/{eid}/end", json={"ended_at": (T0 + timedelta(minutes=2)).isoformat()})
    assert r.json()["ended_at"].startswith("2026-10-24T05:02")
    r = client.post(f"/api/events/{eid}/media", params={"kind": "snapshot"},
                    files={"file": ("f.jpg", b"\xff\xd8fakejpeg", "image/jpeg")})
    url = r.json()["snapshot_path"]
    assert client.get(url).content == b"\xff\xd8fakejpeg"


def test_stats(client):
    pid = client.get("/api/events", params={"pattern_id": "phone"}).json()["items"][0]["id"]
    client.post(f"/api/events/{pid}/status", json={"status": "false_alarm", "user": "оператор-2"})
    s = client.get("/api/stats/summary").json()
    assert s["total"] == 2 and s["by_shift"] == {"day": 1, "night": 1}  # 08:00 и 23:00 по Москве
    ts = client.get("/api/stats/timeseries", params={"bucket": "hour"}).json()
    assert ts[0]["total"] == 1 and sum(x["total"] for x in ts) == 2 and len(ts) == 16
    hm = client.get("/api/stats/heatmap").json()
    assert sum(sum(r) for r in hm["values"]) == 2 and len(hm["values"][0]) == 24
    q = client.get("/api/stats/quality").json()
    assert q["by_pattern"]["fall"]["precision"] == 1.0
    assert q["by_pattern"]["phone"]["precision"] == 0.0
    assert q["overall"]["precision"] == 0.5
    assert client.get("/api/stats/summary", params={"tz": "Mars/Base"}).status_code == 400


def test_export_csv(client):
    r = client.get("/api/events/export.csv")
    text = r.content.decode("utf-8")
    assert text.startswith("﻿№;Начало") and text.count("\n") == 3


def test_websocket_feed(client):
    with client.websocket_connect("/ws/events") as ws:
        client.post("/api/events", json=ev(event_uid="ws1"))
        msg = ws.receive_json()
        assert msg["type"] == "event_created" and msg["event"]["event_uid"] == "ws1"


def test_simulate_and_ui(client):
    r = client.post("/api/dev/simulate", params={"severity": "critical"})
    assert r.status_code == 201 and r.json()["severity"] == "critical"
    assert client.get("/ui/").status_code == 200
    assert client.get("/", follow_redirects=False).headers["location"] == "/ui/"


def test_metrics_push_nodes_capacity(client):
    body = dict(node_id="n1", node_name="Узел 1", tpu_cores=1, window_sec=5, frames=250,
                per_stream={"a": 10, "b": 9.5, "c": 10, "d": 10.5, "e": 10}, tpu_busy_ms=2500,
                models={"yolo-pose": {"calls": 250, "mean_ms": 10, "p50_ms": 9.8, "p95_ms": 13}})
    r = client.post("/api/metrics", json=body)
    assert r.status_code == 201 and r.json()["fps_total"] == 50 and r.json()["tpu_util"] == 0.5
    n = client.get("/api/metrics/nodes", params={"target_fps": 5, "util_cap": 0.8}).json()[0]
    assert n["online"] and n["latest"]["fps_stream_min"] == 9.5 and n["latest"]["streams"] == 5
    cap = n["capacity"]
    assert cap["tpu_ms_per_frame"] == 10 and cap["max_streams"] == 16 and cap["headroom"] == 11
    # собственная загрузка от LinQ важнее расчётной
    client.post("/api/metrics", json={**body, "tpu_util": 0.9})
    assert client.get("/api/metrics/nodes").json()[0]["latest"]["tpu_util"] == 0.9
    s = client.get("/api/metrics/series", params={"minutes": 5}).json()
    assert len(s["n1"]) == 2


def test_demo_metrics_seed(client):
    from app.db import SessionLocal
    from app.metrics import seed_demo_metrics
    with SessionLocal() as s:
        assert seed_demo_metrics(s, minutes=3) == 6
    nodes = {n["id"]: n for n in client.get("/api/metrics/nodes").json()}
    d = nodes["linq-node-1"]
    assert d["source"] == "demo" and 0 < d["latest"]["tpu_util"] < 1 and d["capacity"]["max_streams"] > 0


def test_clip_updates_live_and_operator_decision_survives(client):
    with client.websocket_connect('/ws/events') as ws:
        e = client.post('/api/events', json=ev(event_uid='clip-test', extra={
            'backend': 'onnx', 'clip': {'state': 'recording', 'requested_before_sec': 3}})).json()
        assert ws.receive_json()['type'] == 'event_created'
        eid = e['id']
        r = client.patch(f'/api/events/{eid}', json={'extra': {'clip': {'state': 'encoding'}}})
        assert r.status_code == 200
        assert ws.receive_json()['event']['extra']['clip']['state'] == 'encoding'
        assert r.json()['extra']['clip']['requested_before_sec'] == 3
        r = client.post(f'/api/events/{eid}/media', params={'kind': 'clip'},
                        files={'file': ('clip.mp4', b'video-bytes', 'video/mp4')})
        assert r.status_code == 200
        message = ws.receive_json()
        assert message['type'] == 'event_media_updated'
        assert message['event']['extra']['clip']['state'] == 'ready'
        url = r.json()['clip_path']
        assert client.get(url).content == b'video-bytes'
        assert client.get(url, headers={'Range': 'bytes=0-4'}).status_code == 206
        client.post(f'/api/events/{eid}/status', json={
            'status': 'confirmed', 'user': 'tester', 'comment': 'Reviewed video'})
        assert ws.receive_json()['event']['status'] == 'confirmed'
        client.patch(f'/api/events/{eid}', json={'extra': {
            'followup': {'condition': 'Неподвижен 10 с'}, 'clip': {'partial': True}}})
        assert ws.receive_json()['event']['extra']['followup']['condition'] == 'Неподвижен 10 с'
        detail = client.get(f'/api/events/{eid}').json()
        assert detail['status'] == 'confirmed' and detail['clip_path'] == url
        assert detail['history'][0]['comment'] == 'Reviewed video'
        assert detail['history'][0]['at'].endswith(('Z', '+00:00'))
        assert detail['extra']['backend'] == 'onnx'
        assert detail['extra']['clip']['state'] == 'ready'
        assert client.patch(f'/api/events/{eid}', json={'status': 'false_alarm'}).status_code == 422
        assert client.patch('/api/events/999999', json={'extra': {}}).status_code == 404
