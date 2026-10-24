"""Облачный инференс на LinQ TPU хакатона — тот же протокол, что в examples/example.ipynb.

Задача уходит в RabbitMQ (обменник tpu.direct, ключ input) с заголовком uri = путь к программе .tpu,
воркер с TPU отвечает в нашу очередь. Программа должна лежать в /auto/cache/<логин Jupyter>/ — один раз
скопировать из Jupyter: !cp yolov8n-pose_head.tpu /auto/cache/<логин>/. Брокер — 10.0.114.30:5672 в сети VPN
хакатона: из Jupyter он виден всегда, с ноутбука — если VPN пускает маршрут 10.0.114.0/24.

    from linq_cloud import LinqCloud, program_info
    tpu = LinqCloud("/auto/cache/hackathon12345/torch_resnet50_b1_o5_h1m.tpu")
    outs, info = tpu.infer({"input.1": x})   # outs: {имя выхода: float32}; info: rtt_ms, tpu_ms, worker_pid
    tpu.throughput({"input.1": x}, n=64, in_flight=8)   # сколько задач в секунду при нескольких в полёте
    program_info("torch_resnet50_b1_o5_h1m.tpu")        # имена и формы входов/выходов из metadata.json

Конвейер core/ (backend: linq) подключает его через make_infer — см. core/cameras.yaml, секция models.linq.
Нужны pika и tpu_pycloud_format (в Jupyter хакатона уже стоят; локально — pip install pika и колесо из examples).
"""
import json
import time
import uuid
import zipfile

import numpy as np
import pika
from tpu_pycloud_format import (MESSAGE_ID_ATTR_NAME, RESPONSE_ROUTING_KEYS_ATTR_NAME, TARGET_URI_ATTR_NAME,
                                PyTask, deserialize, serialize)

HOST = "10.0.114.30"          # cloud-брокер (из example.ipynb)
PORT = 5672
CREDENTIALS = ("guest", "guest")
EXCHANGE = "tpu.direct"
INPUT_QUEUE = "input"


class LinqError(RuntimeError):
    """Воркер не смог выполнить задачу: в ответе нет тензоров, есть status_code и error_message."""


def program_info(path: str) -> dict:
    """Входы и выходы программы .tpu (это zip, внутри metadata.json): имя → форма, тип, как на TPU."""
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("metadata.json"))

    def side(key: str) -> dict:
        res = {}
        for group in meta.get(key, {}).values():
            for name, d in group.items():
                res[name] = {"shape": d.get("user_shape"), "dtype": d.get("user_dtype"), "tpu_dtype": d.get("tpu_dtype")}
        return res

    return {"inputs": side("inputs"), "outputs": side("outputs"), "api_version": meta.get("iva_tpu_api_version")}


class LinqCloud:
    """Одно соединение с брокером и своя очередь ответов на всё время работы (в примере — на каждый вызов).
    Не потокобезопасен: на поток (или на модель под общей блокировкой) — свой экземпляр."""

    def __init__(self, program: str, host: str = HOST, port: int = PORT, credentials=CREDENTIALS,
                 timeout: float = 30.0, raw: bool = False):
        self.uri = program if "://" in program else f"file://{program}?raw={raw}"
        self.host, self.port, self.credentials, self.timeout = host, port, tuple(credentials), timeout
        self.conn = self.ch = None
        self._connect()

    # ---------- соединение ----------

    def _connect(self) -> None:
        self.close()
        self.conn = pika.BlockingConnection(pika.ConnectionParameters(
            self.host, self.port, credentials=pika.PlainCredentials(*self.credentials),
            heartbeat=600, blocked_connection_timeout=self.timeout))
        self.ch = self.conn.channel()
        self.reply = f"team94_{uuid.uuid4().hex[:8]}"
        # exclusive: очередь живёт, пока открыто соединение. auto_delete нельзя — брокер удаляет такую очередь,
        # как только _collect отписывается (cancel) после первого же вызова, и следующий падает с 404 NOT_FOUND
        self.ch.queue_declare(queue=self.reply, exclusive=True)
        self.ch.queue_bind(exchange=EXCHANGE, queue=self.reply, routing_key=self.reply)

    def close(self) -> None:
        if self.conn is not None and self.conn.is_open:
            try:
                self.conn.close()
            except Exception:
                pass
        self.conn = self.ch = None

    # ---------- задачи ----------

    def _publish(self, tensors: dict) -> str:
        mid = uuid.uuid4().hex
        self.ch.basic_publish(
            exchange=EXCHANGE, routing_key=INPUT_QUEUE, body=serialize(PyTask(dict(tensors))),
            properties=pika.BasicProperties(headers={
                TARGET_URI_ATTR_NAME: self.uri,
                RESPONSE_ROUTING_KEYS_ATTR_NAME: json.dumps([self.reply]),
                MESSAGE_ID_ATTR_NAME: mid,
            }))
        return mid

    def _collect(self, ids: list[str], started: float) -> dict[str, tuple[bytes, float]]:
        """Ждёт ответы на ids; чужие и опоздавшие ответы (от прошлых задач по тайм-ауту) пропускает."""
        want, got = set(ids), {}
        try:
            for method, props, body in self.ch.consume(self.reply, auto_ack=True, inactivity_timeout=0.2):
                if method is None:
                    if time.monotonic() - started > self.timeout:
                        raise TimeoutError(f"LinQ: нет ответа за {self.timeout:.0f} с ({len(want - got.keys())} задач)")
                    continue
                mid = (props.headers or {}).get(MESSAGE_ID_ATTR_NAME)
                if mid in want and mid not in got:
                    got[mid] = (body, time.monotonic())
                    if len(got) == len(want):
                        break
        finally:
            self.ch.cancel()
        return got

    @staticmethod
    def _decode(body: bytes, rtt_s: float) -> tuple[dict, dict]:
        data, _meta, worker = deserialize(body)
        if not data:
            raise LinqError(f"воркер LinQ: status_code={worker['status_code']}, {worker['error_message']}")
        raw = float(worker.get("inference_time") or 0.0)
        # единица inference_time не описана: время на воркере не больше полного круга, значит, если оно
        # меньше rtt в секундах — это секунды, иначе миллисекунды
        tpu_ms = raw * 1000 if raw <= rtt_s * 1.05 else raw
        info = {"rtt_ms": rtt_s * 1000, "tpu_ms": tpu_ms, "inference_time_raw": raw,
                "worker_pid": worker.get("worker_pid"), "status_code": worker.get("status_code")}
        return {k: np.array(v) for k, v in data.items()}, info   # копия: frombuffer отдаёт read-only

    def _run(self, batch: list[dict]) -> list[tuple[dict, dict]]:
        t0 = time.monotonic()
        ids = [self._publish(t) for t in batch]
        got = self._collect(ids, t0)
        return [self._decode(got[i][0], got[i][1] - t0) for i in ids]

    def infer(self, tensors: dict) -> tuple[dict, dict]:
        """Один инференс: ({имя выхода: массив}, {rtt_ms, tpu_ms, ...}). Обрыв связи — одно переподключение."""
        try:
            return self._run([tensors])[0]
        except (pika.exceptions.AMQPError, OSError):
            self._connect()
            return self._run([tensors])[0]

    def infer_many(self, batch: list[dict]) -> list[tuple[dict, dict]]:
        """Несколько задач сразу в полёте (например, кадры нескольких камер) — ответы в том же порядке."""
        return self._run(list(batch))

    def throughput(self, tensors: dict, n: int = 64, in_flight: int = 8) -> dict:
        """Пропускная способность: n задач пачками по in_flight. Показывает, сколько кадров в секунду узел
        отдаёт, когда запросы от разных камер идут параллельно, а не по одному."""
        t0, tpu, rtt = time.monotonic(), [], []
        for i in range(0, n, in_flight):
            for _, info in self.infer_many([tensors] * min(in_flight, n - i)):
                tpu.append(info["tpu_ms"])
                rtt.append(info["rtt_ms"])
        sec = time.monotonic() - t0
        return {"tasks": n, "in_flight": in_flight, "sec": round(sec, 3), "tasks_per_sec": round(n / sec, 2),
                "tpu_ms_mean": round(float(np.mean(tpu)), 3), "rtt_ms_mean": round(float(np.mean(rtt)), 2)}


def make_infer(program: str, input_name: str = "images", host: str = HOST, port: int = PORT,
               user: str = CREDENTIALS[0], password: str = CREDENTIALS[1], timeout: float = 10.0, **_):
    """Для core.infer.LinqBackend (models.linq.module: "bench.linq_cloud:make_infer"):
    infer(tensor) -> {имя выхода: массив}. Декодер головы в core узнаёт выходы по форме, порядок не важен."""
    client = LinqCloud(program, host=host, port=port, credentials=(user, password), timeout=timeout)
    return lambda tensor: client.infer({input_name: tensor})[0]
