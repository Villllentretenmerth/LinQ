# Замер моделей на LinQ TPU (Jupyter хакатона)

TPU хакатона — облачный: Jupyter (`hackathon.lin-q.ru`, через VPN) отправляет задачу в RabbitMQ
`10.0.114.30:5672`, воркер с TPU выполняет программу `.tpu` из `/auto/cache/<логин>/` и отвечает в нашу очередь.
Так устроен `examples/example.ipynb` организаторов; `linq_cloud.py` — тот же протокол, но с одним соединением
на всё время работы и временем инференса на самом воркере.

| Файл | Что делает |
|---|---|
| `linq_cloud.py` | клиент облака: `LinqCloud(program).infer({вход: тензор})`, `infer_many`, `throughput`; `program_info(.tpu)` — имена и формы входов/выходов из `metadata.json` внутри программы; `make_infer` — для конвейера `core/` |
| `linq_bench.py` | замер: p50/p95, FPS, цена кадра каскада, сколько камер выдержит узел; `bench_result.json` для журнала |
| `linq_bench.ipynb` | готовый ноутбук: пример resnet50 → проверка класса 257 → задержка → пропускная способность → сохранение; последний раздел — то же для `yolov8n-pose` |

## Порядок

1. Загрузить в Jupyter рядом с ноутбуком: `linq_cloud.py`, `linq_bench.py`, `linq_bench.ipynb`.
2. В первой ячейке вписать свой логин (`LOGIN`) и, если нужно, путь к папке примера (`EXAMPLES`).
3. Выполнить ячейки по порядку. Проверка resnet50 должна дать класс 257 (ньюфаундленд).
4. Скачать `bench_result.json` и загрузить в журнал (из папки `journal`):
   `python -m scripts.import_metrics bench_result.json --url http://localhost:8000`.
5. Когда организаторы вернут `yolov8n-pose_head.tpu` (и `yolov8n_head.tpu`) — положить рядом и выполнить раздел 5.

## Что меряется

- **Время на TPU** (`tpu_ms`) — воркер присылает `inference_time`; из него считаются загрузка TPU и запас по камерам.
  Единица в протоколе не описана — клиент определяет её сам (время воркера не может быть больше полного круга),
  а сырое значение видно в `info["inference_time_raw"]`.
- **Полный круг** (`rtt_ms`) — отправка → очередь → TPU → ответ. Ограничивает FPS одной камеры, если кадры
  отправлять строго по одному.
- **Пропускная способность** (`throughput`) — несколько задач в полёте, как от нескольких камер сразу:
  задач в секунду при `in_flight` = 1, 2, 4, 8, 16.

```python
from linq_cloud import LinqCloud
from linq_bench import Bench

tpu = LinqCloud("/auto/cache/<логин>/torch_resnet50_b1_o5_h1m.tpu")
b = Bench(node_id="linq-hackathon", node_name="LinQ TPU хакатона (облако)", tpu_cores=1)
b.measure("resnet50", lambda: tpu.infer({"input.1": x}), tpu_time=lambda r: r[1]["tpu_ms"])
b.table()
b.save("bench_result.json")
```

`calls_per_frame` — сколько раз модель вызывается на кадр в среднем (каскад: дорогие модели — только для
кандидатов). Первые 10 вызовов — прогрев, в статистику не входят. Без `tpu_time` замер считает время
вызова целиком, как раньше.
