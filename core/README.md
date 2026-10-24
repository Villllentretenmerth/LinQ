# Конвейер: видео → нарушения → события в журнале

```
источник кадров ─► yolo-pose (люди + 17 точек) ─► ByteTrack ─► движок карточек patterns/*.yaml ─► журнал
(RTSP / видео /    yolo-det (машины, телефоны) ─► ByteTrack    зоны, N из M, дедупликация,        событие + кадр,
 папка кадров)                                                  расписание                          метрики узла
```

Модели — предобученные YOLOv8 из Ultralytics (COCO), без дообучения. Пока нет своих `behavior_v1` и
`crop_attr_v1`, поведение считается по `fallback_rule` карточек из скелета и трека: наклон корпуса,
скорость в ростах в секунду, падение центра тела.

## Запуск на своём видео

```bash
pip install -r core/requirements.txt                    # модели уже лежат в core/models (или: python core/export_models.py)

python -m core.tools.grid D:/video/test.mp4             # первый кадр с сеткой координат -> test_grid.jpg
# в core/cameras.yaml: source, зоны (полигоны в пикселях кадра) и список patterns камеры

cd journal && docker compose up -d && cd ..             # журнал и дашборд на http://localhost:8000
python -m core.run --config core/cameras.yaml --journal http://localhost:8000 --offline --source D:/video/test.mp4
```

- `--offline` — видео и папки читаются покадрово, без пропусков; время в правилах — время видео. Без него
  файл играет с исходным FPS, как камера, а лишние кадры выбрасываются (уходят в метрики как `dropped`).
- `--source` — подменить источник у первой камеры: RTSP-адрес, видеофайл или папка кадров (сортируются по имени).
- `--ignore-schedule` — все карточки активны круглосуточно (у `danger_zone` в карточке 07:55–12:00).
- Без `--journal` — сухой прогон: события в логе, кадры-доказательства и `events.jsonl` в `--evidence-dir`
  (по умолчанию `%TEMP%/team94-evidence`).

В журнале: события на вкладках «Оператор» и «Журнал» с кадром (рамка, подпись, зоны), а на вкладке
«Система» — узел из `node:` конфига с `source=pipeline`: FPS, задержки yolo-pose и yolo-det, выброшенные кадры.

## Как снимать, чтобы сработало

Человек целиком в кадре и ростом не меньше ~150 px (для телефона — 180 px, так задано в карточке).
Камера сверху-сбоку, как на производстве. Перед каждым сценарием 2–3 секунды просто стоять в кадре:
трекеру и правилам нужна история.

| Карточка | Что сделать в кадре | Когда тревога |
|---|---|---|
| `fall` | постоять 3 с, упасть, лежать 12 с и больше | ~1,5–2 с после падения; через 10 с неподвижности — второй этап |
| `run` | бежать не меньше 3 с в зоне `tech_area` (быстрее ~1,5 роста/с) | через ~2,5 с бега |
| `danger_zone` | зайти в зону `rack_forklift` и стоять | через ~3 с (N из M + удержание 2 с) |
| `vehicle_proximity` | машина (car/truck из COCO) едет ближе одного роста от ног | через ~1 с |
| `phone` | телефон в руке, крупно, человек идёт (или кисть в зоне `equipment`) | через ~2 с + удержание 3 с |
| `climb_fence` | пересечь линию с тегом `fence` внутри зоны `fence_area` | через ~1,5 с |

Повторы не дублируются: одно событие на трек и блокировка повторов `cooldown_sec` по ключу
«камера + зона + тип», в том числе если после перекрытия человек получил новый ID трека.

## Что работает и что выключено

| Карточка | Как считается сейчас |
|---|---|
| `fall` | `torso_angle_deg > 60 and center_drop_heights > 0.4`; неподвижность 10 с — второй этап |
| `run` | `speed_heights_per_sec > 1.5` + ноги в зоне `tech_area` |
| `danger_zone` | точка ног в полигоне + расписание карточки |
| `vehicle_proximity` | COCO car/truck, расстояние от ног до рамки ≤ 1 роста, техника движется |
| `phone` | COCO cell phone у кисти (вместо `crop_attr_v1`) |
| `climb_fence` | линия `fence` (ветка `behavior_v1` выключена) |
| `crawl_under`, `carry_wrong` | правило уже работает — включите карточку и разметьте зоны `under_conveyor` и т. п. |
| `fire_smoke`, `no_helmet`, `smoking`, `bare_hands_wires`, `height_no_harness`, `carry_no_ppe`, `fight` | нужна своя модель (`yolo-det` с огнём, `crop_attr_v1`, `pair_v1`) — пропускаются с предупреждением в лог |

Ограничения:
- **Второй этап падения** («лежит без движения 10 с») конвейер отправляет как `PATCH /api/events/{id}`
  с `{"extra": {...}}`. В журнале такого метода пока нет, поэтому этап пишется только в лог и `events.jsonl`.
- Клип-доказательство (`evidence.clip_sec`) пока не пишется, только кадр.
- На CPU ноутбука ~3 кадра/с на поток (две модели по ~100–150 мс), это для разработки. Скорость считается на TPU.

## LinQ TPU

Квантование и компиляцию могут сделать организаторы, а можно и самим (см. «Своя компиляция» ниже). Путь такой:

1. Модели — `core/models/yolov8n-pose_head.onnx` и `core/models/yolov8n_head.onnx`: вход `images`
   1×3×640×640 float32 (RGB 0..1 после letterbox, без mean/std), opset 13, без NMS. Граф обрезан перед
   декодированием головы, как в документации SDK для `yolov8n_pose`. На выходе 9 (поза) и 6 (детектор)
   сырых свёрток, их декодирует `core.infer.YoloHeadDecoder` на CPU. Пересобрать: `python core/export_models.py`
   (сам режет через `core/tools/split_onnx.py` и сверяет с целой моделью).
2. Проверить тот же путь данных у себя: `models.backend: onnx_head` (голова на ONNX Runtime + декодер на numpy).
3. **Облако хакатона** (как `examples/example.ipynb`): TPU стоит не у нас, задачи уходят через RabbitMQ
   `10.0.114.30:5672` в сети VPN, программа указывается путём на сервере. Полученные `.tpu` скопировать
   из Jupyter в `/auto/cache/<логин>/` и в `core/cameras.yaml` раскомментировать облачный вариант:

```yaml
models:
  backend: linq
  linq:
    module: bench.linq_cloud:make_infer
    pose_program: /auto/cache/hackathonXXXXX/yolov8n-pose_head.tpu
    det_program: /auto/cache/hackathonXXXXX/yolov8n_head.tpu
```

   Конвейер тогда работает на ноутбуке, а модели — на TPU, если с ноутбука виден брокер
   (`Test-NetConnection 10.0.114.30 -Port 5672` при включённом VPN). Если не виден — запускать `core` в Jupyter.
   Нужны `pip install pika` и колесо `tpu_pycloud_format` из примера организаторов.
   Время в метриках узла при этом — полный круг через брокер, а не чистое время TPU; чистое меряет
   `bench/linq_bench.ipynb` (воркер присылает `inference_time`).
4. Локальный TPU (есть `/dev/tpu*`): `.tpu` в `core/models/`, `backend: linq` без `module` — вызов через pytpu,
   как в `usage_example_python_api` (`Device.open` → `load` → `inference.sync`).

### Своя компиляция (SDK `tpu_framework_2026.9.45.zip`)

SDK работает только на Linux x86_64 + Python 3.12, поэтому — в Docker. Пример SDK калибрует int8 на случайном
шуме; для YOLO нужны реальные кадры, иначе съезжают рамки и точки скелета.

```bash
# 1. Калибровочные кадры из своих роликов (на Windows, обычное окружение core): ~100 кадров разных сцен
python -m core.tools.make_calib "D:/video/*.mp4" --n 100 --out calib.npy --preview calib_preview

# 2. Образ с SDK (один раз; ~8,6 ГБ, сборка 10–20 минут). Контекст — распакованная папка tpu_framework
docker build -t team94-linq-sdk -f core/tools/linq_sdk.Dockerfile C:/Users/<вы>/Downloads/sdk/tpu_framework

# 3. Квантование + компиляция обеих голов (≈1 минута на модель при 3 кадрах калибровки; на 100 кадрах дольше)
docker run --rm -v "C:/путь/к/team94-safety:/repo" -w /repo team94-linq-sdk python core/tools/compile_tpu.py \
    core/models/yolov8n-pose_head.onnx core/models/yolov8n_head.onnx --calib calib.npy --out core/models/tpu
```

В Git Bash перед `docker run` добавьте `MSYS_NO_PATHCONV=1`, иначе пути `/repo` исказятся.

Результат — `core/models/tpu/yolov8n-pose_head_b1_o5_h1m.tpu`, `yolov8n_head_b1_o5_h1m.tpu` и отчёты `*.report.json`.
Скрипт сам пробует O5, а если не собирается — O1. `--verify 8` (по умолчанию) сравнивает квантованную модель
с исходной ONNX на CPU: косинусная близость по каждому выходу головы. Ниже ~0,98 у выходов `cv4.*`
(ключевые точки) — повод взять больше и разнообразнее кадров или `--samplewise-percentile 99.9`.
Дальше `.tpu` — в Jupyter, раздел 5 `bench/linq_bench.ipynb`.

Проверено 24.10 на SDK 2026.9.45: обе головы компилируются в O5, вход `images` 1×3×640×640, 9 и 6 выходов —
ровно то, что ждёт `YoloHeadDecoder`.

## Устройство

| Файл | Что внутри |
|---|---|
| `run.py` | точка входа, поток на камеру, отправка метрик |
| `ingest.py` | RTSP «последний кадр», видеофайл с темпом исходного FPS, папка кадров, покадровый офлайн |
| `infer.py` | бэкенды `onnx` / `onnx_head` / `linq`, letterbox, декодер головы, NMS |
| `tracking.py` | ByteTrack (`frame_rate` = FPS обработки), сшивка истории при смене ID на падении |
| `features.py` | признаки скелета для `fallback_rule`, безопасный разбор правил |
| `engine.py` | карточки: условия, N из M + `hold_sec`, дедупликация, расписание, второй этап `still` |
| `geometry.py` | зоны, линии, расстояния |
| `render.py` | кадр-доказательство |
| `sink.py` | очередь отправки в журнал с повторами (`journal/client`) |
| `cameras.yaml` | узел, модели, камеры, зоны, включённые карточки и поправки к ним |

Тесты логики без моделей: `python -m pytest core/tests -q`.
