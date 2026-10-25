# behavior_v1 на открытых данных

Классификатор поведения по окну скелета (2 с, 10 к/с): `normal` / `fall` / `run`. Признаки — `core/behavior_v2.py`
(одни и те же при обучении и в конвейере), модель — `core/models/behavior_v1_open.onnx`, отчёт —
`core/models/behavior_v1_open.report.md`.

## Данные

Готовые скелеты (17 точек COCO, HRNet) из MMAction2 / PYSKL:
- `hmdb51_2d.pkl` — https://download.openmmlab.com/mmaction/v1.0/skeleton/data/hmdb51_2d.pkl
- `ntu60_2d.pkl` — https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl

Видео, скелеты из которых извлекает наша YOLO-pose: KTH (`walking.zip`, `jogging.zip`, `running.zip`,
https://www.csc.kth.se/cvap/actions/). Поддержаны и UR Fall, CAUCAFall (`extract_pose.py --dataset`).

## Обучение

```bash
# скелеты из видео KTH (вход 320 для роликов 160×120; прерванный запуск продолжается с места остановки)
python ml/behavior/extract_pose.py --dataset kth --videos datasets/kth --out datasets/kth_pose.pkl \
       --model core/models/yolov8n-pose.onnx --max-sec 0
# окна по 2 с: падение — только окна, где видно само падение; «норма» — бытовые действия
python ml/behavior/prepare.py --hmdb datasets/hmdb51_2d.pkl --out datasets/win_hmdb.npz
python ml/behavior/prepare.py --ntu datasets/ntu60_2d.pkl --ntu-normal-frac 0.25 --out datasets/win_ntu.npz
python ml/behavior/prepare.py --custom datasets/kth_pose.pkl --out datasets/win_kth.npz
# обучение (~7 мин на CPU) и выгрузка ONNX с классами и версией признаков в метаданных
python ml/behavior/train.py --data datasets/win_hmdb.npz datasets/win_kth.npz datasets/win_ntu.npz \
       --out core/models/behavior_v1_open.onnx --epochs 15
```

## Проверка на роликах команды

```bash
python -m core.tools.eval_clips --clips clips/clips.yaml --out out/beh --cache out/cache \
       --behavior core/models/behavior_v1_open.onnx
python -m core.tools.evaluate --labels eval/labels.csv --events out/beh/events.jsonl
```

Какие классы карточки берут у модели — `models.behavior_classes` в `core/cameras.yaml` (сейчас `[run]`).
`team_windows.py` — окна из роликов команды для дообучения на своём ракурсе с перекрёстной проверкой по фолдам.
