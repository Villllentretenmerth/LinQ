"""Bounded background upload processing. Run one API worker for the embedded queue."""
import asyncio
import copy
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = Path(os.getenv('UPLOAD_DIR', ROOT / 'journal' / 'uploads')).resolve()
CONFIG = Path(os.getenv('UPLOAD_CONFIG', ROOT / 'core' / 'cameras.yaml'))
JOURNAL_URL = os.getenv('UPLOAD_JOURNAL_URL', 'http://127.0.0.1:8000')
MAX_BYTES = 500 * 1024 * 1024
router = APIRouter(prefix='/api/uploads', tags=['Video uploads'])
manager = None


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False))
    temp.replace(path)


def config():
    if not CONFIG.is_file() or not (ROOT / 'core' / 'upload_video.py').is_file():
        raise HTTPException(503, 'Обработка видео не настроена на сервере. Установите core и укажите UPLOAD_CONFIG.')
    if any(importlib.util.find_spec(m) is None for m in ('cv2', 'onnxruntime', 'supervision', 'imageio_ffmpeg')):
        raise HTTPException(503, 'На сервере нужны зависимости core/requirements.txt для обработки видео.')
    cfg = yaml.safe_load(CONFIG.read_text())
    for name in ('pose', 'det'):
        path = (cfg.get('models') or {}).get(name)
        if path and cfg['models'].get('backend', 'onnx') in ('onnx', 'onnx_head'):
            path = Path(path)
            if not (path if path.is_absolute() else ROOT / path).is_file():
                raise HTTPException(503, 'Модели не найдены. Подготовьте модели по инструкции core/README.md.')
    return cfg


class Jobs:
    def __init__(self):
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(3)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='video-upload')
        self.process = None
        self.closed = False
        for path in UPLOAD_DIR.glob('*/status.json'):
            try:
                data = json.loads(path.read_text())
            except (ValueError, OSError):
                continue
            if data.get('state') in ('queued', 'processing'):
                save(path, {**data, 'state': 'failed', 'message': 'Сервер перезапущен. Загрузите видео повторно.'})

    def update(self, folder, **fields):
        with self.lock:
            path = folder / 'status.json'
            save(path, {**json.loads(path.read_text()), **fields})

    def run(self, folder):
        try:
            with self.lock:
                if self.closed:
                    return
            self.update(folder, state='processing', message='Видео обрабатывается…')
            with (folder / 'processing.log').open('w') as log:
                with self.lock:
                    if self.closed:
                        return
                    proc = subprocess.Popen([sys.executable, '-m', 'core.upload_video', str(folder)],
                        cwd=ROOT, stdout=log, stderr=log, env={**os.environ, 'OPENCV_FFMPEG_CAPTURE_OPTIONS': ''})
                    self.process = proc
                try:
                    code = proc.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
                    raise RuntimeError('Обработка превысила 30 минут. Попробуйте более короткое видео.')
                finally:
                    with self.lock:
                        self.process = None
            if code or not (folder / 'result.json').exists():
                error = folder / 'error.json'
                raise RuntimeError(json.loads(error.read_text())['message'] if error.exists() else 'Обработка прервана. Повторите загрузку.')
            result = json.loads((folder / 'result.json').read_text())
            self.update(folder, state='done', result=result,
                        message=f'Найдено событий: {result["events"]}.' if result['events'] else 'Нарушений не обнаружено.')
        except Exception as exc:
            self.update(folder, state='failed', message=str(exc))
        finally:
            self.slots.release()

    def close(self):
        with self.lock:
            self.closed = True
            if self.process:
                self.process.terminate()
        self.executor.shutdown(wait=False, cancel_futures=True)


@router.get('/options')
def options():
    try:
        cfg = config()
        return {'available': True, 'max_mb': MAX_BYTES // 1024**2,
                'profiles': [{'id': c['id'], 'name': c.get('name', c['id'])} for c in cfg.get('cameras', [])]}
    except HTTPException as exc:
        return {'available': False, 'message': exc.detail, 'profiles': []}


@router.post('', status_code=202)
async def upload(file: UploadFile = File(...), profile: str = Form('')):
    cfg = copy.deepcopy(config())
    # Keep batch inference from occupying all CPU threads used by the live service.
    if not cfg.setdefault('models', {}).get('threads'):
        cfg['models']['threads'] = 2
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in ('.mp4', '.mov', '.avi', '.mkv', '.webm'):
        await file.close()
        raise HTTPException(415, 'Выберите MP4, MOV, AVI, MKV или WebM.')
    if profile:
        camera = next((c for c in cfg.get('cameras', []) if c['id'] == profile), None)
        if camera is None:
            await file.close()
            raise HTTPException(422, 'Неизвестный профиль камеры.')
    else:
        camera = {'fps': 10, 'patterns': ['fall', 'carry_wrong'], 'area': 'Загруженные видео', 'zones': [], 'lines': []}
    # Uploaded recordings should be checked independently of the current wall-clock shift.
    camera['overrides'] = {**camera.get('overrides', {})}
    for entry in camera.get('patterns', []):
        pid = entry if isinstance(entry, str) else entry['id']
        camera['overrides'][pid] = {**camera['overrides'].get(pid, {}), 'schedule': 'always'}
    cfg['clips'] = {**cfg.get('clips', {}), 'enabled': True}
    if manager is None or manager.closed or not manager.slots.acquire(blocking=False):
        await file.close()
        raise HTTPException(429, 'Очередь заполнена. Дождитесь завершения обработки и повторите загрузку.')
    folder = UPLOAD_DIR / uuid.uuid4().hex
    queued = False
    try:
        folder.mkdir()
        target = folder / f'source{suffix}'
        size = 0
        with target.open('wb') as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, 'Максимальный размер видео — 500 МБ.')
                await asyncio.to_thread(output.write, chunk)
        if not size:
            raise HTTPException(422, 'Файл пустой.')
        filename = (file.filename or 'Видео').replace('\\', '/').split('/')[-1][:160]
        save(folder / 'request.json', {'config': cfg, 'camera': camera, 'filename': filename,
              'source': target.name, 'journal_url': JOURNAL_URL})
        data = {'id': folder.name, 'filename': filename, 'state': 'queued', 'message': 'Видео в очереди.'}
        save(folder / 'status.json', data)
        manager.executor.submit(manager.run, folder)
        queued = True
        return data
    finally:
        await file.close()
        if not queued:
            shutil.rmtree(folder, ignore_errors=True)
            manager.slots.release()


@router.get('/{job_id}')
def status(job_id: str):
    if not re.fullmatch('[a-f0-9]{32}', job_id):
        raise HTTPException(404, 'Загрузка не найдена.')
    folder = UPLOAD_DIR / job_id
    path = folder / 'status.json'
    if not path.is_file():
        raise HTTPException(404, 'Загрузка не найдена.')
    data = json.loads(path.read_text())
    progress = folder / 'progress.json'
    if data['state'] == 'processing' and progress.exists():
        data.update(json.loads(progress.read_text()))
    return data
