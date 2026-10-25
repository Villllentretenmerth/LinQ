"""One isolated uploaded-video job; no event is created unless the engine detects it."""
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

# RTSP low-latency flags can prevent decoding short local files.
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = ''
import cv2
from .pipeline import CameraPipeline, Models, load_patterns
from .ingest import offline_frames
from .sink import JournalSink

ROOT = Path(__file__).resolve().parents[1]

def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False))
    tmp.replace(path)


def run(job_dir):
    job = json.loads((job_dir / 'request.json').read_text())
    cfg, camera = job['config'], job['camera']
    source = job_dir / job['source']
    cap = cv2.VideoCapture(str(source))
    try:
        if not cap.isOpened():
            raise ValueError('Не удалось открыть видео. Выберите MP4, MOV, AVI, MKV или WebM с поддерживаемым кодеком.')
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or count <= 0 or count / fps > 600:
            raise ValueError('Нужно видео продолжительностью до 10 минут с корректной длительностью.')
        if cap.get(cv2.CAP_PROP_FRAME_WIDTH) * cap.get(cv2.CAP_PROP_FRAME_HEIGHT) > 3840 * 2160:
            raise ValueError('Разрешение видео должно быть не больше 4K.')
        if not cap.read()[0]:
            raise ValueError('Не удалось прочитать кадры видео.')
        duration = count / fps
    finally:
        cap.release()

    class UploadSink(JournalSink):
        def create(self, key, payload, snapshot=None):
            payload['extra'] = {**payload.get('extra', {}), 'upload_id': job_dir.name,
                                'uploaded_filename': job['filename'], 'source': 'uploaded_video'}
            super().create(key, payload, snapshot)

    models = Models(cfg, ROOT)
    sink = UploadSink(job['journal_url'], job_dir / 'evidence', retries=2)
    pipeline = None
    first = last = None
    try:
        camera = {**camera, 'id': f'upload-{job_dir.name}', 'name': job['filename'], 'source': str(source), 'loop': False}
        pipeline = CameraPipeline(camera, cfg, models, load_patterns(cfg, ROOT), sink, None, True, threading.Event(), ROOT)
        if not pipeline.engine.cards:
            raise ValueError('Нет доступных сценариев проверки для выбранных моделей и настроек.')
        sink.register_camera(camera['id'], camera['name'], pipeline.area, 'uploaded video')
        progress_at = 0
        for frame in offline_frames(str(source), pipeline.fps):
            if first is None:
                first = frame.ts
            pipeline.process(frame)
            last = frame
            if time.monotonic() - progress_at > 1:
                save(job_dir / 'progress.json', {'progress': min(99, round((frame.ts-first)/duration*100)), 'frames': pipeline.n})
                progress_at = time.monotonic()
        if last is None:
            raise ValueError('Видео не содержит читаемых кадров.')
        for action in pipeline.engine.close(last.ts):
            pipeline.emit(action, last, None)
    finally:
        if pipeline and pipeline.clips:
            pipeline.clips.close()
        sink.close()
    if sink.failed:
        raise ValueError('Обработка выполнена, но часть результатов не дошла до журнала. Проверьте подключение к серверу.')
    save(job_dir / 'result.json', {'events': pipeline.events, 'event_ids': list(sink.ids.values()),
         'frames': pipeline.n, 'progress': 100, 'patterns': [c.name for c in pipeline.engine.cards]})

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    folder = Path(sys.argv[1])
    try:
        run(folder)
    except Exception as exc:
        logging.exception('Upload processing failed')
        save(folder / 'error.json', {'message': str(exc) if isinstance(exc, ValueError)
             else 'Не удалось обработать видео. Проверьте модели и настройки обработчика на сервере.'})
        raise SystemExit(1)
