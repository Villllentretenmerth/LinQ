"""Split the labelled team CP3 montage; keep existing dataset examples."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1] / 'web' / 'dataset-media'
# Include each title, action and overlay. The final summary slide is not an event.
SCENES = [
 ('zone', 'danger_zone', 'Опасная зона: двое в кадре', 0, 10, 8, True),
 ('fall', 'fall', 'Падение и последующая неподвижность', 10, 19, 17, True),
 ('run', 'run', 'Бег в сторону камеры', 19, 24.8, 23.5, True),
 ('carry', 'carry_wrong', 'Подъём груза спиной', 24.8, 30.8, 29.5, True),
 ('safe-lift', 'carry_wrong', 'Контроль: правильный подъём — без нарушения', 30.8, 36.4, 34, False),
 ('fast-walk', 'run', 'Контроль: быстрый шаг — не бег', 36.4, 43, 40, False),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    args = parser.parse_args()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    manifest = ROOT / 'manifest.json'
    entries = json.loads(manifest.read_text())
    entries = [e for e in entries if e.get('source_group') != 'cp3']
    for label, pattern, title, start, end, shot, violation in SCENES:
        key = f'cp3-{label}'
        path = ROOT / f'{key}.mp4'
        subprocess.run([ffmpeg, '-v', 'error', '-y', '-ss', str(start), '-i', str(args.video),
            '-t', str(end-start), '-an', '-c:v', 'libx264', '-crf', '23', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(path)], check=True)
        subprocess.run([ffmpeg, '-v', 'error', '-y', '-ss', str(shot-start), '-i', str(path),
            '-frames:v', '1', str(ROOT / f'{key}.jpg')], check=True)
        frames = imageio_ffmpeg.read_frames(str(path)); meta = next(frames); next(frames); frames.close()
        entries.append(dict(key=key, pattern_id=pattern, name=f'CP3 · {title}', dataset='Демонстрация команды CP3',
            source_group='cp3', source_file='demo_cp3.mp4',
            source_url='https://github.com/Villllentretenmerth/LinQ/blob/main/docs/demo_cp3.mp4',
            authors='team_94', license='Видео команды', duration_sec=meta['duration'],
            source_start_sec=start, source_end_sec=end, expected_violation=violation,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            description=('Эпизод из демонстрации CP3 с исходными подписями. '
                + ('В исходном ролике показано срабатывание.' if violation else
                   'Безопасное действие: ожидается отсутствие тревоги. Карточка создана вручную для контрольной проверки, а не детектором.'))))
    manifest.write_text(json.dumps(entries, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
