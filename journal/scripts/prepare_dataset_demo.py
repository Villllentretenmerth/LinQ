"""Build 20 demo videos from UR Fall and KTH. Requires core dependencies.
Run from journal: python -m scripts.prepare_dataset_demo --cache /tmp/linq-datasets
"""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
import imageio_ffmpeg
import requests

ROOT = Path(__file__).resolve().parents[1] / 'web' / 'dataset-media'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    def download(name, url):
        target = args.cache / name
        if not target.exists():
            response = requests.get(url, timeout=180)
            response.raise_for_status()
            target.write_bytes(response.content)
        return target
    archive = download('running.zip', 'https://www.csc.kth.se/cvap/actions/running.zip')
    entries = []
    with zipfile.ZipFile(archive) as running:
        for kind in ('fall', 'run'):
            for n in range(1, 11):
                key = f'{kind}-{n:02d}'
                if kind == 'fall':
                    filename = f'fall-{n:02d}-cam0.mp4'
                    url = f'https://fenix.ur.edu.pl/~mkepski/ds/data/{filename}'
                    raw = download(f'fall-{n:02d}.mp4', url)
                    source = 'https://fenix.ur.edu.pl/~mkepski/ds/uf.html'
                    dataset, license_ = 'UR Fall Detection', 'CC BY-NC-SA 4.0'
                    authors = 'Bogdan Kwolek, Michal Kepski (2014)'
                else:
                    filename = f'person{n:02d}_running_d1_uncomp.avi'
                    raw = args.cache / filename
                    raw.write_bytes(running.read(filename))
                    url = 'https://www.csc.kth.se/cvap/actions/running.zip'
                    source = 'https://www.csc.kth.se/cvap/actions/'
                    dataset, license_ = 'KTH Actions', 'Non-commercial use; see source terms'
                    authors = 'Christian Schuldt, Ivan Laptev, Barbara Caputo (ICPR 2004)'
                video = ROOT / f'{key}.mp4'
                subprocess.run([ffmpeg, '-v', 'error', '-y', '-i', str(raw), '-an',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', '-movflags', '+faststart', str(video)], check=True)
                frames = imageio_ffmpeg.read_frames(str(video))
                meta = next(frames); next(frames); frames.close()
                snapshot_time = .65 if kind == 'run' else meta['duration'] * .7
                subprocess.run([ffmpeg, '-v', 'error', '-y', '-ss', str(snapshot_time),
                    '-i', str(video), '-frames:v', '1', str(ROOT / f'{key}.jpg')], check=True)
                entries.append(dict(key=key, pattern_id=kind,
                    name=f'{"Падение" if kind == "fall" else "Бег"} — {dataset}, запись {n:02d}',
                    dataset=dataset, source_url=source, download_url=url, source_file=filename,
                    authors=authors, license=license_, duration_sec=meta['duration'],
                    sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                    description='Учебная запись из датасета, не результат работы детектора. '
                    + ('Бег сам по себе не доказывает нарушение требований безопасности.' if kind == 'run'
                       else 'Постановочное падение для проверки интерфейса.')))
                print(key, meta['duration'], flush=True)
    (ROOT / 'manifest.json').write_text(json.dumps(entries, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
