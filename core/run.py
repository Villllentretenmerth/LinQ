"""Точка входа конвейера: видео / RTSP / папка кадров → нарушения → события в журнале.

    python -m core.run --config core/cameras.yaml --journal http://localhost:8000            # как камера
    python -m core.run --config core/cameras.yaml --journal http://localhost:8000 --offline  # покадрово
    python -m core.run --config core/cameras.yaml                                            # без журнала

Без --journal события пишутся в лог и в папку доказательств (кадры + events.jsonl).
"""
import argparse
import logging
import signal
import socket
import tempfile
import threading
import time
from pathlib import Path

import yaml

from .pipeline import CameraPipeline, Models, load_patterns
from .sink import JournalSink, MetricsReporter

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("core.run")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m core.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "core" / "cameras.yaml"))
    ap.add_argument("--journal", help="адрес журнала, например http://localhost:8000")
    ap.add_argument("--offline", action="store_true", help="видео и папки — покадрово, без пропусков и пауз")
    ap.add_argument("--camera", action="append", help="запустить только эти камеры (id), можно несколько раз")
    ap.add_argument("--source", help="подменить источник у камеры (удобно, если камера одна)")
    ap.add_argument("--ignore-schedule", action="store_true", help="все карточки активны круглосуточно (для показа)")
    ap.add_argument("--max-seconds", type=float, help="остановиться через N секунд видео")
    ap.add_argument("--evidence-dir", help="куда класть кадры-доказательства и events.jsonl")
    ap.add_argument("--no-metrics", action="store_true", help="не слать метрики узла в журнал (прогон ролика на ноутбуке для показа)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cameras = [c for c in cfg.get("cameras") or [] if not args.camera or c["id"] in args.camera]
    if not cameras:
        log.error("нет камер для запуска (проверьте --camera и %s)", args.config)
        return 2
    if args.source:
        cameras[0] = {**cameras[0], "source": args.source}
    if args.ignore_schedule:
        for c in cameras:
            ids = [p if isinstance(p, str) else p["id"] for p in c.get("patterns") or []]
            ov = dict(c.get("overrides") or {})
            for pid in ids:
                ov[pid] = {**(ov.get(pid) or {}), "schedule": "always"}
            c["overrides"] = ov

    models = Models(cfg, ROOT)
    cards = load_patterns(cfg, ROOT)
    log.info("карточек в patterns/: %d; бэкенд: %s", len(cards), models.backend)

    evidence = Path(args.evidence_dir) if args.evidence_dir else Path(tempfile.gettempdir()) / "team94-evidence"
    sink = JournalSink(args.journal, evidence)
    log.info("кадры-доказательства и events.jsonl: %s", evidence)

    rep = None
    if args.journal and not args.no_metrics:
        node = cfg.get("node") or {}
        rep = MetricsReporter(args.journal, node_id=node.get("id", socket.gethostname()),
                              node_name=node.get("name"), host=socket.gethostname(),
                              tpu_cores=int(node.get("tpu_cores", 1)), window_sec=float(node.get("metrics_window_sec", 5)),
                              tpu_models={"yolo-pose", "yolo-det"}, source="pipeline")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    pipes = [CameraPipeline(c, cfg, models, cards, sink, rep, args.offline, stop, ROOT, args.max_seconds)
             for c in cameras]
    threads = [threading.Thread(target=p.run, name=f"cam:{p.id}", daemon=True) for p in pipes]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop.set()
    for t in threads:
        t.join()  # camera shutdown drains clip encoders before closing the upload queue

    sink.close()
    if rep:
        rep.stop()          # последнее окно метрик
        if rep.last_error:
            log.warning("метрики не дошли до журнала: %s", rep.last_error)
    total = sum(p.events for p in pipes)
    log.info("готово: событий %d, отправлено в журнал %d, ошибок отправки %d", total, sink.sent, sink.failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
