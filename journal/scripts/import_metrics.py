"""Импорт результата замера из Jupyter (bench_result.json) в журнал.

    python -m scripts.import_metrics bench_result.json --url http://localhost:8000
"""
import argparse
import json

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--url", default="http://localhost:8000")
    args = ap.parse_args()
    data = json.load(open(args.path, encoding="utf-8"))
    metric = data.get("metric", data)
    r = requests.post(args.url.rstrip("/") + "/api/metrics", json=metric, timeout=10)
    r.raise_for_status()
    print("Загружено:", r.json())


if __name__ == "__main__":
    main()
