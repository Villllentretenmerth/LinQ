"""Качество по событиям: сравнение тревог конвейера с разметкой labels.csv.

labels.csv (разделитель «;», время — секунды от начала ролика):
    file;start;end;type;comment
    01_fall_1.mp4;3.0;19.5;fall;упал на бок
    04_trap_squat.mp4;;;none;присел — тревоги быть не должно

type — id карточки (fall, danger_zone, run, crawl_under, carry_wrong…) или по-русски:
падение, зона, бег, пролезание, подъём спиной; «none» / «нет» / «ловушка» — ролик без нарушений.

Тревога засчитывается (TP), если она того же типа, в том же ролике и подтверждена в интервале
[start − 2 с; end + 3 с]. Второй и последующие тревоги на тот же интервал — дубли (они нарушают
«одно нарушение — одно оповещение»). Тревога без подходящего интервала — ложная (FP).

    python -m core.tools.evaluate --labels clips/labels.csv --events out/eval/events.jsonl --report out/eval/quality.md
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict

ALIASES = {
    "падение": "fall", "упал": "fall", "зона": "danger_zone", "опасная зона": "danger_zone",
    "бег": "run", "пролезание": "crawl_under", "подъём спиной": "carry_wrong", "подъем спиной": "carry_wrong",
    "перелезание": "climb_fence", "телефон": "phone", "без каски": "no_helmet", "драка": "fight",
    "нет": "none", "ловушка": "none", "": "none",
}
BEFORE, AFTER = 2.0, 3.0


def _num(x):
    x = (x or "").strip().replace(",", ".")
    return float(x) if x else None


def load_labels(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        sample = f.read(2048)
        f.seek(0)
        delim = ";" if sample.count(";") >= sample.count(",") else ","
        rows = []
        for r in csv.DictReader(f, delimiter=delim):
            t = (r.get("type") or "").strip().lower()
            rows.append({"file": r["file"].strip(), "start": _num(r.get("start")), "end": _num(r.get("end")),
                         "type": ALIASES.get(t, t), "comment": (r.get("comment") or "").strip()})
    return rows


def load_events(path: str) -> list[dict]:
    """Тревоги из events.jsonl. В полном журнале прогона есть и update (второй этап) и end — их не считаем."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if r.get("kind", "create") == "create"]


def pct(v, q):
    if not v:
        return None
    s = sorted(v)
    i = (len(s) - 1) * q
    lo = int(i)
    return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (i - lo)


def evaluate(labels: list[dict], events: list[dict]) -> dict:
    gts = [g for g in labels if g["type"] != "none" and g["start"] is not None]
    trap_files = {g["file"] for g in labels if g["type"] == "none"}
    files = {g["file"] for g in labels}
    matched: dict[int, list[dict]] = defaultdict(list)
    fps = []
    for e in sorted(events, key=lambda e: e["confirm"]):
        if e["file"] not in files:
            continue                                  # ролик не размечен — не оцениваем
        cand = [i for i, g in enumerate(gts) if g["file"] == e["file"] and g["type"] == e["pattern_id"]
                and g["start"] - BEFORE <= e["confirm"] <= (g["end"] if g["end"] is not None else g["start"] + 30) + AFTER]
        if cand:
            matched[min(cand, key=lambda i: abs(gts[i]["start"] - e["confirm"]))].append(e)
        else:
            fps.append(e)

    types = sorted({g["type"] for g in gts} | {e["pattern_id"] for e in events if e["file"] in files})
    per_type = {}
    for t in types:
        g_idx = [i for i, g in enumerate(gts) if g["type"] == t]
        tp = sum(1 for i in g_idx if matched.get(i))
        fn = len(g_idx) - tp
        fp = sum(1 for e in fps if e["pattern_id"] == t)
        dups = sum(max(0, len(matched.get(i, [])) - 1) for i in g_idx)
        tta = [matched[i][0]["confirm"] - gts[i]["start"] for i in g_idx if matched.get(i)]
        per_type[t] = {
            "incidents": len(g_idx), "tp": tp, "fn": fn, "fp": fp, "duplicates": dups,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / len(g_idx) if g_idx else None,
            "alerts_per_incident": (tp + dups) / tp if tp else None,
            "tta_median_s": statistics.median(tta) if tta else None, "tta_p95_s": pct(tta, 0.95),
        }
    tp = sum(v["tp"] for v in per_type.values())
    fp = len(fps)
    dups = sum(v["duplicates"] for v in per_type.values())
    tta_all = [matched[i][0]["confirm"] - gts[i]["start"] for i in matched if matched[i]]
    trap_alerts = [e for e in events if e["file"] in trap_files]
    overall = {
        "incidents": len(gts), "tp": tp, "fn": len(gts) - tp, "fp": fp, "duplicates": dups,
        "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / len(gts) if gts else None,
        "alerts_per_incident": (tp + dups) / tp if tp else None,
        "tta_median_s": statistics.median(tta_all) if tta_all else None, "tta_p95_s": pct(tta_all, 0.95),
        "trap_clips": len(trap_files), "trap_alerts": len(trap_alerts),
    }
    return {"overall": overall, "per_type": per_type,
            "false_alarms": [{k: e[k] for k in ("file", "pattern_id", "confirm")} for e in fps],
            "missed": [g for i, g in enumerate(gts) if not matched.get(i)]}


def fmt(x, pctg=False, sec=False):
    if x is None:
        return "—"
    if pctg:
        return f"{x * 100:.0f}%"
    if sec:
        return f"{x:.1f} с".replace(".", ",")
    return f"{x:.2f}".replace(".", ",") if isinstance(x, float) else str(x)


def report(res: dict) -> str:
    o = res["overall"]
    lines = ["# Качество по событиям", "",
             f"Инцидентов в разметке: **{o['incidents']}**, найдено: **{o['tp']}**, пропущено: {o['fn']}, "
             f"ложных тревог: {o['fp']}, дублей: {o['duplicates']}.", "",
             "| Нарушение | Инцидентов | TP | FN | FP | Precision | Recall | Оповещений на инцидент | До тревоги, медиана | p95 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for t, v in res["per_type"].items():
        lines.append(f"| {t} | {v['incidents']} | {v['tp']} | {v['fn']} | {v['fp']} | {fmt(v['precision'], True)} | "
                     f"{fmt(v['recall'], True)} | {fmt(v['alerts_per_incident'])} | {fmt(v['tta_median_s'], sec=True)} | "
                     f"{fmt(v['tta_p95_s'], sec=True)} |")
    lines.append(f"| **Итого** | {o['incidents']} | {o['tp']} | {o['fn']} | {o['fp']} | **{fmt(o['precision'], True)}** | "
                 f"**{fmt(o['recall'], True)}** | {fmt(o['alerts_per_incident'])} | {fmt(o['tta_median_s'], sec=True)} | "
                 f"{fmt(o['tta_p95_s'], sec=True)} |")
    lines += ["", f"Ролики-ловушки (нарушений нет): {o['trap_clips']}, тревог на них: **{o['trap_alerts']}**."]
    if res["false_alarms"]:
        lines += ["", "**Ложные тревоги:**"] + [f"- {e['file']}: {e['pattern_id']} на {e['confirm']:.1f} с" for e in res["false_alarms"]]
    if res["missed"]:
        lines += ["", "**Пропущено:**"] + [f"- {g['file']}: {g['type']} {g['start']}–{g['end']} с {g['comment']}" for g in res["missed"]]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--report", default=None, help="сохранить отчёт в Markdown")
    ap.add_argument("--json", default=None, help="сохранить метрики в JSON")
    a = ap.parse_args(argv)
    res = evaluate(load_labels(a.labels), load_events(a.events))
    text = report(res)
    print(text)
    if a.report:
        open(a.report, "w", encoding="utf-8").write(text + "\n")
    if a.json:
        json.dump(res, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
