"""Квантование и компиляция голов YOLO в программы LinQ TPU (.tpu) — тот же путь, что compile.py из SDK,
но с калибровкой на реальных кадрах (core/tools/make_calib.py), а не на случайном шуме.

SDK — только Linux x86_64 + Python 3.12, поэтому запускается в контейнере (см. core/tools/linq_sdk.Dockerfile):

    docker build -t team94-linq-sdk -f core/tools/linq_sdk.Dockerfile <папка tpu_framework из SDK>
    docker run --rm -v "<репозиторий>:/repo" -w /repo team94-linq-sdk \
        python core/tools/compile_tpu.py core/models/yolov8n-pose_head.onnx core/models/yolov8n_head.onnx \
        --calib calib.npy --out core/models/tpu

Для каждой модели: оптимизация графа → калибровка порогов int8 → квантование → компиляция (O5, при ошибке O1).
--verify N сравнивает квантованную модель (эмуляция int8 на CPU средствами SDK) с исходной ONNX на N кадрах:
так видно, сколько точности съел int8, ещё до TPU. Файлы: <out>/<модель>_b1_o5_h1m.tpu и рядом отчёт .json.
Скрипт самодостаточный (без пакета core): в контейнере только SDK, numpy и onnx.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import onnx


def log(msg: str) -> None:
    print(msg, flush=True)


def prepare(onnx_path: str, work: str) -> tuple[str, dict]:
    """Оптимизация графа SDK (как в compile.py): убрать инициализаторы из входов, проверить статические формы."""
    from tpu_framework import preliminary_model_conditioning

    model = onnx.load(onnx_path)
    inits = {i.name for i in model.graph.initializer}
    for i in [i for i in model.graph.input if i.name in inits]:
        model.graph.input.remove(i)
    inputs = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in model.graph.input}
    for name, shape in inputs.items():
        if 0 in shape:
            raise SystemExit(f"{onnx_path}: вход {name} с динамической формой {shape} — выгрузите со статическим 640")
    opt_path = os.path.join(work, "opt.onnx")
    onnx.save(preliminary_model_conditioning(model), opt_path)
    return opt_path, inputs


def calibrate(opt_path: str, inputs: dict, calib: np.ndarray, work: str, samplewise: float | None):
    from tpu_framework import RegularModel

    (name, shape), = inputs.items()
    if tuple(calib.shape[1:]) != tuple(shape[1:]):
        raise SystemExit(f"калибровка {calib.shape} не подходит ко входу {name} {shape}")
    regular = RegularModel(model=onnx.load(opt_path))
    mapping = getattr(regular, "anchors_mapping", None) or {}
    data = {mapping.get(name, name): calib}
    kw = {"samplewise_percentile": samplewise} if samplewise is not None else {}
    thresholds = regular.calibrate_torch(calibration_data=data, **kw)
    RegularModel.dump_thresholds(os.path.join(work, "thresholds.json"), thresholds)
    return regular.quantize(thresholds)


def verify(quant, onnx_path: str, inputs: dict, samples: np.ndarray) -> dict:
    """Квантованная модель на CPU (as_torch_graph, как в End-To-End Test SDK) против исходной ONNX:
    по каждому выходу — косинусная близость и относительная ошибка. Выходы сопоставляются по форме и имени."""
    import onnxruntime as ort
    import torch
    try:
        from tpu_framework import TpuDtype
    except ImportError:          # в SDK 2026.9 он только в dnn_quant (документация импортирует из tpu_framework)
        from dnn_quant.api import TpuDtype

    # эталон — исходная ONNX: в оптимизированном графе SDK свои операции (custom_op.tpu:swish), ORT их не знает
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    (in_name, _), = inputs.items()
    g = quant.as_torch_graph(batch_size=1)
    anchors = {**g.get_outputs_anchors(), **g.get_inputs_anchors()}   # исходное имя -> имя в графе
    to_orig = {v: k for k, v in anchors.items()}
    gm = g.gm.eval()
    stats: dict[str, list] = {n: [] for n in out_names}
    for x in samples:
        x = x[None]
        ref = dict(zip(out_names, sess.run(None, {in_name: x})))
        q_in = quant.rescale({anchors.get(in_name, in_name): x}, TpuDtype.Input)
        with torch.no_grad():
            outs = gm(*[torch.as_tensor(q_in[n]) for n in g._input_layer_names])
        outs = outs if isinstance(outs, (tuple, list)) else (outs,)
        out_layer = list(getattr(g, "_output_layer_names", []) or [])
        raw = {out_layer[i] if i < len(out_layer) else f"out{i}": o.detach().cpu().numpy() for i, o in enumerate(outs)}
        deq = quant.rescale(raw, TpuDtype.Output)
        free = dict(ref)
        for k, v in deq.items():
            key = to_orig.get(k, k)
            if key not in free:   # имя не сопоставилось — ищем выход ONNX той же формы
                key = next((n for n, r in free.items() if r.size == v.size), None)
            if key is None:
                continue
            r = free.pop(key).ravel().astype(np.float64)
            q = np.asarray(v).ravel().astype(np.float64)
            cos = float(r @ q / (np.linalg.norm(r) * np.linalg.norm(q) + 1e-12))
            rel = float(np.abs(r - q).mean() / (np.abs(r).mean() + 1e-12))
            stats[key].append((cos, rel))
    return {n: {"cosine_mean": round(float(np.mean([c for c, _ in v])), 5),
                "rel_err_mean": round(float(np.mean([e for _, e in v])), 5)} for n, v in stats.items() if v}


def compile_program(quant, preset_name: str, platform_hw, path: str) -> None:
    from tpu_framework import O1, O5, Network, TpuProgram, compiler

    network, _ = Network.from_quantized_model(quant)
    executable, tensors = compiler.compile_(hardware_parameters=platform_hw, network=network,
                                            parameters={"o5": O5, "o1": O1}[preset_name])
    TpuProgram.from_executable(executable, tensors).to_file(path)


def run_one(onnx_path: str, calib: np.ndarray, a) -> dict:
    from tpu_framework import TPU_H1M, QuantizedModel

    name = os.path.splitext(os.path.basename(onnx_path))[0]
    work = os.path.join(a.out, name)
    os.makedirs(work, exist_ok=True)
    rep: dict = {"model": onnx_path, "calib_frames": int(len(calib))}
    t0 = time.time()

    log(f"\n=== {name}: 1/3 оптимизация графа")
    opt_path, inputs = prepare(onnx_path, work)
    rep["inputs"] = inputs

    log(f"=== {name}: 2/3 калибровка на {len(calib)} кадрах и квантование")
    quant = calibrate(opt_path, inputs, calib, work, a.samplewise_percentile)
    qm_name = f"quant_{name}.qm.onnx"
    quant.save(file_dir=work, file_name=qm_name)
    rep["quantize_sec"] = round(time.time() - t0, 1)

    if a.verify:
        log(f"=== {name}: проверка int8 против ONNX на {min(a.verify, len(calib))} кадрах")
        try:
            rep["verify"] = verify(quant, onnx_path, inputs, calib[:a.verify])
            for out, s in rep["verify"].items():
                log(f"    {out}: косинус {s['cosine_mean']:.4f}, относительная ошибка {s['rel_err_mean']:.3%}")
        except Exception as e:   # проверка — подсказка, не повод останавливать компиляцию
            rep["verify_error"] = f"{type(e).__name__}: {e}"
            log(f"    проверка не удалась: {rep['verify_error']}")

    presets = [a.preset] if a.preset == "o1" or a.no_fallback else ["o5", "o1"]
    for preset in presets:
        log(f"=== {name}: 3/3 компиляция {preset.upper()}")
        path = os.path.join(a.out, f"{name}_b{inputs[next(iter(inputs))][0]}_{preset}_{a.platform}.tpu")
        try:
            compile_program(QuantizedModel.load(os.path.join(work, qm_name)), preset, TPU_H1M, path)
            rep.update(program=path, preset=preset, size_mb=round(os.path.getsize(path) / 2**20, 1))
            break
        except Exception as e:
            rep[f"compile_error_{preset}"] = f"{type(e).__name__}: {e}"
            log(f"    {preset.upper()} не скомпилировалась: {e}")
            if preset == presets[-1]:
                traceback.print_exc()
    rep["total_sec"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, f"{name}.report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="ONNX-модели (головы из core/tools/split_onnx.py)")
    ap.add_argument("--calib", required=True, help="calib.npy из core/tools/make_calib.py, (N, 3, 640, 640)")
    ap.add_argument("--calib-n", type=int, default=0, help="взять только первые N кадров калибровки")
    ap.add_argument("--out", default="core/models/tpu")
    ap.add_argument("--preset", choices=["o5", "o1"], default="o5", help="O5 быстрее; если не собирается — O1")
    ap.add_argument("--no-fallback", action="store_true", help="не пробовать O1 после неудачи O5")
    ap.add_argument("--platform", default="h1m")
    ap.add_argument("--samplewise-percentile", type=float, default=None,
                    help="пороги по перцентилю покадровых максимумов (например 99.9) — устойчивее к выбросам")
    ap.add_argument("--verify", type=int, default=8, help="на скольких кадрах сравнить int8 с ONNX (0 — не сравнивать)")
    a = ap.parse_args()

    calib = np.load(a.calib).astype(np.float32)
    if a.calib_n:
        calib = calib[:a.calib_n]
    log(f"калибровка: {calib.shape}, диапазон {calib.min():.2f}..{calib.max():.2f}")
    os.makedirs(a.out, exist_ok=True)
    results = [run_one(m, calib, a) for m in a.models]
    log("\nИтог:")
    ok = True
    for r in results:
        if "program" in r:
            log(f"  {r['program']}  ({r['preset'].upper()}, {r['size_mb']} МБ, {r['total_sec']} с)")
        else:
            ok = False
            log(f"  {r['model']}: не скомпилировалась — см. {a.out}/*.report.json")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
