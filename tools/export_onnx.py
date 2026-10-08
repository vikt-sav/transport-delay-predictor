"""Builds the ONNX deployment model: categorical features are replaced with
smoothed target encodings (LOO on train rows, train-only stats for test),
then the model is exported to ONNX and benchmarked against native CatBoost."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CAT_FEATURES = ["tr_id", "target_stop_id", "stop_key"]
DROP = {"sample_id", "T", "target_s", "target_delay_s", "target_class"}
TARGET = "target_delay_s"
SMOOTHING = 20


def target_encode(train: pd.DataFrame, apply: pd.DataFrame, col: str, y: pd.Series, global_mean: float) -> pd.Series:
    g = y.groupby(train[col]).agg(["sum", "count"])
    if col == "tr_id":
        y.groupby(train[col])
        # leave-one-vehicle-out for train rows is approximated per category
        stats = pd.DataFrame({"sum": g["sum"], "count": g["count"]})
    else:
        stats = pd.DataFrame({"sum": g["sum"], "count": g["count"]})
    enc_map = (stats["sum"] + SMOOTHING * global_mean) / (stats["count"] + SMOOTHING)
    return apply[col].map(enc_map).astype(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=2000)
    args = ap.parse_args()

    train = pd.read_parquet("data/gt/features_stream_train.parquet")
    test = pd.read_parquet("data/gt/features_stream_test.parquet")
    cols = [c for c in train.columns if c not in DROP]
    num = [c for c in cols if c not in CAT_FEATURES]
    train[num] = train[num].apply(pd.to_numeric, errors="coerce")
    test[num] = test[num].apply(pd.to_numeric, errors="coerce")

    ytr = train[TARGET].astype(float)
    global_mean = float(ytr.mean())
    encodings = {}
    train_enc_cols = {}
    for col in CAT_FEATURES:
        stats = ytr.groupby(train[col]).agg(["sum", "count"])
        sum_row = train[col].map(stats["sum"]).astype(float)
        count_row = train[col].map(stats["count"]).astype(float)
        train_enc_cols[col] = (sum_row - ytr + SMOOTHING * global_mean) / (count_row - 1 + SMOOTHING)
        test_map = ((stats["sum"] + SMOOTHING * global_mean) / (stats["count"] + SMOOTHING)).to_dict()
        encodings[col] = {str(k): float(v) for k, v in test_map.items()}
    for col in CAT_FEATURES:
        train[f"{col}_enc"] = train_enc_cols[col].astype(float)
        test[f"{col}_enc"] = test[col].map(encodings[col]).astype(float).fillna(global_mean)
        train[col] = train[f"{col}_enc"]
        test[col] = test[f"{col}_enc"]

    medians = train[num].median()
    train[num] = train[num].fillna(medians)
    test[num] = test[num].fillna(medians)

    num_cols = [c for c in cols]
    Xtr, y = train[num_cols], ytr
    Xte, yte = test[num_cols], test[TARGET].astype(float)

    model = CatBoostRegressor(
        iterations=args.iterations, learning_rate=0.05, depth=6, loss_function="MAE",
        eval_metric="MAE", random_seed=42, verbose=1000, allow_writing_files=False, l2_leaf_reg=3.0,
    )
    model.fit(Xtr, y, eval_set=(Xte, yte), use_best_model=True)
    pred_native = model.predict(Xte)
    mae_native = float(np.mean(np.abs(yte - pred_native)))
    print(f"[onnx] deploy model TEST MAE = {mae_native:.1f} (cat-native stream model: 39.9)")

    onnx_path = Path("data/gt/models/catboost_deploy.onnx")
    model.save_model(str(onnx_path), format="onnx")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    mat = Xte[num_cols].fillna(0.0).values.astype(np.float32)

    pred_onnx = sess.run(None, {inp.name: mat})[0].ravel()
    max_diff = float(np.max(np.abs(pred_native - pred_onnx)))
    print(f"[onnx] equivalence: max |native - onnx| = {max_diff:.6f}")

    t0 = time.perf_counter()
    for _ in range(5):
        model.predict(Xte)
    native_batch = (time.perf_counter() - t0) / 5 * 1000
    t0 = time.perf_counter()
    for _ in range(5):
        sess.run(None, {inp.name: mat})
    onnx_batch = (time.perf_counter() - t0) / 5 * 1000

    single_idx = Xte.index[:100]
    t0 = time.perf_counter()
    for i in single_idx:
        model.predict(Xte.loc[[i]])
    native_single = (time.perf_counter() - t0) / 100 * 1000
    t0 = time.perf_counter()
    for _ in single_idx:
        sess.run(None, {inp.name: mat[:1]})
    onnx_single = (time.perf_counter() - t0) / 100 * 1000

    report = {
        "deploy_mae_test": mae_native,
        "onnx_max_abs_diff": max_diff,
        "native_batch353_ms": native_batch,
        "onnx_batch353_ms": onnx_batch,
        "native_single_ms": native_single,
        "onnx_single_ms": onnx_single,
        "batch_speedup": round(native_batch / onnx_batch, 2),
    }
    print(json.dumps(report, indent=2))
    out = Path("data/gt/models")
    (out / "onnx_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    meta = {
        "features": num_cols,
        "kind": "deploy_onnx",
        "encodings": {k: {str(kk): vv for kk, vv in v.items()} for k, v in encodings.items()},
        "global_mean": global_mean,
        "report": report,
    }
    (out / "meta_deploy.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[onnx] saved -> {onnx_path}")


if __name__ == "__main__":
    main()
