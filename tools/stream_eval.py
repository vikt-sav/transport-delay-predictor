"""Streaming verification: replay test traffic through the ONLINE pipeline and
score predictions against labels_test. This is the honest streaming evaluation:
no ground-truth hints are available at predict time."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mtp.gt import load_points, load_traffic
from mtp.online import OnlinePipeline


def run(args) -> None:
    base = Path("dataset")
    traffic = load_traffic(base / args.split / "traffic.csv")
    labels = load_points(base / "labels" / f"labels_{args.split}.csv")
    sched_name = "schedule.csv" if args.split != "validate" else "schedule_plan.csv"
    sched_path = base / args.split / sched_name

    pipeline = OnlinePipeline(
        str(sched_path), args.model,
        priors_path="data/gt/stop_priors.parquet",
        use_hmm_matching=args.hmm,
    )
    if args.use_est_dev:
        pipeline.use_est_dev_as_cur = True

    traffic = traffic.sort_values("event_time")
    t0 = int(traffic["ts"].min())

    preds = {}
    t_end = int(traffic["ts"].max())
    rows = list(traffic.itertuples(index=False))
    i = 0
    n = len(rows)
    label_ts = sorted({int(v) for v in labels["T_s"].unique()})
    label_rows = {}
    for lb in labels.itertuples(index=False):
        label_rows.setdefault(int(lb.T_s), []).append(lb)
    started = time.perf_counter()
    max_lag = 0.0
    for T in label_ts:
        if t0 > T or t_end < T:
            continue
        while i < n and rows[i].ts <= T:
            r = rows[i]
            pipeline.add_record(
                int(r.tr_id), float(r.ts),
                0.0 if pd.isna(r.lat) else float(r.lat),
                0.0 if pd.isna(r.lon) else float(r.lon),
                bool(r.location_valid),
                0.0 if pd.isna(r.speed) else float(r.speed),
                0.0 if pd.isna(r.heading) else float(r.heading),
            )
            i += 1
        t1 = time.perf_counter()
        for lb in label_rows.get(T, []):
            row = pipeline.predict_vehicle(int(lb.tr_id), T)
            if row is not None:
                preds[(int(lb.tr_id), T)] = row
        max_lag = max(max_lag, (time.perf_counter() - t1) * 1000)
    elapsed = time.perf_counter() - started

    y_true, y_pred = [], []
    matched, mismatched = 0, 0
    for lb in labels.itertuples(index=False):
        key = (int(lb.tr_id), int(lb.T_s))
        if key not in preds:
            continue
        row = preds[key]
        if int(row["stop_id"]) == int(lb.target_stop_id):
            matched += 1
        else:
            mismatched += 1
            continue
        y_true.append(float(lb.target_delay_s))
        y_pred.append(float(row["pred_s"]))

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")
    print(f"[stream-eval] split={args.split} scored points: {len(y_true)} (stop matched {matched}, mismatched skipped {mismatched})")
    print(f"[stream-eval] STREAMING MAE = {mae:.1f} s  (offline model MAE = 40.9)")
    print(f"[stream-eval] tick latency: mean {max_lag:.0f} ms worst; wall {elapsed:.0f} s for {len(label_ts)} ticks")
    if len(y_true):
        print(f"[stream-eval] mae_zero = {np.mean(np.abs(y_true)):.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--model", default="data/gt/models/catboost_final.cbm")
    ap.add_argument("--use-est-dev", action="store_true")
    ap.add_argument("--hmm", action="store_true", help="enable HMM street matching of telemetry (Stage 2)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
