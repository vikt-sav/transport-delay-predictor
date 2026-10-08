"""Measures online inference latency (target: well under 1-2 s per vehicle stream)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from mtp.online import OnlinePipeline

p = OnlinePipeline(
    "dataset/validate/schedule_plan.csv",
    "data/gt/models/catboost_final.cbm",
    priors_path="data/gt/stop_priors.parquet",
)
va = pd.read_csv(
    "dataset/validate/traffic.csv",
    usecols=["tr_id", "event_time", "location_valid", "lat", "lon", "speed", "heading"],
    parse_dates=["event_time"],
    low_memory=False,
)
va["ts"] = va["event_time"].astype("datetime64[ns]").astype("int64") // 10**9
va = va.sort_values("event_time")
t0 = va.ts.min()
sub = va[(va.ts >= t0 + 12300) & (va.ts <= t0 + 14700) & (va.location_valid == True) & va.lat.notna()]
for r in sub.itertuples(index=False):
    p.add_record(
        int(r.tr_id), float(r.ts), float(r.lat), float(r.lon), True,
        0.0 if pd.isna(r.speed) else float(r.speed),
        0.0 if pd.isna(r.heading) else float(r.heading),
    )

lat_pred, lat_tick = [], []
for _ in range(30):
    t1 = time.perf_counter()
    p.tick()
    lat_tick.append((time.perf_counter() - t1) * 1000)
for tr in p.vehicles:
    t1 = time.perf_counter()
    p.predict_vehicle(tr, max(r[0] for v in p.vehicles.values() for r in v.records))
    lat_pred.append((time.perf_counter() - t1) * 1000)

print(f"tick (full fleet re-predict): mean {np.mean(lat_tick):.1f} ms, p95 {np.percentile(lat_tick, 95):.1f} ms")
print(f"per-vehicle predict:          mean {np.mean(lat_pred):.1f} ms, p95 {np.percentile(lat_pred, 95):.1f} ms")
print(f"snapshot vehicles: {len(p.last_snapshot)}")
