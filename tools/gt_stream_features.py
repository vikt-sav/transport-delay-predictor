"""Recomputes stream-available deviation estimates for labeled points and
produces streaming-model feature tables where cur_dev_s is our own GPS-based
estimate (the ground-truth delay hint is unavailable in a live stream)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mtp.gt import (
    load_points,
    load_schedule,
    load_traffic,
    lonlat_to_xy,
    reconstruct_stop_matches,
)

RADIUS_M = 90.0
WINDOW_S = 300.0
PLAN_LEAD_S = 30.0


def estimate_series(veh_traffic, veh_sched):
    """Per stop (plan_s): matched delay estimate via the shared matcher."""
    veh_t = veh_traffic[(veh_traffic.location_valid == True) & veh_traffic.lat.notna()]
    veh_s = veh_sched.dropna(subset=["stop_lat", "stop_lon"]).sort_values("plan_s")
    if veh_t.empty or veh_s.empty:
        return pd.DataFrame(columns=["plan_s", "est"])
    lat0 = float(veh_s.stop_lat.median())
    lon0 = float(veh_s.stop_lon.median())
    gy, gx = lonlat_to_xy(veh_t.lat.values, veh_t.lon.values, lat0, lon0)
    sy, sx = lonlat_to_xy(veh_s.stop_lat.values, veh_s.stop_lon.values, lat0, lon0)
    matches = reconstruct_stop_matches(
        np.column_stack([gx, gy]),
        veh_t["ts"].values.astype(np.int64),
        np.column_stack([sx, sy]),
        veh_s["plan_s"].values.astype(np.int64),
        veh_s["tt_action_item_id"].values,
        radius_m=RADIUS_M,
        window_s=WINDOW_S,
    )
    rows = matches[matches["matched"] == 1][["plan_s", "est_delay_s"]]
    return rows.rename(columns={"est_delay_s": "est"})


def build_stream_features(split: str) -> pd.DataFrame:
    base = Path("dataset")
    traffic = load_traffic(base / split / "traffic.csv")
    sched = load_schedule(base / split / "schedule.csv")
    labels = load_points(base / "labels" / f"labels_{split}.csv")
    feats = pd.read_parquet(f"data/gt/features_{split}.parquet")

    est_by_point = {}
    for tr_id, grp in labels.groupby("tr_id"):
        veh_traffic = traffic[traffic.tr_id == tr_id]
        veh_sched = sched[sched.tr_id == tr_id]
        est_series = estimate_series(veh_traffic, veh_sched)
        for lb in grp.itertuples(index=False):
            T = int(lb.T_s)
            past = est_series[est_series["plan_s"] <= T - PLAN_LEAD_S]
            est_by_point[lb.sample_id] = (past["est"].tolist(), past["plan_s"].tolist())

    rows = []
    for f in feats.itertuples(index=False):
        T = int(pd.Timestamp(f.T).timestamp())
        ests, plans = est_by_point.get(f.sample_id, ([], []))
        row = {
            "sample_id": f.sample_id,
            "est_dev_last": ests[-1] if ests else np.nan,
            "est_dev_slope": ests[-1] - ests[-2] if len(ests) >= 2 else np.nan,
            "est_dev_mean3": float(np.mean(ests[-3:])) if ests else np.nan,
            "est_dev_std5": float(np.std(ests[-5:])) if len(ests) >= 2 else np.nan,
            "n_matched_30m": int(sum(abs(d) <= 1800 for d in ests)),
            "n_matched_60m": len(ests),
            "time_since_match_s": float(T - plans[-1]) if plans else np.nan,
        }
        rows.append(row)
    est_df = pd.DataFrame(rows).set_index("sample_id")

    out = feats.copy()
    idx = out["sample_id"].map(est_df["est_dev_last"])
    out["cur_dev_s"] = idx
    for col in ("est_dev_last", "est_dev_slope", "est_dev_mean3", "est_dev_std5", "n_matched_30m", "n_matched_60m", "time_since_match_s"):
        out[col] = out["sample_id"].map(est_df[col])
    out["cur_x_horizon"] = out["cur_dev_s"] * out["horizon_s"] / 900.0
    out["cur_over_horizon"] = out["cur_dev_s"] / out["horizon_s"].clip(lower=1)
    return out


def main() -> None:
    for split in ("train", "test"):
        out = build_stream_features(split)
        path = Path(f"data/gt/features_stream_{split}.parquet")
        out.to_parquet(path, index=False)
        est = out["cur_dev_s"]
        cur = pd.read_parquet(f"data/gt/features_{split}.parquet")["cur_dev_s"]
        mask = est.notna()
        corr = np.corrcoef(est[mask], cur[mask])[0, 1]
        print(f"[stream-feats] {split}: saved {len(out)} rows, est-vs-hint corr={corr:.3f} on {mask.sum()}")


if __name__ == "__main__":
    main()
