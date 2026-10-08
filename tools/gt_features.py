"""Builds the per-point feature table for the delay-prediction task.

Anti-leak rule: for a point at moment T only telemetry with event_time <= T is used.
"""
from __future__ import annotations

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
from mtp.gt_priors import attach_priors, build_stop_priors

WINDOWS_MIN = (5, 10, 15, 30)
TRIP_GAP_S = 900


def add_trip_features(veh_sched: pd.DataFrame) -> pd.DataFrame:
    sc = veh_sched.sort_values("plan_s").reset_index(drop=True)
    gap = sc["plan_s"].diff().fillna(TRIP_GAP_S + 1)
    sc["trip_id"] = (gap > TRIP_GAP_S).cumsum()
    sc["idx_in_trip"] = sc.groupby("trip_id").cumcount()
    sc["n_stops_in_trip"] = sc.groupby("trip_id")["tt_action_item_id"].transform("count")
    sc["run_from_trip_start_s"] = sc["plan_s"] - sc.groupby("trip_id")["plan_s"].transform("min")
    return sc


def build_vehicle_point_features(
    traffic: pd.DataFrame,
    schedule: pd.DataFrame,
    points: pd.DataFrame,
) -> pd.DataFrame:
    out_frames = []
    for tr_id, tr_points in points.groupby("tr_id"):
        veh_traffic = traffic[traffic["tr_id"] == tr_id]
        veh_sched = schedule[schedule["tr_id"] == tr_id].sort_values("plan_s").reset_index(drop=True)

        t_all = veh_traffic["ts"].values.astype(np.int64)
        valid = (veh_traffic["location_valid"] == True) & veh_traffic["lat"].notna()
        gv = veh_traffic[valid]
        g_t = gv["ts"].values.astype(np.int64)
        g_lat = gv["lat"].values.astype(float)
        g_lon = gv["lon"].values.astype(float)
        g_speed = gv["speed"].fillna(0.0).values.astype(float)

        lat0 = float(np.median(g_lat)) if len(g_lat) else 55.75
        lon0 = float(np.median(g_lon)) if len(g_lon) else 37.6
        gy, gx = lonlat_to_xy(g_lat, g_lon, lat0, lon0)
        gps_xy = np.column_stack([gx, gy])

        sy, sx = lonlat_to_xy(
            veh_sched["stop_lat"].values.astype(float),
            veh_sched["stop_lon"].values.astype(float),
            lat0,
            lon0,
        )
        stop_xy_all = np.column_stack([sx, sy])
        veh_sched = add_trip_features(veh_sched)
        stop_plan_sorted = veh_sched["plan_s"].values.astype(np.int64)
        pos_of_id = {sid: i for i, sid in enumerate(veh_sched["tt_action_item_id"].values)}
        trip_of_pos = veh_sched["trip_id"].values
        idx_in_trip = veh_sched["idx_in_trip"].values
        n_in_trip = veh_sched["n_stops_in_trip"].values
        run_from_start = veh_sched["run_from_trip_start_s"].values

        for p in tr_points.itertuples(index=False):
            T = int(p.T_s)
            t_cut_mask = t_all <= T
            g_cut = int(np.searchsorted(g_t, T, side="right"))

            feats = {
                "sample_id": p.sample_id,
                "tr_id": p.tr_id,
                "target_stop_id": p.target_stop_id,
                "cur_dev_s": p.cur_dev_s,
                "T": p.T,
                "target_s": p.target_s,
            }
            if hasattr(p, "target_delay_s"):
                feats["target_delay_s"] = p.target_delay_s

            horizon = int(p.target_s - T)
            feats["horizon_s"] = horizon

            if g_cut > 0:
                feats["age_last_s"] = T - float(g_t[g_cut - 1])
                feats["speed_last"] = float(g_speed[g_cut - 1])
                cur_xy = gps_xy[g_cut - 1]
            else:
                feats["age_last_s"] = np.nan
                feats["speed_last"] = np.nan
                cur_xy = None

            for win in WINDOWS_MIN:
                w = win * 60
                mask = (g_t[:g_cut] > T - w) & (g_t[:g_cut] <= T)
                v = g_speed[:g_cut][mask]
                feats[f"speed_mean_{win}m"] = float(np.mean(v)) if len(v) else np.nan
                feats[f"speed_std_{win}m"] = float(np.std(v)) if len(v) > 1 else np.nan
                feats[f"speed_max_{win}m"] = float(np.max(v)) if len(v) else np.nan
                feats[f"npts_{win}m"] = int(mask.sum())
                feats[f"idle_frac_{win}m"] = float(np.mean(v < 5.0)) if len(v) else np.nan
                if cur_xy is not None and mask.sum() >= 2:
                    pts_xy = gps_xy[:g_cut][mask]
                    travel = float(np.sum(np.linalg.norm(np.diff(pts_xy, axis=0), axis=1)))
                    disp = float(np.linalg.norm(pts_xy[-1] - pts_xy[0]))
                    feats[f"travel_m_{win}m"] = travel
                    feats[f"disp_m_{win}m"] = disp
                    feats[f"avg_v_{win}m"] = travel / w
                else:
                    feats[f"travel_m_{win}m"] = np.nan
                    feats[f"disp_m_{win}m"] = np.nan
                    feats[f"avg_v_{win}m"] = np.nan

            feats["valid_rate_all"] = float(valid.values[t_cut_mask].mean()) if t_cut_mask.any() else np.nan

            ti = pos_of_id.get(p.target_stop_id)
            if ti is not None and ti > 0:
                feats["stops_before_target"] = int(ti)
                feats["planned_gap_prev_s"] = float(p.target_s - stop_plan_sorted[ti - 1])
            elif ti == 0:
                feats["stops_before_target"] = 0
                feats["planned_gap_prev_s"] = np.nan
            else:
                feats["stops_before_target"] = np.nan
                feats["planned_gap_prev_s"] = np.nan

            if ti is not None:
                _ = trip_of_pos[ti]
                feats["idx_in_trip"] = int(idx_in_trip[ti])
                feats["n_stops_in_trip"] = int(n_in_trip[ti])
                feats["stops_left_in_trip"] = int(n_in_trip[ti] - idx_in_trip[ti])
                feats["run_from_trip_start_s"] = float(run_from_start[ti])
            else:
                for f in ("idx_in_trip", "n_stops_in_trip", "stops_left_in_trip", "run_from_trip_start_s"):
                    feats[f] = np.nan

            if cur_xy is not None and ti is not None:
                txy = stop_xy_all[ti]
                feats["dist_to_target_m"] = float(np.hypot(txy[0] - cur_xy[0], txy[1] - cur_xy[1]))
                feats["req_speed_mps"] = feats["dist_to_target_m"] / max(horizon, 1)
            else:
                feats["dist_to_target_m"] = np.nan
                feats["req_speed_mps"] = np.nan

            if ti is not None:
                feats["stop_lat"] = veh_sched["stop_lat"].iloc[ti]
                feats["stop_lon"] = veh_sched["stop_lon"].iloc[ti]

            cut_stops = int(np.searchsorted(stop_plan_sorted, T, side="right"))
            if cut_stops > 0 and g_cut > 0:
                matches = reconstruct_stop_matches(
                    gps_xy[:g_cut],
                    g_t[:g_cut],
                    stop_xy_all[:cut_stops],
                    stop_plan_sorted[:cut_stops],
                    veh_sched["tt_action_item_id"].values[:cut_stops],
                )
                m = matches[matches["matched"] == 1]
                delays = m["est_delay_s"].values
                feats["est_dev_last"] = float(delays[-1]) if len(delays) else np.nan
                feats["est_dev_slope"] = float(delays[-1] - delays[-2]) if len(delays) >= 2 else np.nan
                feats["est_dev_mean3"] = float(np.mean(delays[-3:])) if len(delays) >= 1 else np.nan
                feats["est_dev_std5"] = float(np.std(delays[-5:])) if len(delays) >= 2 else np.nan
                feats["n_matched_30m"] = int((m["plan_s"].values >= T - 1800).sum())
                feats["n_matched_60m"] = int((m["plan_s"].values >= T - 3600).sum())
                feats["time_since_match_s"] = T - float(m["plan_s"].values[-1]) if len(m) else np.nan
            else:
                for f in ("est_dev_last", "est_dev_slope", "est_dev_mean3", "est_dev_std5", "time_since_match_s"):
                    feats[f] = np.nan
                feats["n_matched_30m"] = 0
                feats["n_matched_60m"] = 0

            feats["hour"] = p.T.hour
            feats["minute"] = p.T.minute
            feats["dow"] = p.T.dayofweek
            feats["targ_hour"] = p.target_time_begin.hour
            feats["targ_minute"] = p.target_time_begin.minute
            feats["cur_x_horizon"] = p.cur_dev_s * horizon / 900.0
            feats["cur_over_horizon"] = p.cur_dev_s / max(horizon, 1)

            out_frames.append(feats)

    return pd.DataFrame(out_frames)


def main() -> None:
    base = Path("dataset")
    out_dir = Path("data/gt")
    out_dir.mkdir(parents=True, exist_ok=True)

    load_traffic(base / "train" / "traffic.csv")
    train_schedule = load_schedule(base / "train" / "schedule.csv")
    priors = build_stop_priors(train_schedule)
    priors.to_parquet(out_dir / "stop_priors.parquet", index=False)
    print(f"[features] stop priors: {len(priors)} stops")

    for split in ("train", "test", "validate"):
        if split == "validate":
            traffic = load_traffic(base / "validate" / "traffic.csv")
            schedule = load_schedule(base / "validate" / "schedule_plan.csv")
            points = load_points(base / "validate" / "points.csv")

        else:
            traffic = load_traffic(base / split / "traffic.csv")
            schedule = load_schedule(base / split / "schedule.csv")
            points = load_points(base / "labels" / f"labels_{split}.csv")
        print(f"[features] {split}: traffic {len(traffic)}, schedule {len(schedule)}, points {len(points)}")
        feats = build_vehicle_point_features(traffic, schedule, points)
        if split == "train":
            feats = attach_priors(feats, priors, target_col="target_delay_s")
        else:
            feats = attach_priors(feats, priors)
        feats.to_parquet(out_dir / f"features_{split}.parquet", index=False)
        print(f"[features] saved {len(feats)} rows -> {out_dir / f'features_{split}.parquet'}")


if __name__ == "__main__":
    main()
