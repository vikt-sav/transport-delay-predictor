"""Online prediction pipeline: stream state, feature computation, model inference.

Mirrors the offline feature definitions (tools/gt_features.py) using only data
available at prediction time, per the anti-leak rule.
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from .gt import load_schedule, lonlat_to_xy, reconstruct_stop_matches
from .gt_priors import add_stop_key

WINDOW_S = 600
HORIZON_LO_S = 600
HORIZON_HI_S = 900
ACTIVE_S = 180
MATCH_RADIUS_M = 90.0
MATCH_WINDOW_S = 300.0
PLAN_LEAD_S = 30.0
WHATIF_BUNCHING_SHARE = 0.35
WHATIF_MIN_FLEET = 1.0
HMM_PER_TICK = 3
JUMP_MIN_M = 800.0
JUMP_VMAX_MPS = 50.0
CENTER_MAX_M = 40_000.0
STAY_MAX_M = 5_000.0
ROUTE_SNAP_MAX_M = 400.0
_LAT_M_DEG = 111_132.0


def _lon_scale(lat0: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat0))


def _drop_gps_jumps(ts, lat, lon, spd, hdg, lat0, lon0):
    """Drop isolated GPS teleports (dataset glitches) from a vehicle track.

    A point is dropped when it is far from BOTH neighbours (implied speed over
    50 m/s or over 800 m jump), or when it is the newest/oldest point of the
    window and far from its single neighbour, or farther than 40 km from the
    route center. Legit movement survives: thresholds scale with the time gap.
    """
    n = len(ts)
    if n < 3:
        return ts, lat, lon, spd, hdg
    x = (lon - lon0) * _lon_scale(lat0)
    y = (lat - lat0) * _LAT_M_DEG
    d = np.hypot(np.diff(x), np.diff(y))
    dt = np.abs(np.diff(ts))
    thr = np.maximum(JUMP_MIN_M, JUMP_VMAX_MPS * dt)
    far_from_prev = np.zeros(n, dtype=bool)
    far_from_prev[1:] = d > thr
    far_to_next = np.zeros(n, dtype=bool)
    far_to_next[:-1] = d > thr
    keep = ~(far_from_prev & far_to_next)
    # Endpoints: drop only when the single neighbour itself survived (otherwise
    # the gap just spans an already-dropped glitch).
    keep[0] = keep[0] and not (far_to_next[0] and keep[1])
    keep[-1] = keep[-1] and not (far_from_prev[-1] and keep[-2])
    keep &= np.hypot(x, y) <= CENTER_MAX_M
    if keep.all():
        return ts, lat, lon, spd, hdg
    return ts[keep], lat[keep], lon[keep], spd[keep], hdg[keep]


class VehicleState:
    __slots__ = ("records", "total", "valid")

    def __init__(self):
        self.records: deque = deque(maxlen=8000)
        self.total = 0
        self.valid = 0


class OnlinePipeline:
    def __init__(self, schedule_path: str, model_path: str, meta_path: str | None = None,
                 priors_path: str | None = None, use_hmm_matching: bool = False,
                 hmm_roads_path: str | None = None):
        self.schedule = load_schedule(schedule_path)
        self.schedule = add_stop_key(self.schedule)
        self.schedule["trip_id"] = (
            (self.schedule.groupby("tr_id")["plan_s"].diff().fillna(10**9) > 900).cumsum()
        )
        self.schedule["idx_in_trip"] = self.schedule.groupby(["tr_id", "trip_id"]).cumcount()
        self.schedule["n_in_trip"] = self.schedule.groupby(["tr_id", "trip_id"])["tt_action_item_id"].transform("count")
        self.schedule["stops_left"] = self.schedule["n_in_trip"] - self.schedule["idx_in_trip"]

        if priors_path and Path(priors_path).exists():
            priors = pd.read_parquet(priors_path)
            priors["prior_mean"] = priors["prior_sum"] / priors["prior_n"].replace(0, np.nan)
            self.schedule = self.schedule.merge(
                priors[["stop_key", "prior_median", "prior_mean", "prior_sum", "prior_n"]],
                on="stop_key",
                how="left",
                suffixes=("", "_p"),
            )
        else:
            for c in ("prior_median", "prior_mean", "prior_sum", "prior_n"):
                self.schedule[c] = np.nan

        import json

        from catboost import CatBoostRegressor

        self.model = CatBoostRegressor()
        self.model.load_model(model_path)
        meta_file = Path(meta_path) if meta_path else Path(model_path).parent / "meta_final.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        self.features = meta["features"]
        self.cat_features = meta["cat"]
        self.vehicles: dict[int, VehicleState] = {}
        self.last_snapshot: list[dict] = []
        self.snapshot_ts: float = 0.0
        self.stats = {"records": 0, "vehicles": 0, "predicts": 0}
        self._last_feats: dict[int, dict] = {}
        self._stream_ts: float | None = None
        self.hmm = None
        if use_hmm_matching:
            try:
                from .config import ROOT
                from .matching.hmm import HMMTrackMatcher, default_roads_path

                roads = Path(hmm_roads_path) if hmm_roads_path else default_roads_path(ROOT)
                if roads is not None and Path(roads).exists():
                    self.hmm = HMMTrackMatcher(roads)
                    print(f"[hmm] HMM street matching enabled ({roads.name})")
                else:
                    print("[hmm] no OSM roads cache found, HMM matching disabled")
            except Exception as exc:
                print(f"[hmm] HMM matching init failed, falling back to radius matching: {exc}")
                self.hmm = None
        self._in_tick = False
        self._hmm_left = 0
        self._stop_trees: dict[int, object] = {}
        self._route_lines: dict[int, tuple] | None = None
        self._last_onroute: dict[int, tuple[float, float]] = {}
        self._unit_map: dict[int, int] | None = None
        self._sched_ref_ts: float | None = None
        self._traffic_csv = Path(schedule_path).parent / "traffic.csv"

    def add_record(self, tr_id: int, ts: float, lat: float, lon: float, valid: bool, speed: float, heading: float, altitude: float = 0.0) -> None:
        state = self.vehicles.setdefault(tr_id, VehicleState())
        state.records.append((ts, lat, lon, valid, speed, heading, altitude))
        state.total += 1
        if valid:
            state.valid += 1
        self._stream_ts = ts
        self.stats["records"] += 1

    def map_tr_id(self, unit_id: int) -> int:
        """unit_id потока → tr_id расписания (связка 1:1 из traffic.csv).

        Внешний поток (эмулятор бортовых терминалов) шлёт датасетные unit_id; в replay
        unit_id уже равен tr_id — словарь возвращает его же.
        """
        if self._unit_map is None:
            mapping: dict[int, int] = {}
            try:
                t = pd.read_csv(self._traffic_csv, usecols=["unit_id", "tr_id"])
                t = t.dropna().astype({"unit_id": int, "tr_id": int})
                mapping = t.groupby("unit_id")["tr_id"].agg(
                    lambda s: int(s.value_counts().idxmax())
                ).to_dict()
            except Exception as exc:
                print(f"[units] unit_id->tr_id map unavailable ({exc}); fallback to unit_id")
            self._unit_map = mapping
        return self._unit_map.get(int(unit_id), int(unit_id))

    def align_ts(self, ts: float) -> float:
        """Приводит время потока к суткам расписания.

        Эмулятор ставит в пакеты текущее время, а расписание датасета на
        06.01.2026 — сдвигаем поток целыми сутками к дню расписания (время
        суток сохраняется; день недели может сместиться, его вклад в фичи
        незначителен). Поток из самого датасета проходит без сдвига.
        """
        if self._sched_ref_ts is None and not self.schedule.empty:
            self._sched_ref_ts = float(self.schedule["plan_s"].median())
        ref = self._sched_ref_ts
        if ref is None or not np.isfinite(ts):
            return ts
        k = round((ts - ref) / 86400.0)
        return ts - k * 86400.0 if k else ts

    def _target_stop(self, tr_id: int, now_s: float):
        sc = self.schedule[self.schedule["tr_id"] == tr_id]
        if sc.empty:
            return None
        mask = (sc["plan_s"] > now_s + HORIZON_LO_S) & (sc["plan_s"] <= now_s + HORIZON_HI_S)
        cand = sc[mask].sort_values("plan_s")
        if cand.empty:
            return None
        return cand.iloc[0]

    def _stop_tree(self, tr_id: int, sc_all, lat0: float, lon0: float):
        """KDTree of the vehicle's own stop coordinates (cached per vehicle)."""
        if tr_id in self._stop_trees:
            return self._stop_trees[tr_id]
        tree = None
        s = sc_all.dropna(subset=["stop_lat", "stop_lon"])
        if len(s):
            from scipy.spatial import cKDTree

            sy, sx = lonlat_to_xy(s["stop_lat"].values, s["stop_lon"].values, lat0, lon0)
            tree = cKDTree(np.column_stack([sx, sy]))
        self._stop_trees[tr_id] = tree
        return tree

    def _route_line_trees(self) -> dict[int, tuple]:
        """Route polylines (the ones drawn on the dashboard) for display snapping."""
        if self._route_lines is not None:
            return self._route_lines
        self._route_lines = {}
        try:
            from scipy.spatial import cKDTree

            from .config import ROOT

            path = ROOT / "data" / "osm" / "routes_osm.geojson"
            if path.exists():
                geo = json.loads(path.read_text(encoding="utf-8"))
                for feat in geo.get("features", []):
                    if feat.get("geometry", {}).get("type") != "LineString":
                        continue
                    tr_id = int(feat["properties"].get("tr_id", -1))
                    coords = np.array(feat["geometry"]["coordinates"], dtype=float)
                    if len(coords) >= 2:
                        self._route_lines[tr_id] = (cKDTree(coords), coords)
        except Exception as exc:
            print(f"[routes] display line-snap disabled: {exc}")
            self._route_lines = {}
        return self._route_lines

    def _vehicle_features(self, tr_id: int, now_s: float, target) -> dict:
        state = self.vehicles.get(tr_id)
        if state is None or not state.records:
            return {}
        recs = [r for r in state.records if r[0] <= now_s]
        if not recs:
            return {}
        vrecs = [r for r in recs if r[3] and r[1] is not None and np.isfinite(r[1]) and np.isfinite(r[2])]
        if not vrecs:
            return {}
        ts = np.array([r[0] for r in vrecs], dtype=np.int64)
        lat = np.array([r[1] for r in vrecs])
        lon = np.array([r[2] for r in recs if r[3] and r[1] is not None and np.isfinite(r[1]) and np.isfinite(r[2])])
        spd = np.array([r[4] for r in vrecs])
        hdg = np.array([r[5] for r in vrecs])

        sc_all = self.schedule[self.schedule["tr_id"] == tr_id]
        if len(sc_all) and sc_all["stop_lat"].notna().any():
            lat0 = float(sc_all["stop_lat"].median())
            lon0 = float(sc_all["stop_lon"].median())
        else:
            lat0, lon0 = float(np.nanmedian(lat)), float(np.nanmedian(lon))

        # GPS-гигиена: изолированные телепорты не должны попадать ни в фичи,
        # ни в позицию маркера. Коридор собственных остановок применяется
        # только к позиции маркера (disp_*): в датасете встречаются ТС,
        # чей трек легитимно далёк от расписания — их фичи трогать нельзя.
        ts, lat, lon, spd, hdg = _drop_gps_jumps(ts, lat, lon, spd, hdg, lat0, lon0)
        if len(ts) == 0:
            return {}
        st_tree = self._stop_tree(tr_id, sc_all, lat0, lon0)
        corr_d = None
        if st_tree is not None:
            corr_x = (lon - lon0) * _lon_scale(lat0)
            corr_y = (lat - lat0) * _LAT_M_DEG
            corr_d, _ = st_tree.query(np.column_stack([corr_x, corr_y]))

        f: dict = {}
        f["cur_dev_s"] = np.nan
        f["horizon_s"] = float(target["plan_s"] - now_s)
        f["age_last_s"] = float(now_s - ts[-1])
        f["speed_last"] = float(spd[-1])
        f["lat_last"] = float(lat[-1])
        f["lon_last"] = float(lon[-1])
        f["valid_rate_all"] = float(state.valid / max(state.total, 1))
        heading = float(hdg[-1]) if len(hdg) else np.nan
        f["heading"] = heading if (np.isfinite(heading) and 0.0 <= heading < 360.0) else np.nan

        # Позиция маркера: последняя точка в коридоре собственных остановок.
        # Если хвост трека «улетел» (цепочка глитчей, которую не поймал
        # jump-фильтр) — показываем последнюю правдоподобную точку.
        if corr_d is not None:
            ok_idx = np.nonzero(corr_d <= STAY_MAX_M)[0]
            if len(ok_idx):
                f["disp_lat"] = float(lat[ok_idx[-1]])
                f["disp_lon"] = float(lon[ok_idx[-1]])

        gy, gx = lonlat_to_xy(np.nan_to_num(lat), np.nan_to_num(lon), lat0, lon0)
        for win in (5, 10, 15, 30):
            w = win * 60
            m = (ts > now_s - w) & (ts <= now_s)
            v = spd[m]
            f[f"speed_mean_{win}m"] = float(np.mean(v)) if m.any() else np.nan
            f[f"speed_std_{win}m"] = float(np.std(v)) if m.sum() > 1 else np.nan
            f[f"speed_max_{win}m"] = float(np.max(v)) if m.any() else np.nan
            f[f"npts_{win}m"] = int(m.sum())
            f[f"idle_frac_{win}m"] = float(np.mean(v < 5.0)) if m.any() else np.nan
            if m.sum() >= 2:
                pts_xy = np.column_stack([gx[m], gy[m]])
                travel = float(np.sum(np.linalg.norm(np.diff(pts_xy, axis=0), axis=1)))
                disp = float(np.linalg.norm(pts_xy[-1] - pts_xy[0]))
                f[f"travel_m_{win}m"] = travel
                f[f"disp_m_{win}m"] = disp
                f[f"avg_v_{win}m"] = travel / w
            else:
                f[f"travel_m_{win}m"] = np.nan
                f[f"disp_m_{win}m"] = np.nan
                f[f"avg_v_{win}m"] = np.nan

        cur_xy = np.array([gx[-1], gy[-1]])
        tlat, tlon = float(target["stop_lat"]), float(target["stop_lon"])
        ty, tx = lonlat_to_xy(np.array([tlat]), np.array([tlon]), lat0, lon0)
        f["dist_to_target_m"] = float(np.hypot(tx[0] - cur_xy[0], ty[0] - cur_xy[1]))
        f["req_speed_mps"] = f["dist_to_target_m"] / max(f["horizon_s"], 1)

        sc = self.schedule[self.schedule["tr_id"] == tr_id]
        sc = sc[sc["plan_s"] <= now_s - PLAN_LEAD_S].dropna(subset=["stop_lat", "stop_lon"]).sort_values("plan_s")
        delays, matched_plans = [], []
        stuck_street = None
        stuck_s = None
        if not sc.empty and len(ts) >= 1:
            sy, sx = lonlat_to_xy(sc["stop_lat"].values, sc["stop_lon"].values, lat0, lon0)
            matches = reconstruct_stop_matches(
                np.column_stack([gx, gy]),
                ts,
                np.column_stack([sx, sy]),
                sc["plan_s"].values.astype(np.int64),
                sc["tt_action_item_id"].values,
                radius_m=MATCH_RADIUS_M,
                window_s=MATCH_WINDOW_S,
            )
            m = matches[matches["matched"] == 1]
            delays = m["est_delay_s"].tolist()
            matched_plans = m["plan_s"].tolist()

            hmm = self.hmm
            if hmm is not None and len(ts) >= 3:
                allow = (not self._in_tick) or self._hmm_left > 0
                snapped, fresh = hmm.snap_track(tr_id, ts, lat, lon, allow_compute=allow)
                if fresh:
                    self._hmm_left -= 1
                if snapped is not None and len(m):
                    m_pos = m.index.to_numpy()
                    hmm_delays = hmm.stop_delays(
                        snapped,
                        m["tt_action_item_id"].values,
                        m["plan_s"].values.astype(np.float64),
                        sc["stop_lat"].values[m_pos],
                        sc["stop_lon"].values[m_pos],
                    )
                    if hmm_delays:
                        delays = [hmm_delays.get(int(sid), d) for sid, d in zip(m["tt_action_item_id"].values, delays, strict=False)]
                if snapped is not None:
                    stuck = hmm.stuck_street(snapped, now_s)
                    if stuck is not None:
                        stuck_street, stuck_s = stuck
                    # Смещение «сырая точка -> точка на улице» для отображения:
                    # маркер садится на дорогу даже при боковом дрейфе GPS.
                    if snapped.raw_lat is not None:
                        off_lat = snapped.lat[-1] - snapped.raw_lat
                        off_lon = snapped.lon[-1] - snapped.raw_lon
                        if abs(off_lat) * _LAT_M_DEG < 400.0 and abs(off_lon) * _lon_scale(lat0) < 400.0:
                            base_lat = f.get("disp_lat", f["lat_last"])
                            base_lon = f.get("disp_lon", f["lon_last"])
                            f["disp_lat"] = base_lat + off_lat
                            f["disp_lon"] = base_lon + off_lon

        # Финальная гарантия визуала: маркер прижимается к своей линии маршрута
        # (в пределах 400 м), иначе держится последней позиции на линии —
        # «улететь в пустоту» отображение не может в принципе.
        rt = self._route_line_trees().get(tr_id)
        if rt is not None:
            base_lat = f.get("disp_lat", f["lat_last"])
            base_lon = f.get("disp_lon", f["lon_last"])
            dd, ii = rt[0].query([[base_lon, base_lat]], k=8)
            cand = [
                (float(dd[0][j]), float(rt[1][int(ii[0][j]), 1]), float(rt[1][int(ii[0][j]), 0]))
                for j in range(ii.shape[1])
                if float(dd[0][j]) <= ROUTE_SNAP_MAX_M
            ]
            if cand:
                prev = self._last_onroute.get(tr_id)
                if prev is not None:
                    cand.sort(key=lambda c: (c[1] - prev[0]) ** 2 + (c[2] - prev[1]) ** 2)
                f["disp_lat"], f["disp_lon"] = cand[0][1], cand[0][2]
                self._last_onroute[tr_id] = (f["disp_lat"], f["disp_lon"])
            elif tr_id in self._last_onroute:
                f["disp_lat"], f["disp_lon"] = self._last_onroute[tr_id]
        f["stuck_street"] = stuck_street
        f["stuck_s"] = stuck_s
        if delays:
            f["est_dev_last"] = float(delays[-1])
            f["est_dev_slope"] = float(delays[-1] - delays[-2]) if len(delays) >= 2 else np.nan
            f["est_dev_mean3"] = float(np.mean(delays[-3:]))
            f["est_dev_std5"] = float(np.std(delays[-5:])) if len(delays) >= 2 else np.nan
        else:
            for k in ("est_dev_last", "est_dev_slope", "est_dev_mean3", "est_dev_std5"):
                f[k] = np.nan
        f["n_matched_30m"] = int(sum(abs(d) <= 1800 for d in delays))
        f["n_matched_60m"] = len(delays)
        f["time_since_match_s"] = float(now_s - matched_plans[-1]) if matched_plans else np.nan

        if getattr(self, "use_est_dev_as_cur", False):
            f["cur_dev_s"] = f.get("est_dev_last", np.nan)

        f["stops_before_target"] = int((sc_all["plan_s"] < target["plan_s"]).sum())
        prev_all = sc_all[sc_all["plan_s"] < target["plan_s"]]["plan_s"].values
        f["planned_gap_prev_s"] = float(target["plan_s"] - prev_all[-1]) if len(prev_all) else np.nan
        trow = self.schedule[(self.schedule["tt_action_item_id"] == target["tt_action_item_id"])]
        if not trow.empty:
            f["idx_in_trip"] = int(trow["idx_in_trip"].iloc[0])
            f["n_stops_in_trip"] = int(trow["n_in_trip"].iloc[0])
            f["stops_left_in_trip"] = int(trow["stops_left"].iloc[0])
            f["run_from_trip_start_s"] = float(
                target["plan_s"]
                - self.schedule[(self.schedule["tr_id"] == tr_id) & (self.schedule["trip_id"] == trow["trip_id"].iloc[0])]["plan_s"].min()
            )
        else:
            f["idx_in_trip"] = np.nan
            f["n_stops_in_trip"] = np.nan
            f["stops_left_in_trip"] = np.nan
            f["run_from_trip_start_s"] = np.nan

        f["stop_lat"] = target["stop_lat"]
        f["stop_lon"] = target["stop_lon"]
        f["prior_mean"] = target.get("prior_mean", np.nan)
        f["prior_sum"] = target.get("prior_sum", np.nan)
        f["prior_n"] = target.get("prior_n", np.nan)
        f["prior_median"] = target.get("prior_median", np.nan)
        f["prior_known"] = 1.0 if pd.notna(target.get("prior_mean", np.nan)) else 0.0

        dt_obj = pd.Timestamp(now_s, unit="s")
        f["hour"] = dt_obj.hour
        f["minute"] = dt_obj.minute
        f["dow"] = dt_obj.dayofweek
        f["targ_hour"] = pd.Timestamp(target["plan_s"], unit="s").hour
        f["targ_minute"] = pd.Timestamp(target["plan_s"], unit="s").minute
        f["cur_x_horizon"] = f["cur_dev_s"] * f["horizon_s"] / 900.0
        f["cur_over_horizon"] = f["cur_dev_s"] / max(f["horizon_s"], 1)
        f["tr_id"] = str(tr_id)
        f["target_stop_id"] = str(target["tt_action_item_id"])
        f["stop_key"] = target.get("stop_key", "nan_nan")
        return f

    def predict_vehicle(self, tr_id: int, now_s: float) -> dict | None:
        target = self._target_stop(tr_id, now_s)
        if target is None:
            return None
        feats = self._vehicle_features(tr_id, now_s, target)
        if not feats:
            return None
        row = {c: feats.get(c, np.nan) for c in self.features}
        self._last_feats[tr_id] = feats
        x = pd.DataFrame([row])
        for c in self.cat_features:
            x[c] = x[c].astype(str)
        pred = float(self.model.predict(x)[0])
        self.stats["predicts"] += 1

        def _num(v):
            if v is None:
                return None
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return None
            return fv if np.isfinite(fv) else None

        return {
            "tr_id": int(tr_id),
            "lat": _num(feats.get("lat_last")),
            "lon": _num(feats.get("lon_last")),
            "stop_id": int(target["tt_action_item_id"]),
            "planned_arrival": pd.Timestamp(target["plan_s"], unit="s").isoformat(),
            "horizon_s": _num(feats.get("horizon_s")),
            "pred_s": round(max(-420.0, min(700.0, pred)), 1),
            "est_dev_s": _num(feats.get("est_dev_last")),
            "speed_last": _num(feats.get("speed_last")),
            "heading": _num(feats.get("heading")),
            "disp_lat": _num(feats.get("disp_lat")),
            "disp_lon": _num(feats.get("disp_lon")),
            "stuck_street": feats.get("stuck_street"),
            "stuck_s": _num(feats.get("stuck_s")),
        }

    def tick(self) -> list[dict]:
        now_s = self._stream_ts if self._stream_ts is not None else time.time()
        self._in_tick = True
        self._hmm_left = HMM_PER_TICK
        try:
            rows = self._tick_impl(now_s)
        finally:
            self._in_tick = False
        rows.sort(key=lambda r: -(r["pred_s"] if r["pred_s"] is not None else -1e9))
        self.last_snapshot = rows
        self.snapshot_ts = time.time()
        self.stats["vehicles"] = len(self.vehicles)
        return rows

    def _tick_impl(self, now_s: float) -> list[dict]:
        rows = []
        for tr_id, state in self.vehicles.items():
            if not state.records:
                continue
            last_seen = max((r[0] for r in state.records if r[0] <= now_s), default=None)
            if last_seen is None:
                continue
            if now_s - last_seen > ACTIVE_S:
                continue
            row = self.predict_vehicle(tr_id, now_s)
            if row is not None:
                feats = self._last_feats.get(tr_id)
                if feats:
                    for extra in ("sched_headway_s", "stops_left_in_trip", "count_route_10m"):
                        value = feats.get(extra)
                        row[extra] = float(value) if value is not None and np.isfinite(value) else None
                rows.append(row)
        return rows

    def whatif(self, extra_vehicles: int = 0, dwell_reduction_s: float = 0.0) -> dict:
        """Scenario layer on top of ML predictions.

        extra_vehicles: additional vehicles dispatched on the line -> interval
        shrinks by N/(N+K), which reduces the bunching share of predicted delay.
        dwell_reduction_s: seconds saved per remaining stop before the target.
        """
        results = []
        for row in self.last_snapshot:
            headway = float(row.get("sched_headway_s") or 600.0)
            horizon = float(row.get("horizon_s") or 720.0)
            stops_left = row.get("stops_left_in_trip")
            n_stops_ahead = int(np.clip(round(horizon / max(headway, 60.0)), 1, max(stops_left or 1, 1)))
            saved_dwell = dwell_reduction_s * n_stops_ahead

            fleet_n = max(WHATIF_MIN_FLEET, round(3600.0 / max(headway, 60.0)))
            fleet_factor = 1.0 - fleet_n / (fleet_n + max(extra_vehicles, 0))
            saved_bunching = fleet_factor * WHATIF_BUNCHING_SHARE * max(row["pred_s"], 0.0)

            adjusted = float(np.clip(row["pred_s"] - saved_dwell - saved_bunching, -600.0, 700.0))
            results.append(
                {
                    "tr_id": row["tr_id"],
                    "stop_id": row["stop_id"],
                    "pred_s": row["pred_s"],
                    "whatif_s": round(adjusted, 1),
                    "saved_dwell_s": round(saved_dwell, 1),
                    "saved_bunching_s": round(saved_bunching, 1),
                }
            )
        base_mean = float(np.mean([r["pred_s"] for r in results])) if results else 0.0
        scen_mean = float(np.mean([r["whatif_s"] for r in results])) if results else 0.0
        headways = [float(r.get("sched_headway_s") or 600.0) for r in self.last_snapshot]
        headway_before = float(np.mean(headways)) if headways else 0.0
        fleet_n_mean = float(np.mean([max(1.0, round(3600.0 / max(h, 60.0))) for h in headways])) if headways else 0.0
        headway_after = headway_before * fleet_n_mean / (fleet_n_mean + max(extra_vehicles, 0))
        return {
            "extra_vehicles": extra_vehicles,
            "dwell_reduction_s": dwell_reduction_s,
            "vehicles": results,
            "summary": {
                "n_vehicles": len(results),
                "mean_base_pred_s": round(base_mean, 1),
                "mean_scenario_pred_s": round(scen_mean, 1),
                "improvement_s": round(base_mean - scen_mean, 1),
                "mean_interval_before_s": round(headway_before, 0),
                "mean_interval_after_s": round(headway_after, 0),
            },
        }

    def alerts(self) -> list[dict]:
        out = []
        for r in self.last_snapshot:
            if r["pred_s"] is None or r["pred_s"] < 120:
                continue
            if (r.get("speed_last") or 0) < 5:
                cause = "длительный простой / затор"
                recommendation = "Сократить время стоянки ТС на ближайших остановках для догона графика"
            elif (r.get("est_dev_s") or 0) > 60:
                cause = "накопленное отставание от графика"
                recommendation = "Изменить количество ТС на линии: выпустить резервное ТС для восстановления интервала"
            else:
                cause = "снижение скорости на сегменте"
                recommendation = "Контролировать сегмент; при сохранении темпа — подготовить резервное ТС на линии"
            out.append({**r, "cause": cause, "recommendation": recommendation})
        return out
