"""Stage 2 map matching: HMM matching of vehicle telemetry onto OSM streets.

Snap GPS tracks to the drivable road graph (leuvenmapmatching, HMM over edges)
and use the snapped track to:

* refine stop-pass delay estimates for recently passed stops -> cleaner
  ``est_dev_*`` features than raw radius matching alone;
* detect the street segment where a vehicle is stuck ("участок застревания")
  for the incident card on the dashboard.

The road graph is loaded from the Overpass cache (``data/osm/roads_*.json``)
once into compact numpy arrays; a binary ``.compact.npz`` cache is written next
to the JSON so subsequent startups skip the heavy JSON parse. Matching runs on
a small sub-map cropped around the recent track, so the per-vehicle cost stays
in the tens-to-hundreds of milliseconds and is amortized across ticks.

Everything here is best-effort: any failure (missing cache, no path, too few
points) degrades silently to the plain radius matching of the online pipeline.
"""
from __future__ import annotations

import json
import logging
import math
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

_LAT_M = 111_132.0
_MATCHER_LOGGER = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")


def _lon_m(lat0: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat0))


def default_roads_path(root: Path | None = None) -> Path | None:
    """Finds the Overpass roads cache in the usual locations.

    `root` (package ROOT) works for repo checkouts; in Docker the package is
    installed into site-packages, so the CWD (/app) is checked as well.
    """
    osm_dirs = []
    if root is not None:
        osm_dirs.append(Path(root) / "data" / "osm")
    osm_dirs.append(Path.cwd() / "data" / "osm")
    osm_dirs.append(Path("/app") / "data" / "osm")
    for osm_dir in osm_dirs:
        if not osm_dir.exists():
            continue
        candidates = sorted(osm_dir.glob("roads_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    return None


class RoadGraphIndex:
    """Compact OSM road graph: node coords, edges, street names of nodes."""

    def __init__(self, node_ids: np.ndarray, coords: np.ndarray, edges: np.ndarray,
                 way_nodes: np.ndarray, way_ptr: np.ndarray, way_names: np.ndarray):
        from scipy.spatial import cKDTree

        self.node_ids = node_ids          # int64 [N]
        self.coords = coords              # float64 [N, 2] (lat, lon)
        self.edges = edges                # int64 [E, 2] node indices
        self.way_nodes = way_nodes        # int64 flat node indices on named ways
        self.way_ptr = way_ptr            # int64 [W + 1] offsets into way_nodes
        self.way_names = way_names        # unicode [W]

        lat0 = float(np.median(coords[:, 0]))
        lon0 = float(np.median(coords[:, 1]))
        self._lat0, self._lon0 = lat0, lon0
        self._lon_scale = _lon_m(lat0)
        xy = np.column_stack([
            (coords[:, 1] - lon0) * self._lon_scale,
            (coords[:, 0] - lat0) * _LAT_M,
        ])
        self._tree = cKDTree(xy)

        named = np.zeros(len(node_ids), dtype=bool)
        named[way_nodes] = True
        self._named_idx = np.nonzero(named)[0]
        self._named_names = np.empty(len(self._named_idx), dtype=way_names.dtype)
        filled = np.zeros(len(self._named_idx), dtype=bool)
        pos_of = {int(idx): i for i, idx in enumerate(self._named_idx)}
        for w in range(len(way_ptr) - 1):
            name = str(way_names[w])
            for nidx in way_nodes[way_ptr[w]:way_ptr[w + 1]]:
                i = pos_of.get(int(nidx))
                if i is not None and not filled[i]:
                    filled[i] = True
                    self._named_names[i] = name
        if len(self._named_idx):
            self._named_tree = cKDTree(xy[self._named_idx])
        else:
            self._named_tree = None

    # -- construction -------------------------------------------------------

    @classmethod
    def from_overpass(cls, data: dict) -> RoadGraphIndex:
        nodes: dict[int, tuple[float, float]] = {}
        ways: list[tuple[list[int], str]] = []
        for el in data.get("elements", []):
            t = el.get("type")
            if t == "node" and "lat" in el:
                nodes[int(el["id"])] = (float(el["lat"]), float(el["lon"]))
            elif t == "way" and "nodes" in el:
                name = str((el.get("tags") or {}).get("name") or "")
                ways.append(([int(n) for n in el["nodes"]], name))

        n = len(nodes)
        node_ids = np.fromiter(nodes.keys(), dtype=np.int64, count=n)
        coords = np.fromiter(
            (v for latlon in nodes.values() for v in latlon), dtype=np.float64, count=2 * n
        ).reshape(n, 2)
        id2idx = {int(nid): i for i, nid in enumerate(node_ids)}

        edge_pairs: list[tuple[int, int]] = []
        named_flat: list[int] = []
        named_ptr = [0]
        names: list[str] = []
        for way_node_ids, name in ways:
            seq = [id2idx[nid] for nid in way_node_ids if nid in id2idx]
            for a, b in pairwise(seq):
                if a != b:
                    edge_pairs.append((a, b))
            if name:
                uniq = []
                seen = set()
                for s in seq:
                    if s not in seen:
                        seen.add(s)
                        uniq.append(s)
                if uniq:
                    named_flat.extend(uniq)
                    named_ptr.append(len(named_flat))
                    names.append(name)

        edges = (
            np.unique(np.array(edge_pairs, dtype=np.int64), axis=0)
            if edge_pairs else np.zeros((0, 2), dtype=np.int64)
        )
        return cls(
            node_ids,
            coords,
            edges,
            np.array(named_flat, dtype=np.int64),
            np.array(named_ptr, dtype=np.int64),
            np.array(names, dtype=np.str_),
        )

    @classmethod
    def load(cls, roads_json: Path | str) -> RoadGraphIndex:
        roads_json = Path(roads_json)
        compact = roads_json.with_name(roads_json.stem + ".compact.npz")
        if compact.exists():
            try:
                with np.load(compact, allow_pickle=False) as z:
                    return cls(z["node_ids"], z["coords"], z["edges"], z["way_nodes"], z["way_ptr"], z["way_names"])
            except Exception as exc:
                _MATCHER_LOGGER.warning("compact road cache unreadable, reparsing json: %s", exc)
        t0 = time.perf_counter()
        data = json.loads(roads_json.read_text(encoding="utf-8"))
        graph = cls.from_overpass(data)
        print(f"[hmm] roads parsed in {time.perf_counter() - t0:.1f}s: "
              f"{len(graph.coords)} nodes, {len(graph.edges)} edges, {len(graph.way_names)} named ways")
        try:
            graph.save(compact)
            print(f"[hmm] compact cache -> {compact.name}")
        except Exception as exc:
            print(f"[hmm] compact cache not written: {exc}")
        return graph

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            node_ids=self.node_ids,
            coords=self.coords,
            edges=self.edges,
            way_nodes=self.way_nodes,
            way_ptr=self.way_ptr,
            way_names=self.way_names,
        )

    # -- queries ------------------------------------------------------------

    def sub_graph(self, latlons: np.ndarray, pad_m: float = 300.0):
        """Nodes/edges around the given points (corridor of width ~2*pad_m)."""
        xy = np.column_stack([
            (np.asarray(latlons)[:, 1] - self._lon0) * self._lon_scale,
            (np.asarray(latlons)[:, 0] - self._lat0) * _LAT_M,
        ])
        mask = np.zeros(len(self.coords), dtype=bool)
        for p in xy:
            near = self._tree.query_ball_point(p, r=pad_m)
            if near:
                mask[near] = True
        sel = np.nonzero(mask)[0]
        if len(sel) < 2 or len(self.edges) == 0:
            return None
        emask = mask[self.edges[:, 0]] & mask[self.edges[:, 1]]
        sub_edges = self.edges[emask]
        if len(sub_edges) == 0:
            return None
        local = np.full(len(self.coords), -1, dtype=np.int64)
        local[sel] = np.arange(len(sel))
        return sel, self.coords[sel], local[sub_edges]

    def nearest_named_street(self, lat: float, lon: float, max_m: float = 150.0) -> str | None:
        if self._named_tree is None:
            return None
        x = (lon - self._lon0) * self._lon_scale
        y = (lat - self._lat0) * _LAT_M
        dist, idx = self._named_tree.query([x, y])
        if dist > max_m:
            return None
        return str(self._named_names[int(idx)])


def _build_inmem_map(sel_coords: np.ndarray, sub_edges: np.ndarray):
    from leuvenmapmatching.map.inmem import InMemMap

    mapp = InMemMap("sub", use_latlon=True, use_rtree=False)
    graph = mapp.graph
    for i, (la, lo) in enumerate(sel_coords):
        graph[i] = ((float(la), float(lo)), [])
    for a, b in sub_edges:
        ia, ib = int(a), int(b)
        graph[ia][1].append(ib)
        graph[ib][1].append(ia)
    return mapp


class SnappedTrack:
    __slots__ = ("lat", "lon", "raw_lat", "raw_lon", "src_ts", "ts")

    def __init__(self, ts: list[float], lat: list[float], lon: list[float], src_ts: float,
                 raw_lat: float | None = None, raw_lon: float | None = None):
        self.ts = ts
        self.lat = lat
        self.lon = lon
        self.src_ts = src_ts
        self.raw_lat = raw_lat
        self.raw_lon = raw_lon

    def __len__(self):
        return len(self.ts)

    def as_array(self) -> np.ndarray:
        return np.column_stack([self.lat, self.lon])


class _VehState:
    __slots__ = ("gap", "snapped", "src_ts")

    def __init__(self, min_gap_s: float):
        self.snapped: SnappedTrack | None = None
        self.src_ts: float = float("-inf")
        self.gap: float = min_gap_s


class HMMTrackMatcher:
    """Snap per-vehicle tracks with HMM; estimate stop delays and stuck streets.

    Track window covers the recent history: enough for the ±window_s bands of
    recently due stops and for stuck detection. Older stops keep the plain
    radius-matching estimates (feature distribution stays close to training).
    """

    def __init__(self,
                 roads_json: Path | str,
                 radius_m: float = 90.0,
                 window_s: float = 300.0,
                 track_window_s: float = 720.0,
                 min_dt_s: float = 10.0,
                 pad_m: float = 300.0,
                 max_track_points: int = 90,
                 min_gap_s: float = 240.0,
                 stuck_window_s: float = 900.0,
                 stuck_min_s: float = 120.0,
                 stuck_speed_mps: float = 2.5):
        self.graph = RoadGraphIndex.load(roads_json)
        self.radius_m = radius_m
        self.window_s = window_s
        self.track_window_s = track_window_s
        self.min_dt_s = min_dt_s
        self.pad_m = pad_m
        self.max_track_points = max_track_points
        self.min_gap_s = min_gap_s
        self.stuck_window_s = stuck_window_s
        self.stuck_min_s = stuck_min_s
        self.stuck_speed_mps = stuck_speed_mps
        self._state: dict[int, _VehState] = {}
        _MATCHER_LOGGER.setLevel(logging.ERROR)

    # -- track snapping -----------------------------------------------------

    def _decimate(self, ts: np.ndarray, lat: np.ndarray, lon: np.ndarray):
        if len(ts) < 3:
            return ts, lat, lon
        keep = [0]
        last_t = ts[0]
        for i in range(1, len(ts) - 1):
            if ts[i] - last_t >= self.min_dt_s:
                keep.append(i)
                last_t = ts[i]
        keep.append(len(ts) - 1)
        if len(keep) > self.max_track_points:
            step = math.ceil(len(keep) / self.max_track_points)
            keep = keep[::step]
            if keep[-1] != len(ts) - 1:
                keep[-1] = len(ts) - 1
        return ts[keep], lat[keep], lon[keep]

    def _make_matcher(self, mapp):
        from leuvenmapmatching.matcher.distance import DistanceMatcher

        return DistanceMatcher(
            mapp,
            max_dist=300.0,
            max_dist_init=300.0,
            obs_noise=25.0,
            min_prob_norm=None,
            non_emitting_states=True,
            max_lattice_width=10,
            only_edges=True,
        )

    def _match_track(self, ts: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> SnappedTrack | None:
        pts = np.column_stack([lat, lon])
        sub = self.graph.sub_graph(pts, pad_m=self.pad_m)
        if sub is None:
            return None
        sel_coords, sub_edges = sub[1], sub[2]
        try:
            mapp = _build_inmem_map(sel_coords, sub_edges)
            matcher = self._make_matcher(mapp)
            matcher.match([tuple(p) for p in pts])
            snapped = self._project_lattice(mapp, ts, lat, lon, matcher.lattice_best)
        except Exception:
            return None
        if snapped is None or len(snapped) < 2:
            return None
        # Early stop (graph gaps, tunnels, off-road fallback segments):
        # re-match the recent tail so fresh observations — the ones that
        # matter for est_dev_last and stuck detection — stay covered.
        covered = len(snapped)
        if covered < len(ts) and len(ts) >= 8:
            tail0 = max(covered, len(ts) // 2)
            tail = self._match_simple(ts[tail0:], lat[tail0:], lon[tail0:])
            if tail is not None:
                return SnappedTrack(
                    snapped.ts + tail.ts,
                    snapped.lat + tail.lat,
                    snapped.lon + tail.lon,
                    float(ts[-1]),
                    raw_lat=float(lat[-1]),
                    raw_lon=float(lon[-1]),
                )
        snapped.raw_lat = float(lat[-1])
        snapped.raw_lon = float(lon[-1])
        return snapped

    def _match_simple(self, ts: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> SnappedTrack | None:
        """Match a shorter track without tail retries."""
        pts = np.column_stack([lat, lon])
        sub = self.graph.sub_graph(pts, pad_m=self.pad_m)
        if sub is None:
            return None
        sel_coords, sub_edges = sub[1], sub[2]
        try:
            mapp = _build_inmem_map(sel_coords, sub_edges)
            matcher = self._make_matcher(mapp)
            matcher.match([tuple(p) for p in pts])
            return self._project_lattice(mapp, ts, lat, lon, matcher.lattice_best)
        except Exception:
            return None

    def _project_lattice(self, mapp, ts: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                         lattice_best) -> SnappedTrack | None:
        """Snap every matched observation to its position on the road.

        lattice_best holds the best lattice path; states carry ``obs`` (the
        observation index) and ``edge_m.pi`` (the projected point on the map
        edge). Non-emitting states in between are ignored.
        """
        if not lattice_best:
            return None
        best: dict[int, tuple[float, float]] = {}
        for st in lattice_best:
            oi = getattr(st, "obs", None)
            if oi is None or oi >= len(ts):
                continue
            seg = getattr(st, "edge_m", None)
            pi = getattr(seg, "pi", None)
            if pi is not None:
                best[int(oi)] = (float(pi[0]), float(pi[1]))
        if not best:
            return None
        out_t: list[float] = []
        out_lat: list[float] = []
        out_lon: list[float] = []
        for i in range(len(ts)):
            if i not in best:
                continue
            out_t.append(float(ts[i]))
            out_lat.append(best[i][0])
            out_lon.append(best[i][1])
        if len(out_t) < 2:
            return None
        return SnappedTrack(out_t, out_lat, out_lon, float(ts[-1]))

    def snap_track(self, tr_id: int, ts: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                   allow_compute: bool = True):
        """Return (snapped_track, fresh) or (None, False). Cached per vehicle.

        A fresh HMM run happens only when new telemetry arrived and at least
        ``min_gap_s`` (or the current backoff gap) of stream time passed since
        the last attempt; otherwise the cached track is reused as-is.
        """
        st = self._state.get(tr_id)
        if st is None:
            st = _VehState(self.min_gap_s)
            self._state[tr_id] = st
        last_ts = float(ts[-1])
        if last_ts <= st.src_ts:
            return st.snapped, False
        if last_ts - st.src_ts < st.gap:
            return st.snapped, False
        if not allow_compute:
            return st.snapped, False

        t0 = time.perf_counter()
        w = ts >= last_ts - self.track_window_s
        sw = ts[w]
        slat, slon = lat[w], lon[w]
        if len(sw) < 3:
            return None, False
        ts_d, lat_d, lon_d = self._decimate(sw, slat, slon)
        snapped = self._match_track(ts_d, lat_d, lon_d)
        dt = time.perf_counter() - t0

        st.src_ts = last_ts
        if snapped is None:
            st.snapped = None
            st.gap = max(st.gap * 2.0, 600.0)
            return None, False
        st.snapped = snapped
        st.gap = self.min_gap_s if dt <= 0.6 else min(max(self.min_gap_s, st.gap) * 2.0, 1800.0)
        return snapped, True

    # -- downstream estimates ------------------------------------------------

    def stop_delays(self, snapped: SnappedTrack, stop_ids: np.ndarray,
                    stop_plan_s: np.ndarray, stop_lat: np.ndarray, stop_lon: np.ndarray) -> dict[int, float]:
        """tt_action_item_id -> delay estimate from the snapped track.

        Mirrors and refines reconstruct_stop_matches semantics: within
        radius/window of the planned arrival take the pass time closest to the
        plan. Because the snapped track is decimated, the pass time is also
        interpolated where the track segment crosses the stop's neighbourhood,
        which recovers sub-sampling time resolution.
        """
        st = np.asarray(snapped.ts, dtype=np.float64)
        slat = np.asarray(snapped.lat, dtype=np.float64)
        slon = np.asarray(snapped.lon, dtype=np.float64)
        out: dict[int, float] = {}
        n = len(st)
        if n == 0:
            return out
        lat0 = float(np.median(slat))
        lon_scale = _lon_m(lat0)
        x = (slon - float(np.median(slon))) * lon_scale
        y = (slat - lat0) * _LAT_M
        dx = np.diff(x)
        dy = np.diff(y)
        seg_len2 = dx * dx + dy * dy
        valid_seg = seg_len2 > 1e-9
        for sid, plan, slat_p, slon_p in zip(stop_ids, stop_plan_s, stop_lat, stop_lon, strict=False):
            if not (np.isfinite(plan) and np.isfinite(slat_p) and np.isfinite(slon_p)):
                continue
            plan = float(plan)
            px = (float(slon_p) - float(np.median(slon))) * lon_scale
            py = (float(slat_p) - lat0) * _LAT_M
            cands: list[float] = []
            # observations inside the stop neighbourhood
            d_obs = np.hypot(x - px, y - py)
            m = (d_obs <= self.radius_m) & (np.abs(st - plan) <= self.window_s)
            cands.extend(st[m] - plan)
            # track segments whose projection lands at the stop
            if valid_seg.any():
                with np.errstate(divide="ignore", invalid="ignore"):
                    tj = ((px - x[:-1]) * dx + (py - y[:-1]) * dy) / seg_len2
                ok = valid_seg & (tj >= 0.0) & (tj <= 1.0)
                if ok.any():
                    qx = x[:-1] + tj * dx
                    qy = y[:-1] + tj * dy
                    d_seg = np.hypot(px - qx, py - qy)
                    ok &= d_seg <= self.radius_m
                    idxs = np.nonzero(ok)[0]
                    for j in idxs:
                        t_pass = st[j] + tj[j] * (st[j + 1] - st[j])
                        if abs(t_pass - plan) <= self.window_s:
                            cands.append(t_pass - plan)
            if not cands:
                continue
            out[int(sid)] = float(min(cands, key=abs))
        return out

    def stuck_street(self, snapped: SnappedTrack, now_s: float):
        """(street_name, stuck_seconds) for the longest low-speed run, or None."""
        st = np.asarray(snapped.ts, dtype=np.float64)
        slat = np.asarray(snapped.lat, dtype=np.float64)
        slon = np.asarray(snapped.lon, dtype=np.float64)
        mask = st >= float(now_s) - self.stuck_window_s
        st, slat, slon = st[mask], slat[mask], slon[mask]
        if len(st) < 3:
            return None
        dlat = np.diff(slat) * _LAT_M
        dlon = np.diff(slon) * _lon_m(float(np.median(slat)))
        dist = np.hypot(dlat, dlon)
        dt = np.diff(st)
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = np.where(dt > 1e-6, dist / dt, np.inf)
        stuck = speed < self.stuck_speed_mps
        best = None
        i = 0
        n = len(stuck)
        while i < n:
            if stuck[i]:
                j = i
                while j + 1 < n and stuck[j + 1]:
                    j += 1
                dur = float(st[j + 1] - st[i])
                if best is None or dur > best[0]:
                    best = (dur, i, j + 1)
                i = j + 1
            else:
                i += 1
        if best is None or best[0] < self.stuck_min_s:
            return None
        dur, a, b = best
        mid = (a + b) // 2
        name = self.graph.nearest_named_street(float(slat[mid]), float(slon[mid]), max_m=150.0)
        if name is None:
            return None
        return name, round(dur, 1)
