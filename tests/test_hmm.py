import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mtp.matching.hmm import HMMTrackMatcher, RoadGraphIndex

_LAT_M = 111_132.0
_STREET = "Тестовая улица"


def _lon_m(lat0):
    return 111_320.0 * math.cos(math.radians(lat0))


def make_overpass(name=_STREET):
    """Straight N-S street at lon 37.600 plus a parallel one at lon 37.62."""
    nodes = []
    main_ids = []
    for i in range(21):
        nid = 1000 + i
        lat = 55.7500 + i * 0.0009
        nodes.append({"type": "node", "id": nid, "lat": round(lat, 6), "lon": 37.6000})
        main_ids.append(nid)
    side_ids = []
    for i in range(21):
        nid = 2000 + i
        lat = 55.7500 + i * 0.0009
        nodes.append({"type": "node", "id": nid, "lat": round(lat, 6), "lon": 37.6200})
        side_ids.append(nid)
    ways = [
        {"type": "way", "id": 10, "nodes": main_ids, "tags": {"highway": "residential", "name": name}},
        {"type": "way", "id": 11, "nodes": side_ids, "tags": {"highway": "residential"}},
    ]
    return {"elements": nodes + ways}


def make_matcher(tmp_path):
    roads = tmp_path / "roads_test.json"
    roads.write_text(__import__("json").dumps(make_overpass()), encoding="utf-8")
    return HMMTrackMatcher(roads, min_gap_s=0.0)


def moving_track(t0=1_700_000_000.0, speed_mps=6.0, n=30, dt=10.0, lat0=55.7500, jitter=0.00004):
    ts, lat, lon = [], [], []
    rng = np.random.default_rng(7)
    for i in range(n):
        ts.append(t0 + i * dt)
        lat.append(lat0 + speed_mps * i * dt / _LAT_M + (rng.uniform(-1, 1) * jitter if i else 0.0))
        lon.append(37.6000 + (rng.uniform(-1, 1) * jitter if i else 0.0))
    return np.array(ts), np.array(lat), np.array(lon)


def test_road_graph_index_build_and_cache(tmp_path):
    data = make_overpass()
    idx = RoadGraphIndex.from_overpass(data)
    assert len(idx.coords) == 42
    assert len(idx.edges) >= 40
    assert len(idx.way_names) == 1
    assert str(idx.way_names[0]) == _STREET
    assert idx.nearest_named_street(55.7550, 37.6002, max_m=150.0) == _STREET
    assert idx.nearest_named_street(55.7550, 37.6200, max_m=150.0) is None

    # compact cache round-trip
    roads = tmp_path / "roads.json"
    import json

    roads.write_text(json.dumps(data), encoding="utf-8")
    m1 = RoadGraphIndex.load(roads)
    compact = roads.with_name("roads.compact.npz")
    assert compact.exists()
    m2 = RoadGraphIndex.load(roads)
    assert np.allclose(m1.coords, m2.coords)
    assert np.array_equal(m1.edges, m2.edges)


def test_snap_track_stays_on_street(tmp_path):
    matcher = make_matcher(tmp_path)
    ts, lat, lon = moving_track()
    snapped, fresh = matcher.snap_track(1, ts, lat, lon, allow_compute=True)
    assert fresh and snapped is not None
    assert len(snapped) >= 3
    # snapped points must lie on the street (lon ~37.6000)
    lon_arr = np.asarray(snapped.lon)
    off_m = np.abs(lon_arr - 37.6000) * _lon_m(55.76)
    assert float(off_m.max()) < 40.0
    # progress must be monotone-ish and match the driven distance
    d = np.diff(np.asarray(snapped.lat)) * _LAT_M
    assert float(np.sum(d)) > 0.9 * speed_m_total(speed=6.0, n=30, dt=10.0)
    # second call with same data: served from cache, no recompute
    snapped2, fresh2 = matcher.snap_track(1, ts, lat, lon, allow_compute=True)
    assert not fresh2 and snapped2 is snapped


def speed_m_total(speed, n, dt):
    return speed * (n - 1) * dt


def test_stop_delays_mirror_radius_semantics(tmp_path):
    matcher = make_matcher(tmp_path)
    ts, lat, lon = moving_track()
    snapped, _fresh = matcher.snap_track(11, ts, lat, lon, allow_compute=True)
    assert snapped is not None
    # stop the vehicle passes ~at t0 + 150 s (6 m/s * 150 s = 900 m along the street)
    plan = float(ts[0]) + 150.0
    stop_lat = 55.7500 + 6.0 * 150.0 / _LAT_M
    delays = matcher.stop_delays(snapped, np.array([9001]), np.array([plan]), np.array([stop_lat]), np.array([37.6000]))
    assert 9001 in delays
    assert abs(delays[9001]) <= 15.0  # moving on time
    # a stop far away from the street must not match
    delays_far = matcher.stop_delays(snapped, np.array([9002]), np.array([plan]), np.array([stop_lat]), np.array([37.6500]))
    assert 9002 not in delays_far


def test_stuck_street_detected(tmp_path):
    matcher = make_matcher(tmp_path)
    ts, lat, lon = moving_track(n=30)
    # vehicle stuck for 300 s with small jitter
    rng = np.random.default_rng(3)
    base_t, base_lat, base_lon = ts[-1], lat[-1], lon[-1]
    for i in range(30):
        ts = np.append(ts, base_t + 10.0 * (i + 1))
        lat = np.append(lat, base_lat + rng.uniform(-1, 1) * 0.00002)
        lon = np.append(lon, base_lon + rng.uniform(-1, 1) * 0.00002)
    snapped, fresh = matcher.snap_track(21, ts, lat, lon, allow_compute=True)
    assert fresh and snapped is not None
    info = matcher.stuck_street(snapped, float(ts[-1]))
    assert info is not None
    name, dur = info
    assert name == _STREET
    assert dur >= matcher.stuck_min_s


def test_no_stuck_for_moving_vehicle(tmp_path):
    matcher = make_matcher(tmp_path)
    ts, lat, lon = moving_track(n=40)
    snapped, _ = matcher.snap_track(31, ts, lat, lon, allow_compute=True)
    assert snapped is not None
    assert matcher.stuck_street(snapped, float(ts[-1])) is None
