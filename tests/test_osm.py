import sys
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_osm_graph_routing():
    from mtp.osm import RoadRouter

    data = {
        "elements": [
            {"type": "node", "id": 1, "lat": 55.7500, "lon": 37.6000},
            {"type": "node", "id": 2, "lat": 55.7590, "lon": 37.6000},
            {"type": "node", "id": 3, "lat": 55.7680, "lon": 37.6000},
            {"type": "node", "id": 4, "lat": 55.7590, "lon": 37.6100},
            {"type": "way", "id": 10, "nodes": [1, 2, 3]},
            {"type": "way", "id": 11, "nodes": [2, 4]},
        ]
    }
    router = RoadRouter(data)
    path = router.route_between(55.7500, 37.6000, 55.7680, 37.6000)
    assert path is not None and len(path) >= 3
    straight = ((55.7680 - 55.7500) * 111_132.0)
    on_road = sum(
        ((b[0] - a[0]) * 111_132.0) ** 2 + ((b[1] - a[1]) * 111_320.0) ** 2
        for a, b in pairwise(path)
    ) ** 0.5
    assert on_road <= straight * 1.10
    far = router.route_between(55.7500, 37.6000, 56.9000, 39.0000)
    assert far is None or isinstance(far, list)
