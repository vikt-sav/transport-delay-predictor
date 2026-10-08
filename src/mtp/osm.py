"""OpenStreetMap road-graph builder via Overpass API (Stage 1 map matching).

Fetches drivable roads for a bbox, builds an undirected graph of OSM nodes,
and computes street-following polylines between stops. Cached on disk; falls
back to straight lines if Overpass is unreachable.
"""
from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np

OVERPASS_ENDPOINTS = [
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

HIGHWAY_RE = "^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|living_street|service)$"

_LAT_M = 111_132.0


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def overpass_query(bbox) -> str:
    min_lat, min_lon, max_lat, max_lon = bbox
    return (
        f"[out:json][timeout:180];"
        f'(way["highway"~"{HIGHWAY_RE}"]({min_lat},{min_lon},{max_lat},{max_lon}););'
        f"(._;>;);out body;"
    )


def fetch_roads(bbox: tuple, cache_path: Path, depth: int = 0) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    query = overpass_query(bbox)
    last_err = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            print(f"[osm] querying {endpoint} (bbox {bbox}, depth {depth}) ...")
            import httpx

            resp = httpx.post(endpoint, data={"data": query}, timeout=300.0)
            resp.raise_for_status()
            data = resp.json()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data), encoding="utf-8")
            print(f"[osm] cached {len(data.get('elements', []))} elements -> {cache_path}")
            return data
        except Exception as exc:
            last_err = exc
            print(f"[osm] {endpoint} failed: {exc}")
    if depth < 2:
        min_lat, min_lon, max_lat, max_lon = bbox
        mid_lat, mid_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
        quadrants = [
            (min_lat, min_lon, mid_lat, mid_lon),
            (min_lat, mid_lon, mid_lat, max_lon),
            (mid_lat, min_lon, max_lat, mid_lon),
            (mid_lat, mid_lon, max_lat, max_lon),
        ]
        merged: dict = {"elements": [], "osm_split": True}
        seen = set()
        for i, q in enumerate(quadrants):
            try:
                part = fetch_roads(q, cache_path.with_name(f"{cache_path.stem}_q{i}{cache_path.suffix}"), depth + 1)
            except RuntimeError:
                continue
            for el in part.get("elements", []):
                key = (el.get("type"), el.get("id"))
                if key in seen:
                    continue
                seen.add(key)
                merged["elements"].append(el)
        if merged["elements"]:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(merged), encoding="utf-8")
            print(f"[osm] merged {len(merged['elements'])} elements from quadrants")
            return merged
    raise RuntimeError(f"all Overpass endpoints failed: {last_err}")


def build_graph(overpass_data: dict):
    import networkx as nx

    nodes: dict[int, tuple[float, float]] = {}
    ways: list[list[int]] = []
    for el in overpass_data.get("elements", []):
        if el.get("type") == "node" and "lat" in el:
            nodes[el["id"]] = (el["lat"], el["lon"])
        elif el.get("type") == "way" and "nodes" in el:
            ways.append(el["nodes"])

    G = nx.Graph()
    for way_nodes in ways:
        seq = [n for n in way_nodes if n in nodes]
        for a, b in pairwise(seq):
            if a == b:
                continue
            la1, lo1 = nodes[a]
            la2, lo2 = nodes[b]
            length = haversine_m(la1, lo1, la2, lo2)
            if length <= 0:
                continue
            if not G.has_edge(a, b) or G[a][b]["length"] > length:
                G.add_edge(a, b, length=length)
    return G, nodes


class RoadRouter:
    """Nearest-node lookup + shortest paths over the OSM road graph."""

    def __init__(self, overpass_data: dict):
        import networkx as nx
        from scipy.spatial import cKDTree

        self.G, self.nodes = build_graph(overpass_data)
        self._node_ids = list(self.nodes.keys())
        lats = np.array([self.nodes[n][0] for n in self._node_ids])
        lons = np.array([self.nodes[n][1] for n in self._node_ids])
        lat0 = float(np.median(lats))
        lon0 = float(np.median(lons))
        self._lat0, self._lon0 = lat0, lon0
        self._lon_scale = _lon_m_correct(lat0)
        y = (lats - lat0) * _LAT_M
        x = (lons - lon0) * self._lon_scale
        self._tree = cKDTree(np.column_stack([x, y]))
        self._nx = nx

    def nearest_node(self, lat: float, lon: float):
        x = (lon - self._lon0) * self._lon_scale
        y = (lat - self._lat0) * _LAT_M
        _dist, idx = self._tree.query([x, y])
        return self._node_ids[int(idx)]

    def route_between(self, lat1: float, lon1: float, lat2: float, lon2: float) -> list[tuple[float, float]] | None:
        n1 = self.nearest_node(lat1, lon1)
        n2 = self.nearest_node(lat2, lon2)
        if n1 == n2:
            return [self.nodes[n1], self.nodes[n2]]
        try:
            path = self._nx.shortest_path(self.G, n1, n2, weight="length")
        except Exception:
            return None
        return [self.nodes[n] for n in path]


def _lon_m_correct(lat0: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat0))
