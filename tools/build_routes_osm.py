"""Builds street-following route polylines from OSM (Stage 1 map matching).

Downloads the OSM road graph for the routes bbox (cached), chains consecutive
planned stops of every vehicle via shortest paths over roads, and writes a
GeoJSON consumed by the /routes endpoint. Falls back to straight segments
per stop pair when the road graph has no path.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mtp.gt import load_schedule
from mtp.osm import RoadRouter, fetch_roads


def bbox_key(bbox) -> str:
    return "_".join(str(round(v, 1)).replace(".", "-").replace("-", "m") for v in bbox)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default="dataset/validate/schedule_plan.csv")
    ap.add_argument("--out", default="data/osm/routes_osm.geojson")
    ap.add_argument("--pad-deg", type=float, default=0.02)
    args = ap.parse_args()

    schedule = load_schedule(args.schedule)
    stops = schedule.dropna(subset=["stop_lat", "stop_lon"])
    min_lat, max_lat = float(stops.stop_lat.min()), float(stops.stop_lat.max())
    min_lon, max_lon = float(stops.stop_lon.min()), float(stops.stop_lon.max())
    pad = args.pad_deg
    bbox = (min_lat - pad, min_lon - pad, max_lat + pad, max_lon + pad)
    print(f"[routes-osm] bbox {bbox}, vehicles: {schedule.tr_id.nunique()}")

    cache = Path("data/osm") / f"roads_{bbox_key(bbox)}.json"
    data = fetch_roads(bbox, cache)
    router = RoadRouter(data)
    print(f"[routes-osm] graph: {router.G.number_of_nodes()} nodes, {router.G.number_of_edges()} edges")

    features = []
    fallback_segments = 0
    total_segments = 0
    for tr_id, group in schedule.dropna(subset=["stop_lat", "stop_lon"]).groupby("tr_id"):
        group = group.sort_values("plan_s")
        coords = []
        prev = None
        for row in group.itertuples(index=False):
            lat, lon = float(row.stop_lat), float(row.stop_lon)
            if prev is not None:
                total_segments += 1
                seg = router.route_between(prev[0], prev[1], lat, lon)
                if seg is None:
                    seg = [(prev[0], prev[1]), (lat, lon)]
                    fallback_segments += 1
                coords.extend([c for c in seg if not coords or c != coords[-1]])
            else:
                coords.append((lat, lon))
            prev = (lat, lon)
        if len(coords) < 2:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "tr_id": int(tr_id),
                    "stops": len(group),
                    "osm": True,
                    "fallback_segments": fallback_segments,
                },
                "geometry": {"type": "LineString", "coordinates": [[round(lo, 6), round(la, 6)] for la, lo in coords]},
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    straight_pct = 100.0 * fallback_segments / max(total_segments, 1)
    print(f"[routes-osm] {len(features)} routes -> {out} (straight-fallback segments: {fallback_segments}/{total_segments} = {straight_pct:.1f}%)")


if __name__ == "__main__":
    main()
