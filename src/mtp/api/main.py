"""Backend API + ML-core wiring.

Modes:
  --combined  : ML-core runs in-process (local dev)
  default     : backend only; ML-core is expected at ML_URL (docker split)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import httpx
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ..config import ROOT
from ..ndtp_server import NDTPServer, nav_to_pipeline
from ..online import OnlinePipeline

ML_URL = os.environ.get("ML_URL", "http://ml-core:8100")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")

BASE = Path(os.environ.get("APP_DIR") or Path.cwd())
DASH = BASE / "dashboard"
PYDOC = BASE / "docs" / "pydoc"
DATA_DEFAULT = ROOT / "dataset" / "validate"


def build_ml_app(base_dir: Path | None = None) -> FastAPI:
    base_dir = Path(base_dir or os.environ.get("DATA_DIR") or DATA_DEFAULT)
    schedule = base_dir / "schedule_plan.csv"
    if not schedule.exists():
        schedule = base_dir / "schedule.csv"
    model = BASE / "data" / "gt" / "models" / "catboost_stream.cbm"
    if not model.exists():
        model = BASE / "data" / "gt" / "models" / "catboost_final.cbm"
    if not model.exists():
        model = BASE / "data" / "gt" / "models" / "catboost_mae.cbm"
    priors = BASE / "data" / "gt" / "stop_priors.parquet"

    pipeline = OnlinePipeline(
        str(schedule), str(model),
        priors_path=str(priors) if priors.exists() else None,
        use_hmm_matching=_env_flag("MTP_HMM_MATCHING", True),
        hmm_roads_path=os.environ.get("MTP_HMM_ROADS"),
    )
    pipeline.use_est_dev_as_cur = True
    server = NDTPServer("0.0.0.0", 9201, lambda rec: nav_to_pipeline(rec, pipeline))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await server.start()
        tick = asyncio.create_task(_tick_loop(pipeline))
        serve = asyncio.create_task(server.serve_forever())
        yield
        for t in (tick, serve):
            t.cancel()

    app = FastAPI(title="ML core", lifespan=lifespan)

    @app.get("/internal/health")
    async def health():
        return {"ok": True, "stats": pipeline.stats, "ndtp_connections": server.connections}

    @app.get("/internal/snapshot")
    async def snapshot():
        pipeline.tick()
        return {"ts": pipeline.snapshot_ts, "vehicles": pipeline.last_snapshot, "alerts": pipeline.alerts(), "stats": pipeline.stats}

    @app.post("/internal/whatif")
    async def whatif(payload: dict):
        pipeline.tick()
        return pipeline.whatif(
            int(payload.get("extra_vehicles", 0)),
            float(payload.get("dwell_reduction_s", 0.0)),
        )

    @app.post("/internal/predict")
    async def predict_one(payload: dict):
        tr_id = int(payload["tr_id"])
        now_s = float(payload.get("now_s") or _max_ts(pipeline))
        row = pipeline.predict_vehicle(tr_id, now_s)
        return row or {"error": "no data for vehicle"}

    return app


def _max_ts(pipeline: OnlinePipeline) -> float:
    return max((r[0] for v in pipeline.vehicles.values() for r in v.records), default=0.0)


@lru_cache(maxsize=4)
def _stop_features_cached(sched_path: str, mtime_ns: int) -> tuple:
    """Point features for every scheduled stop (labels for the route lines)."""
    from ..gt import load_schedule

    try:
        sc = load_schedule(sched_path)
    except (FileNotFoundError, ValueError):
        return ()
    out = []
    for tr_id, g in sc.dropna(subset=["stop_lat", "stop_lon"]).groupby("tr_id"):
        g = g.sort_values("plan_s")
        for i, r in enumerate(g.itertuples(index=False)):
            name = getattr(r, "building_address", None)
            out.append(
                {
                    "type": "Feature",
                    "properties": {
                        "kind": "stop",
                        "tr_id": int(tr_id),
                        "stop_id": int(r.tt_action_item_id),
                        "order": i,
                        "name": None if pd.isna(name) else str(name),
                    },
                    "geometry": {"type": "Point", "coordinates": [float(r.stop_lon), float(r.stop_lat)]},
                }
            )
    return tuple(out)


def _stop_features() -> list:
    data_dir = Path(os.environ.get("DATA_DIR") or DATA_DEFAULT)
    sched_path = data_dir / "schedule_plan.csv"
    if not sched_path.exists():
        sched_path = data_dir / "schedule.csv"
    if not sched_path.exists():
        return []
    try:
        return list(_stop_features_cached(str(sched_path), int(sched_path.stat().st_mtime_ns)))
    except OSError:
        return []


async def _tick_loop(pipeline: OnlinePipeline, interval: float = 5.0):
    while True:
        try:
            pipeline.tick()
        except Exception as exc:
            print(f"[ml] tick error: {exc}")
        await asyncio.sleep(interval)


def build_backend_app(ml_url: str | None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(timeout=10.0)
        yield
        await app.state.client.aclose()

    app = FastAPI(
        title="Предиктор задержек — диспетчерский API",
        description="Backend: прием потока NDTP, ML-инференс, данные для BI-дашборда.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def no_store_middleware(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/dashboard"):
            response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response

    async def _fetch(app: FastAPI, path: str) -> dict:
        if ml_url is None:
            ml_app = app.state.ml_app
            import httpx as _hx

            transport = _hx.ASGITransport(app=ml_app)
            async with _hx.AsyncClient(transport=transport, base_url="http://ml") as c:
                resp = await c.get(path)
                return resp.json()
        resp = await app.state.client.get(ml_url + path)
        return resp.json()

    @app.get("/health")
    async def health():
        return await _fetch(app, "/internal/health")

    @app.get("/snapshot")
    async def snapshot():
        return await _fetch(app, "/internal/snapshot")

    @app.get("/alerts")
    async def alerts():
        data = await _fetch(app, "/internal/snapshot")
        return {"ts": data.get("ts"), "alerts": data.get("alerts", [])}

    @app.post("/whatif")
    async def whatif(payload: dict):
        if ml_url is None:
            ml_app = app.state.ml_app
            import httpx as _hx

            transport = _hx.ASGITransport(app=ml_app)
            async with _hx.AsyncClient(transport=transport, base_url="http://ml") as c:
                resp = await c.post("/internal/whatif", json=payload)
                return resp.json()
        resp = await app.state.client.post(ml_url + "/internal/whatif", json=payload)
        return resp.json()

    @app.get("/routes")
    async def routes():
        osm_file = BASE / "data" / "osm" / "routes_osm.geojson"
        features: list = []
        if osm_file.exists():
            features = list(json.loads(osm_file.read_text(encoding="utf-8")).get("features", []))
        else:
            data_dir = Path(os.environ.get("DATA_DIR") or DATA_DEFAULT)
            sched_path = data_dir / "schedule_plan.csv"
            if not sched_path.exists():
                sched_path = data_dir / "schedule.csv"
            from ..gt import load_schedule

            try:
                sc = load_schedule(sched_path)
            except (FileNotFoundError, ValueError):
                sc = None
            if sc is not None:
                for tr_id, g in sc.groupby("tr_id"):
                    g = g.dropna(subset=["stop_lat", "stop_lon"]).sort_values("plan_s")
                    if len(g) < 2:
                        continue
                    features.append(
                        {
                            "type": "Feature",
                            "properties": {"tr_id": int(tr_id), "stops": len(g), "osm": False},
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [[float(r.stop_lon), float(r.stop_lat)] for r in g.itertuples(index=False)],
                            },
                        }
                    )
        features.extend(_stop_features())
        return JSONResponse({"type": "FeatureCollection", "features": features})

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                try:
                    data = await _fetch(app, "/internal/snapshot")
                except Exception:
                    data = {"ts": None, "vehicles": [], "alerts": [], "degraded": True}
                await websocket.send_text(json.dumps(data, default=str))
                await asyncio.sleep(2)
        except WebSocketDisconnect:
            return

    if DASH.exists():
        app.mount("/dashboard", StaticFiles(directory=str(DASH), html=True), name="dashboard")

    if PYDOC.exists():
        app.mount("/pydoc", StaticFiles(directory=str(PYDOC), html=True), name="pydoc")

        @app.get("/", include_in_schema=False)
        async def index():
            return RedirectResponse("/dashboard/index.html")

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["backend", "ml", "combined"], default="combined")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ml-port", type=int, default=8100)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    data_dir = args.data_dir or os.environ.get("DATA_DIR")

    import threading

    import uvicorn

    if args.mode in ("ml", "combined"):
        ml_app = build_ml_app(Path(data_dir) if data_dir else None)
    if args.mode == "ml":
        uvicorn.run(ml_app, host=args.host, port=args.ml_port)
        return
    if args.mode == "combined":
        ml_config = uvicorn.Config(ml_app, host="127.0.0.1", port=args.ml_port, log_level="warning")
        ml_server = uvicorn.Server(ml_config)
        thread = threading.Thread(target=ml_server.run, daemon=True)
        thread.start()
        while not ml_server.started:
            threading.Event().wait(0.1)
        backend = build_backend_app(f"http://127.0.0.1:{args.ml_port}")
        uvicorn.run(backend, host=args.host, port=args.port)
        return
    backend = build_backend_app(ML_URL)
    uvicorn.run(backend, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
