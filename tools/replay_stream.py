"""Replays validate/test traffic.csv as a live NDTP stream (demo & integration)."""
from __future__ import annotations

import argparse
import asyncio
import struct
import sys
import time
from contextlib import suppress
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def build_realtime_frame(unit_id: int, timestamp: int, lat: float, lon: float, speed: float, heading: float, valid: bool = True) -> bytes:
    lon_raw = int(abs(lon) * 1e7)
    lat_raw = int(abs(lat) * 1e7)
    dop = 0b11100000 if valid else 0
    nav_payload = struct.pack(
        "<III BB HHH HH BB",
        int(timestamp), lon_raw, lat_raw, dop, 0x0C,
        int(speed), int(speed), int(heading) % 360, 0, 150, 9, 20,
    )
    cells = bytes([0, 0]) + nav_payload
    nph = struct.pack("<HHHI", 1, 101, 0, 1)
    payload = nph + cells
    from mtp.ndtp import crc16_modbus

    crc = crc16_modbus(payload)
    crc_swapped = ((crc & 0xFF) << 8) | ((crc >> 8) & 0xFF)
    npl = struct.pack("<HHHHBIH", 0x7E7E, len(payload), 0, crc_swapped, 0x02, unit_id, 0)
    return npl + payload


async def run(args) -> None:
    df = pd.read_csv(args.traffic, parse_dates=["event_time"], low_memory=False)
    df = df.sort_values("event_time")
    df["ts"] = df["event_time"].astype("datetime64[ns]").astype("int64") // 10**9
    t0 = df["ts"].min()
    if args.start_offset:
        df = df[df["ts"] >= t0 + args.start_offset]
    if args.duration:
        floor = df["ts"].min() if args.start_offset else t0
        df = df[df["ts"] <= floor + args.duration]
    print(f"[replay] {len(df)} records -> {args.host}:{args.port}, speed x{args.speed}")

    play = 0
    writer = None
    while True:
        if writer is None or writer.is_closing():
            writer = None
            for _attempt in range(60):
                try:
                    _, writer = await asyncio.open_connection(args.host, args.port)
                    break
                except OSError:
                    await asyncio.sleep(3)
            if writer is None:
                print("[replay] target unreachable, retrying")
                await asyncio.sleep(5)
                continue

        play += 1
        base_wall = time.monotonic()
        first_ts = None
        batch, last_flush = [], time.monotonic()
        try:
            for row in df.itertuples(index=False):
                ts = int(row.ts)
                if first_ts is None:
                    first_ts = ts
                target = base_wall + (ts - first_ts) / args.speed
                now = time.monotonic()
                if target > now:
                    await asyncio.sleep(target - now)
                speed = 0.0 if pd.isna(row.speed) else float(row.speed)
                heading = 0.0 if pd.isna(row.heading) else float(row.heading)
                valid = bool(row.location_valid) and not (pd.isna(row.lat) or pd.isna(row.lon))
                lat = 0.0 if pd.isna(row.lat) else float(row.lat)
                lon = 0.0 if pd.isna(row.lon) else float(row.lon)
                batch.append(build_realtime_frame(int(row.tr_id), ts, lat, lon, speed, heading, valid=valid))
                if len(batch) >= 200 or time.monotonic() - last_flush > 0.5:
                    if batch:
                        writer.write(b"".join(batch))
                        await writer.drain()
                    batch, last_flush = [], time.monotonic()
            if batch:
                writer.write(b"".join(batch))
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError, OSError) as exc:
            print(f"[replay] connection lost ({exc}), reconnecting")
            with suppress(OSError):
                writer.close()
            writer = None
            await asyncio.sleep(3)
            continue
        print(f"[replay] pass {play} finished")
        if not args.loop:
            writer.close()
            break
        await asyncio.sleep(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traffic", default="dataset/validate/traffic.csv")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9201)
    ap.add_argument("--speed", type=float, default=600.0)
    ap.add_argument("--duration", type=float, default=None, help="limit replayed seconds of sim time")
    ap.add_argument("--start-offset", type=float, default=None, help="skip to sim second t0+offset")
    ap.add_argument("--loop", action="store_true", help="repeat the replay endlessly (demo mode)")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
