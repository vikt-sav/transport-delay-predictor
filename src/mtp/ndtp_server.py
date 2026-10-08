"""NDTP TCP server: accepts emulator connections, parses frames, feeds pipeline."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from .ndtp import NavRecord, NDTPFrameParser


class NDTPServer:
    def __init__(self, host: str, port: int, sink):
        self.host = host
        self.port = port
        self.sink = sink
        self.connections = 0
        self._server = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)
        print(f"[ndtp] server listening on {self.host}:{self.port}")

    async def serve_forever(self) -> None:
        async with self._server:
            await self._server.serve_forever()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.connections += 1
        addr = writer.get_extra_info("peername")
        print(f"[ndtp] connection from {addr}")
        parser = NDTPFrameParser()
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                records: list[NavRecord] = parser.feed(data)
                for rec in records:
                    self.sink(rec)
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            self.connections -= 1
            print(f"[ndtp] connection closed {addr}")


def nav_to_pipeline(rec: NavRecord, pipeline) -> None:
    pipeline.add_record(
        tr_id=pipeline.map_tr_id(rec.unit_id),
        ts=pipeline.align_ts(rec.ts.timestamp()),
        lat=rec.lat,
        lon=rec.lon,
        valid=rec.location_valid,
        speed=rec.speed,
        heading=rec.heading,
        altitude=rec.altitude,
    )


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()
