"""NDTP binary protocol: CRC-16/Modbus, frame parser, G6CellNav00 decoding.

Frame: [ NPL 15 bytes ][ NPH 10 bytes ][ cells ]
All little-endian, packed.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import UTC, datetime

NPL_SIG = 0x7E7E
NPL_TYPE_NPH = 0x02
NPH_TYPE_REALTIME = 101
CELL_NAV00 = 0

_NPL_HDR = struct.Struct("<HHHHBIH")
_NPH_HDR = struct.Struct("<HHHI")
_NAV_TS = struct.Struct("<I")
_NAV00 = struct.Struct("<II BB HHH HH BB")


def crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


@dataclass(slots=True)
class NavRecord:
    unit_id: int
    ts: datetime
    lat: float
    lon: float
    location_valid: bool
    speed: float
    heading: float
    altitude: float
    nsat: int


def decode_nav00(unit_id: int, timestamp: int, payload: bytes) -> NavRecord | None:
    if len(payload) < 26:
        return None
    lon_raw, lat_raw, dop, _bat, sp_avg, _sp_max, course, _track, alt, nsat, _pdop = _NAV00.unpack_from(payload, 4)
    lon = (lon_raw / 1e7) * (1.0 if (dop >> 6) & 1 else -1.0)
    lat = (lat_raw / 1e7) * (1.0 if (dop >> 5) & 1 else -1.0)
    valid = bool((dop >> 7) & 1)
    return NavRecord(
        unit_id=unit_id,
        ts=datetime.fromtimestamp(timestamp, tz=UTC),
        lat=lat,
        lon=lon,
        location_valid=valid,
        speed=float(sp_avg),
        heading=float(course % 360),
        altitude=float(alt),
        nsat=nsat,
    )


class NDTPFrameParser:
    """Incremental byte-stream parser: feed() bytes, get list of NavRecord."""

    def __init__(self, max_frame: int = 65535 + 25):
        self._buf = bytearray()
        self.max_frame = max_frame

    def feed(self, data: bytes) -> list[NavRecord]:
        self._buf.extend(data)
        out: list[NavRecord] = []
        while True:
            frame = self._try_pop_frame()
            if frame is None:
                break
            out.extend(self._parse_frame(frame))
        if len(self._buf) > self.max_frame * 4:
            self._buf.clear()
        return out

    def _try_pop_frame(self) -> bytes | None:
        buf = self._buf
        sig = struct.pack("<H", NPL_SIG)
        search_from = 0
        incomplete_pos: int | None = None
        while True:
            pos = buf.find(sig, search_from)
            if pos < 0:
                break
            if len(buf) - pos < 25:
                incomplete_pos = pos if incomplete_pos is None else min(incomplete_pos, pos)
                break
            _sig, data_size, _flags, crc, _type, _peer, _req = _NPL_HDR.unpack_from(buf, pos)
            total = 15 + data_size
            if data_size < 10 or total > self.max_frame:
                search_from = pos + 1
                continue
            if len(buf) - pos < total:
                incomplete_pos = pos if incomplete_pos is None else min(incomplete_pos, pos)
                search_from = pos + 1
                continue
            frame = bytes(buf[pos : pos + total])
            expected = crc16_modbus(frame[15:])
            swapped = ((crc & 0xFF) << 8) | ((crc >> 8) & 0xFF)
            if expected != swapped:
                search_from = pos + 1
                continue
            del buf[: pos + total]
            return frame
        if incomplete_pos is not None:
            del buf[:incomplete_pos]
        else:
            del buf[: max(len(buf) - 1, 0)]
        return None

    def _parse_frame(self, frame: bytes) -> list[NavRecord]:
        npl = frame[:15]
        nph_and_body = frame[15:]
        expected = crc16_modbus(nph_and_body)
        got = _NPL_HDR.unpack_from(npl, 0)[3]
        swapped = ((got & 0xFF) << 8) | ((got >> 8) & 0xFF)
        if expected != swapped:
            return []
        _service_id, nph_type, _flags, _req = _NPH_HDR.unpack_from(nph_and_body, 0)
        if nph_type != NPH_TYPE_REALTIME:
            return []
        peer = _NPL_HDR.unpack_from(npl, 0)[5]
        body = nph_and_body[10:]
        return self._parse_cells(peer, body)

    def _parse_cells(self, peer: int, body: bytes) -> list[NavRecord]:
        out: list[NavRecord] = []
        pos = 0
        while pos + 2 <= len(body):
            ctype = body[pos]
            _number = body[pos + 1]
            payload_len = self._cell_payload_len(ctype)
            if payload_len is None:
                break
            payload = body[pos + 2 : pos + 2 + payload_len]
            if len(payload) < payload_len:
                break
            if ctype == CELL_NAV00:
                if len(payload) < 4:
                    break
                timestamp = struct.unpack_from("<I", payload, 0)[0]
                rec = decode_nav00(peer, timestamp, payload)
                if rec is not None:
                    out.append(rec)
            pos += 2 + payload_len
        return out

    @staticmethod
    def _cell_payload_len(ctype: int) -> int | None:
        return {0: 26, 2: 26, 8: 6, 10: 37, 16: 8, 15: 50}.get(ctype)
