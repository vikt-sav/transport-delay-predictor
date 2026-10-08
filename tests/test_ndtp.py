import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mtp.ndtp import NDTPFrameParser, crc16_modbus


def build_frame(unit_id: int, timestamp: int, lat: float, lon: float, speed: int, course: int, valid: bool = True) -> bytes:
    lon_raw = int(abs(lon) * 1e7)
    lat_raw = int(abs(lat) * 1e7)
    dop = 0b11100000 if valid else 0
    nav_payload = struct.pack(
        "<III BB HHH HH BB",
        timestamp, lon_raw, lat_raw, dop, 0x0C,
        speed, speed, course, 0, 150, 9, 20,
    )
    assert len(nav_payload) == 26, len(nav_payload)
    cells = bytes([0, 0]) + nav_payload
    nph = struct.pack("<HHHI", 1, 101, 0, 1)
    payload = nph + cells
    crc = crc16_modbus(payload)
    crc_swapped = ((crc & 0xFF) << 8) | ((crc >> 8) & 0xFF)
    npl = struct.pack("<HHHHBIH", 0x7E7E, len(payload), 0, crc_swapped, 0x02, unit_id, 0)
    return npl + payload


def test_roundtrip():
    ts = 1767745558
    frame = build_frame(1166336, ts, 55.7551234, 37.617321, 34, 271)
    parser = NDTPFrameParser()
    noise = b"\x00garbage\x7e"
    recs = parser.feed(noise + frame[:7])
    recs += parser.feed(frame[7:] + frame)
    assert len(recs) == 2, recs
    r = recs[0]
    assert r.unit_id == 1166336
    assert abs(r.lat - 55.7551234) < 1e-6
    assert abs(r.lon - 37.617321) < 1e-6
    assert r.speed == 34
    assert r.heading == 271
    assert r.location_valid is True
    assert r.ts.timestamp() == ts


def test_invalid_crc_dropped():
    ts = 1767745558
    frame = bytearray(build_frame(1, ts, 55.0, 37.0, 10, 90))
    frame[-3] ^= 0xFF
    parser = NDTPFrameParser()
    assert parser.feed(bytes(frame)) == []


if __name__ == "__main__":
    test_roundtrip()
    test_invalid_crc_dropped()
    print("ndtp parser OK")
