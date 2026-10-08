"""Strict unordered record validation. Raw bits are the canonical values."""
import json
import math
import re
import struct

WIDTH = {"float16": 16, "bfloat16": 16, "float32": 32, "int32": 32}
LINE = re.compile(r"TLDBG1\|([A-Za-z][A-Za-z0-9_]{0,31})\|0\|(-?\d+)\|(-?\d+)\|(-?\d+)\|(-?\d+)\|(-?\d+)\|(\d+)\|(\d+)")


def value(bits, dtype):
    if dtype == "int32":
        return struct.unpack("<i", struct.pack("<I", bits))[0]
    if dtype == "bfloat16":
        return struct.unpack("<f", struct.pack("<I", bits << 16))[0]
    if dtype == "float16":
        return struct.unpack("<e", struct.pack("<H", bits))[0]
    if dtype == "float32":
        return struct.unpack("<f", struct.pack("<I", bits))[0]
    raise ValueError(f"unsupported dtype: {dtype}")


def parse(log, points):
    specs = {p["id"]: p for p in points}
    found = {k: {} for k in specs}
    for line in log.splitlines():
        if "TLDBG" not in line:
            continue
        match = LINE.fullmatch(line)
        if not match:
            raise ValueError(f"malformed/truncated record: {line[:160]}")
        point, *fields = match.groups()
        if point not in specs:
            raise ValueError(f"unknown point: {point}")
        bx, by, bz, l0, l1, index, bits = map(int, fields)
        p = specs[point]
        expected_loops = (p["loop_values"] + [0, 0])[:2]
        if [bx, by, bz] != p["block"] or [l0, l1] != expected_loops:
            raise ValueError(f"wrong block/iteration at {point}")
        if not 0 <= index < math.prod(p["shape"]) or not 0 <= bits < 2 ** WIDTH[p["dtype"]]:
            raise ValueError(f"invalid index/bit width at {point}")
        if index in found[point]:
            raise ValueError(f"duplicate element at {point}[{index}]")
        found[point][index] = bits
    records = []
    for name, p in specs.items():
        n = math.prod(p["shape"])
        if len(found[name]) != n:
            raise ValueError(f"incomplete {name}: expected {n}, received {len(found[name])}")
        for i in range(n):
            records.append(dict(schema=1, point=name, launch=0, block=p["block"],
                                loops=p["loop_values"], index=i, bits=found[name][i], dtype=p["dtype"]))
    return records


def write_records(path, records):
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records), encoding="utf-8")
