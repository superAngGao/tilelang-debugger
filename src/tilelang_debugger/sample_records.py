"""Versioned samples and conservative source-domain coverage validation."""
import itertools
import math

from .records import value
from .sample_monitor import WIDTH


def parse(log, points):
    specs = {p["id"]: p for p in points}
    events = {p: {} for p in specs}
    for line in log.splitlines():
        if "TLDBG" not in line:
            continue
        fields = line.split("|")
        if fields[-1] != "E":
            raise ValueError("truncated samples event: missing terminator")
        fields.pop()
        if len(fields) < 12 or fields[0] != "TLDBG2" or fields[1] not in specs or fields[2] not in {"R", "B", "D"}:
            raise ValueError("malformed/unknown samples record")
        pid, kind = fields[1:3]
        p = specs[pid]
        try:
            scope, launch, bx, by, bz, tx, *tail = map(int, fields[3:])
        except ValueError as exc:
            raise ValueError("malformed/truncated samples fields") from exc
        if launch != 0 or [bx, by, bz] != p["block"] or not 0 <= tx < p["threads"] or (p["thread"] is not None and tx != p["thread"]):
            raise ValueError("sample launch/block/thread outside selection")
        loops = [s for s in p["scopes"] if s["kind"] != "branch"]
        bounds = [b for s in loops for b in s["actual_bounds"]]
        if kind == "B":
            if not 0 <= scope < len(p["scopes"]) or p["scopes"][scope]["kind"] != "branch":
                raise ValueError("invalid branch scope")
            depth = p["scopes"][scope]["depth"]
        else:
            if scope != -1:
                raise ValueError("invalid root/data scope")
            depth = 0 if kind == "R" else len(bounds)
        if len(tail) != depth + 3:
            raise ValueError("wrong sample coordinate arity")
        coords, index, low, high = tuple(tail[:depth]), *tail[depth:]
        if any(v not in range(*b) for v, b in zip(coords, bounds)):
            raise ValueError("coordinate outside source loop domain")
        if not 0 <= low < 2**32 or not 0 <= high < 2**32:
            raise ValueError("invalid raw word")
        bits = low | (high << 32)
        if kind == "R" and (index or bits) or kind == "B" and (index not in {0, 1} or bits):
            raise ValueError("invalid witness payload")
        if kind == "D":
            if not p.get("bound") or not 0 <= index < p["elements"] or bits >= 2**WIDTH[p["dtype"]]:
                raise ValueError("invalid sample element/bit width")
            for s in loops:
                if "selected_values" in s and list(coords[s["depth"]:s["depth"] + len(s["names"])]) != s["selected_values"]:
                    raise ValueError("data outside selected iteration")
        # A branch may choose either arm, but cannot emit both for the same instance.
        key = (kind, scope, tx, coords, index if kind == "D" else 0)
        if key in events[pid]:
            raise ValueError("duplicate or contradictory sample/witness")
        events[pid][key] = (index, bits)
        if len(events[pid]) > p["event_budget"]:
            raise ValueError("sample event budget exceeded")
    records, coverage = [], {}
    for pid, p in specs.items():
        found = events[pid]
        threads = [p["thread"]] if p["thread"] is not None else list(range(p["threads"]))
        roots = {key[2] for key in found if key[0] == "R"}
        if roots != set(threads):
            raise ValueError(f"missing root witnesses: {pid}")
        data = {key: bits for key, (_, bits) in found.items() if key[0] == "D"}
        for key, (index, _) in found.items():
            kind, scope, tx, coords, _ = key
            if kind == "R":
                continue
            for i, s in enumerate(p["scopes"]):
                if kind == "B" and i >= scope:
                    break
                if s["kind"] == "branch":
                    ancestor = found.get(("B", i, tx, coords[:s["depth"]], 0))
                    if ancestor is None or ancestor[0] != s["arm"]:
                        raise ValueError("event lacks its own ancestor branch witness")
        instances = {}
        for key in data:
            instances.setdefault((key[2], key[3]), set()).add(key[4])
        if any(indices != set(range(p["elements"])) for indices in instances.values()):
            raise ValueError("incomplete local-buffer sample")
        parallel = any(s["kind"] == "parallel" for s in p["scopes"])
        branches = any(s["kind"] == "branch" for s in p["scopes"])
        logical = "unverified" if parallel and (branches or p["thread"] is not None) else "complete"
        expected = set()
        if logical == "complete":
            def walk(position, coords, tx, selected=True):
                if position == len(p["scopes"]):
                    if selected:
                        for idx in range(p.get("elements", 1)):
                            expected.add((tx, coords, idx))
                    return
                s = p["scopes"][position]
                if s["kind"] == "branch":
                    witness = found.get(("B", position, tx, coords, 0))
                    if witness is None:
                        raise ValueError("missing branch witness in closed domain")
                    if witness[0] == s["arm"]:
                        walk(position + 1, coords, tx, selected)
                else:
                    for part in itertools.product(*(range(*b) for b in s["actual_bounds"])):
                        chosen = "selected_values" not in s or list(part) == s["selected_values"]
                        walk(position + 1, coords + part, tx, selected and chosen)
            for tx in ([None] if parallel else threads):
                walk(0, (), tx)
            actual = {(None if parallel else key[2], key[3], key[4]) for key in data}
            if actual != expected:
                raise ValueError(f"missing/unexpected sample instance in closed domain: {pid}")
        coverage[pid] = dict(capture_integrity="complete", logical_coverage=logical,
                             execution_coverage="unverified" if parallel else "complete",
                             point_execution="observed" if data else ("not_executed" if logical == "complete" else "unverified"),
                             build_status="active" if p.get("bound") else "inactive_at_build")
        for key, bits in sorted(data.items()):
            records.append(dict(schema=2, point=pid, launch=0, block=p["block"], thread=key[2],
                                coordinates=list(key[3]), index=key[4], bits=bits, dtype=p["dtype"]))
    return records, coverage
