"""Strict, order-independent TLDBG3 validation without GPU dependencies."""
import math
import struct

from .dtypes import UNIFIED_WIDTH


def value(bits, dtype):
    width = UNIFIED_WIDTH[dtype]
    if not 0 <= bits < 2**width:
        raise ValueError('raw bits outside dtype')
    if dtype == 'bool':
        return bool(bits)
    if dtype.startswith('uint'):
        return bits
    if dtype.startswith('int'):
        return bits - 2**width if bits >= 2**(width - 1) else bits
    if dtype == 'bfloat16':
        return struct.unpack('<f', (bits << 16).to_bytes(4, 'little'))[0]
    return struct.unpack({16: '<e', 32: '<f', 64: '<d'}[width], bits.to_bytes(width // 8, 'little'))[0]


def parse(log, points, *, launch=0, compile_id=0):
    specs = {p['id']: p for p in points}
    events = {pid: {} for pid in specs}
    event_count = 0
    for line in log.splitlines():
        if 'TLDBG' not in line:
            continue
        fields = line.split('|')
        if len(fields) >= 3 and fields[0] == 'TLDBG3' and fields[2] == 'X':
            raise ValueError('observation attempted an out-of-bounds read; no extra memory read was performed')
        if len(fields) < 14 or fields[0] != 'TLDBG3' or fields[-1] != 'Z' or fields[1] not in specs or fields[2] not in {'D', 'E'}:
            raise ValueError('unknown/malformed/truncated TLDBG3 packet')
        pid, kind = fields[1:3]
        p = specs[pid]
        try:
            bx, by, bz, tx, ty, tz, visit, index, low, high, *tail = map(int, fields[3:-1])
        except ValueError as exc:
            raise ValueError('invalid TLDBG3 numeric field') from exc
        extents = p['thread_extents']
        if [bx, by, bz] != p['block'] or any(not 0 <= v < n for v, n in zip((tx, ty, tz), extents)):
            raise ValueError('event block/thread outside root domain')
        tid = tx + extents[0] * (ty + extents[1] * tz)
        if not 0 <= visit < 2**64 or not 0 <= index < 2**32 or not 0 <= low < 2**32 or not 0 <= high < 2**32:
            raise ValueError('event integer outside field width')
        if kind == 'E':
            if tail or low not in (0, 1, 2, 3) or high:
                raise ValueError('invalid lifecycle payload')
            key = (tid, kind)
        else:
            if not p['bound'] or len(tail) != 2 * (p['coordinate_count'] + p['ordinal_count']):
                raise ValueError('unbound point or coordinate arity mismatch')
            if any(not 0 <= v < 2**32 for v in tail):
                raise ValueError('coordinate word outside uint32')
            tail = [value(tail[i] | (tail[i + 1] << 32), 'int64') for i in range(0, len(tail), 2)]
            if p['thread'] is not None and tid != p['thread'] or p.get('reader') is not None and tid != p['reader']:
                raise ValueError('data outside reader selection')
            if index not in p['indices'] or low | (high << 32) >= 2**UNIFIED_WIDTH[p['dtype']]:
                raise ValueError('data index/bit width outside observation')
            coords, ordinals = tail[:p['coordinate_count']], tail[p['coordinate_count']:]
            if any(not -(2**63) <= v < 2**63 for v in tail):
                raise ValueError('coordinate/ordinal outside signed int64')
            cursor, ordinal_index = 0, 0
            for s in p['scopes']:
                if s['kind'] == 'group' and tid // s['group_size'] not in s['groups']:
                    raise ValueError('data outside group domain')
                if 'aliases' not in s:
                    continue
                selection = s.get('selection', {})
                if 'iteration' in selection and ordinals[ordinal_index] != selection['iteration']:
                    raise ValueError('data outside iteration selection')
                if 'coordinates' in selection and coords[cursor:cursor + len(s['aliases'])] != selection['coordinates']:
                    raise ValueError('data outside coordinate selection')
                if s['kind'] != 'parallel' and ordinals[ordinal_index] < 1:
                    raise ValueError('invalid iteration ordinal')
                if s['kind'] == 'parallel' and ordinals[ordinal_index] != 0:
                    raise ValueError('Parallel has coordinate identity, not serial ordinal')
                cursor += len(s['aliases'])
                ordinal_index += 1
            key = (tid, kind, visit, index)
        if key in events[pid]:
            raise ValueError('duplicate lifecycle/data event')
        events[pid][key] = (visit, index, low, high, tail)
        event_count += 1
        if event_count > p['budget']:
            raise ValueError('event budget exceeded')
    records, coverage = [], {}
    for pid, p in specs.items():
        found = events[pid]
        attempted = written = 0
        truncated = overflow = False
        for tid in range(p['threads']):
            if (tid, 'E') not in found:
                raise ValueError(f'missing end for {pid}, frontend thread {tid}')
            count, emitted, overflow_bit, _, _ = found[(tid, 'E')]
            if overflow_bit & 2:
                raise ValueError('observation attempted an out-of-bounds read (recorded by end counter)')
            if emitted > p['budget']:
                raise ValueError('end emitted count exceeds launch budget')
            if emitted != min(count, p.get('capacity', 0)) or overflow_bit and count != 2**64 - 1:
                raise ValueError('end counters inconsistent with capacity')
            data = {k: payload for k, payload in found.items() if k[0] == tid and k[1] == 'D'}
            expected = {(tid, 'D', v, i) for v in range(emitted) for i in p['indices']}
            if set(data) != expected:
                raise ValueError(f'missing/unexpected observation data: {pid}, thread {tid}')
            if not p['bound'] and count:
                raise ValueError('inactive point has visits')
            if (p['thread'] is not None and tid != p['thread'] or p.get('reader') is not None and tid != p['reader']) and count:
                raise ValueError('unselected thread reports visits')
            attempted += count
            written += emitted
            truncated |= count > emitted
            overflow |= bool(overflow_bit)
            for visit in range(emitted):
                identities = {tuple(data[(tid, 'D', visit, i)][4]) for i in p['indices']}
                if len(identities) != 1:
                    raise ValueError('elements in one visit have inconsistent coordinates')
        for key, payload in sorted(found.items()):
            if key[1] != 'D':
                continue
            visit, index, low, high, tail = payload
            tid = key[0]
            ex, ey, _ = p['thread_extents']
            records.append(dict(schema=3, point=pid, compile=compile_id, launch=launch, block=p['block'],
                                thread=tid, thread_space='frontend', thread_coordinates=[tid % ex, tid // ex % ey, tid // (ex * ey)],
                                visit=visit, coordinates=tail[:p['coordinate_count']], ordinals=tail[p['coordinate_count']:],
                                index=index, bits=low | (high << 32), dtype=p['dtype']))
        status = 'incomplete' if overflow else 'truncated' if truncated else 'complete'
        coverage[pid] = dict(capture_integrity=status, attempted_visits=attempted, emitted_visits=written,
                             counter_overflow=overflow, point_execution='observed' if attempted else 'no_selected_visit',
                             build_status='active' if p['bound'] else 'inactive_at_build',
                             execution_coverage='selected_events' if status == 'complete' else status)
    return records, coverage
