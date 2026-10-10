"""Derived logical access reports from already validated TLDBG3 records."""
from collections import defaultdict

from .unified import value
from .dtypes import UNIFIED_WIDTH


def operand_fields(rank):
    return ['active'] + [f'{kind}.{i}' for kind in ('origin', 'extent', 'shape', 'stride') for i in range(rank)] + ['elem_offset']


def report(records, contracts, coverage):
    points = {(c['launch'], p['id']): p for c in contracts for p in c['points'] if p.get('mode') == 'access'}
    groups = defaultdict(list)
    for record in records:
        if (record['launch'], record['point']) in points:
            groups[(record['launch'], record['point'], record['thread'], record['visit'])].append(record)
    rows = []
    for (launch, pid, thread, visit), samples in sorted(groups.items()):
        p = points[(launch, pid)]
        meta = p['access']
        if meta['fields'] != operand_fields(meta['rank']):
            raise ValueError('access operand schema differs')
        if sorted(r['index'] for r in samples) != list(range(len(meta['fields']))):
            raise ValueError('incomplete access operand tuple')
        operands = {meta['fields'][r['index']]: value(r['bits'], r['dtype']) for r in samples}
        if operands['active'] not in (0, 1):
            raise ValueError('access predicate is not boolean')
        axes = {key: [operands[f'{key}.{i}'] for i in range(meta['rank'])] for key in ('origin', 'extent', 'shape', 'stride')}
        start, extent, shape, strides = (axes[k] for k in ('origin', 'extent', 'shape', 'stride'))
        if any(n < 0 for n in shape + extent):
            raise ValueError('negative access shape/extent')
        stop = [a + n for a, n in zip(start, extent)]
        inside = all(0 <= a and b <= n for a, b, n in zip(start, stop, shape))
        intersects = all(max(0, a) < min(b, n) for a, b, n in zip(start, stop, shape))
        bounds = 'empty' if any(n == 0 for n in extent) else 'in_bounds' if inside else 'partial' if intersects else 'out_of_bounds'
        active = bool(operands['active'])
        offset = operands['elem_offset'] + sum(a * s for a, s in zip(start, strides))
        first = samples[0]
        rows.append(dict(launch=launch, compile=first['compile'], point=pid, line=p['line'], source=p['source_path'],
                         block=first['block'], thread=thread, thread_space='frontend', visit=visit,
                         coordinates=first['coordinates'], ordinals=first['ordinals'],
                         semantics='source_logical_request', operation=meta['operation'], kind=meta['kind'],
                         buffer=meta['buffer'], memory_scope=meta['memory_scope'], dtype=meta['dtype'],
                         active=active, status=bounds if active else 'masked', candidate_bounds=bounds,
                         **axes, stop=stop, elem_offset=operands['elem_offset'],
                         derived_element_offset=offset, derived_byte_offset=offset * max(1, UNIFIED_WIDTH[meta['dtype']] // 8),
                         offset_semantics='frontend_logical_not_physical',
                         in_bounds_region=[[min(n, max(0, a)), max(min(n, max(0, a)), min(b, n))] for a, b, n in zip(start, stop, shape)],
                         predicate=meta['predicate'], branches=meta['branches']))
    summaries = []
    for (launch, pid), p in sorted(points.items()):
        found = [r for r in rows if r['launch'] == launch and r['point'] == pid]
        counts = {state: sum(r['status'] == state for r in found) for state in ('in_bounds', 'partial', 'out_of_bounds', 'masked', 'empty')}
        summaries.append(dict(launch=launch, point=pid, line=p['line'], requests=len(found), counts=counts, **coverage[str(launch)][pid]))
    return dict(schema='source-access-v1', semantics='source_logical_request', points=summaries), rows


def markdown(summary):
    lines = ['# Source access observations', '',
             'Logical source requests observed before execution; not physical addresses or completed memory transactions.',
             'Copy regions describe the requested tile, including any boundary tail. Branch-local points record only reached branches.', '',
             'Atomic requests have operation read_write; address requests only form a pointer and do not assert a memory read/write.', '',
             '| Launch | Point | Line | Requests | In bounds | Partial | Out of bounds | Masked | Integrity |',
             '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for p in summary['points']:
        c = p['counts']
        lines.append(f"| {p['launch']} | {p['point']} | {p['line']} | {p['requests']} | {c['in_bounds']} | {c['partial']} | {c['out_of_bounds']} | {c['masked']} | {p['capture_integrity']} |")
    return '\n'.join(lines) + '\n'
