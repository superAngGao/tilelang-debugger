"""Resolve symbolic frontend capture extents against a launch's arguments."""
import copy
from .parameters import evaluate


def resolve(points, symbols):
    from ..emitters.unified import region_indices
    result = copy.deepcopy(points)
    for p in result:
        p['grid'] = [evaluate(v, symbols) for v in p['grid']]
        if any(not 0 <= b < n for b, n in zip(p['block'], p['grid'])):
            raise ValueError('selected block outside actual launch grid')
        if not p['bound']:
            continue
        p['shape'] = [evaluate(v, symbols) for v in p['shape']]
        if any(n <= 0 for n in p['shape']):
            raise ValueError('observed buffer has empty/invalid runtime shape')
        p['indices'] = region_indices(p['shape'], p.get('region'))
        p['elements'] = len(p['indices'])
        p['capacity'] = p['allowance'] // (p['emitters'] * p['elements'])
        if p['capacity'] < 1:
            raise ValueError('budget cannot fit one complete runtime observation')
    return result
