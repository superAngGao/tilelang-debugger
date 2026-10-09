"""Frontend construction identities, scoped to one capture worker."""
from contextlib import contextmanager
import copy

from .. import capture_state

_context = None


@contextmanager
def session(points, instrumented):
    global _context
    if _context is not None:
        raise RuntimeError('nested build registry')
    context = dict(points=points, instrumented=instrumented, builds={})
    _context = context
    try:
        yield context
    finally:
        _context = None


def building(decorator, ids):
    def decorate(function):
        if _context is None:
            raise RuntimeError('instrumented source requires a capture build session')
        points = [copy.deepcopy(p) for p in _context['points'] if p['id'] in ids]
        if _context['instrumented']:
            if capture_state._active is not None:
                raise RuntimeError('nested frontend capture')
            capture_state._active = {p['id']: p for p in points}
            capture_state._buffers = {}
            try:
                result = decorator(function)
                if any(not p.get('root_built') for p in points):
                    raise RuntimeError('selected lexical kernel root was not built')
            finally:
                capture_state._active = None
                capture_state._buffers = None
        else:
            result = decorator(function)
        build_id = len(_context['builds'])
        # A frontend metadata attribute survives normal jit symbol rewriting;
        # it adds no instruction and is never interpreted from lowered IR.
        result = result.with_attr('tldbg_build_id', build_id)
        _context['builds'][build_id] = copy.deepcopy(points)
        return result
    return decorate
