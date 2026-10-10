"""Encode logical access operands with the ordinary TLDBG3 print strategy."""
from .unified import active, observe
from ..frontend.access import operands


def observe_access(selected, other, predicate, pid, counter, coords, ordinals):
    import tilelang.language as T
    p = active(pid)
    metadata, values = operands(selected, other, predicate, p['access_spec'])
    metadata['branches'] = [s for s in p['scopes'] if s['kind'] == 'branch']
    p['access'] = metadata

    def fill(operands):
        for i, value in enumerate(values):
            # Capture operands in the closure: passing a symbolic Var as a
            # macro parameter can rename the original ABI dimension/stride.
            @T.macro
            def assign():
                operands[i] = T.cast(value, 'int64')
            assign()

    @T.macro
    def emit():
        operands = T.alloc_local((len(values),), 'int64')
        fill(operands)
        observe(operands, pid, counter, coords, ordinals)
    emit()
