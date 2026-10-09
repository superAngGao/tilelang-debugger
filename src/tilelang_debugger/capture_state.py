"""Single worker-local state shared by tile and sample emitters."""
from contextlib import contextmanager

_active = None
_buffers = None


@contextmanager
def session(points):
    global _active, _buffers
    if _active is not None:
        raise RuntimeError("nested monitor session")
    state = {p["id"]: dict(p) for p in points}
    _active = state
    _buffers = {}
    try:
        yield state
        if any(not p.get("root_built" if p.get("schema") == 2 else "bound") for p in state.values()):
            raise RuntimeError("an observation point was not built")
    finally:
        _active = None
        _buffers = None


def buffer_bindings():
    return dict(_buffers or {})
