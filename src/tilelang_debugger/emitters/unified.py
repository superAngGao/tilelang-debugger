"""TLDBG3 frontend macros. No lowered-IR inspection or compiler hook."""
import itertools
import math

from .. import capture_state as state
from ..protocols.dtypes import UNIFIED_WIDTH

LOOPS = {'serial', 'unroll', 'parallel', 'pipeline', 'while'}
MAX_COUNTER = 2**64 - 1


def frontend_thread(selection, extents):
    """Normalize an explicit source-thread selector, never a CUDA hardware ID."""
    if selection is None:
        return None
    if isinstance(selection, list):
        if len(selection) != 3 or any(type(x) is not int or not 0 <= x < e for x, e in zip(selection, extents)):
            raise ValueError('thread coordinates outside frontend CTA')
        selection = selection[0] + extents[0] * (selection[1] + extents[1] * selection[2])
    if type(selection) is not int or not 0 <= selection < math.prod(extents):
        raise ValueError('thread selection outside frontend CTA')
    return selection


def active(pid):
    if state._active is None or pid not in state._active:
        raise ValueError('observation outside active session')
    return state._active[pid]


def geometry():
    import tilelang.language as T
    from tilelang.language.kernel import KernelLaunchFrame
    f = KernelLaunchFrame.Current()
    if f is None:
        raise ValueError('observation outside Kernel')
    extents = f.get_thread_extents()
    xyz = f.get_thread_bindings()
    tid = xyz[0] + extents[0] * (xyz[1] + extents[1] * xyz[2])
    block = tuple(f.get_block_bindings())
    return f, extents, xyz, tid, block + (0,) * (3 - len(block))


def selected_block(p, block):
    import tilelang.language as T
    selector = T.bool(True)
    for actual, expected in zip(block, p['block']):
        selector = T.And(selector, actual == expected)
    return selector


def packet(p, kind, visit, index, low, high, coords=(), ordinals=()):
    import tilelang.language as T
    from tvm import tirx
    _, _, xyz, _, block = geometry()
    args = [T.cast(v, 'int32') for v in (*block, *xyz)]
    args += [T.cast(visit, 'uint64'), T.cast(index, 'uint32'), T.cast(low, 'uint32'), T.cast(high, 'uint32')]
    # Integer narrowing may remove int64 casts on loop indices. Encode words,
    # never rely on a %lld vararg retaining the cast through the compiler.
    for v in (*coords, *ordinals):
        bits = T.cast(T.cast(v, 'int64'), 'uint64')
        args += [T.cast(bits, 'uint32'), T.cast(bits >> 32, 'uint32')]
    fmt = f"TLDBG3|{p['id']}|{kind}|" + '|'.join(['%d'] * 6 + ['%llu', '%u', '%u', '%u'] + ['%u', '%u'] * (len(coords) + len(ordinals))) + '|Z\n'
    return tirx.call_extern('int32', 'printf', fmt, *args)


def start(pid):
    import tilelang.language as T
    p = active(pid)
    if p.get('root_built'):
        raise ValueError('point root built twice in one frontend construction')
    frame, extents, _, _, block = geometry()
    grid = [frame.get_block_extent(i) for i in range(len(frame.get_block_bindings()))]
    grid += [1] * (3 - len(grid))
    thread = frontend_thread(p['thread'], extents)
    if any(b >= e for b, e in zip(p['block'], grid)):
        raise ValueError('block/thread selection outside launch')
    p.update(root_built=True, thread=thread, thread_space='frontend', thread_extents=extents, threads=math.prod(extents), grid=grid,
             scope='inactive', dtype=None, shape=[], elements=0, indices=[], bound=False)
    if sum(q.get('threads', 0) for q in state._active.values()) > p['budget']:
        raise ValueError('budget cannot fit all end packets')
    @T.macro
    def initialize():
        counter = T.alloc_local((3,), 'uint64')
        counter[0] = T.uint64(0)
        counter[1] = T.uint64(0)
        counter[2] = T.uint64(0)
        return counter
    return initialize()


def ordinal():
    import tilelang.language as T
    @T.macro
    def allocate():
        value = T.alloc_local((1,), 'int64')
        value[0] = T.int64(0)
        return value
    return allocate()


def iteration(value, start, stop, step):
    return (value - (0 if stop is None else start)) // (1 if step is None else step) + 1


def finish(pid, counter):
    import tilelang.language as T
    p = active(pid)
    _, _, _, _, block = geometry()
    selector = selected_block(p, block)
    @T.macro
    def close():
        if selector:
            packet(p, 'E', counter[0], counter[1], counter[2], 0)
    close()


def region_indices(shape, region):
    if region is None:
        if math.prod(shape) > 65536:
            raise ValueError('whole buffer exceeds event capacity; select a region')
        return list(range(math.prod(shape)))
    if not isinstance(region, list) or len(region) != len(shape):
        raise ValueError('region needs one [start,stop,step] per dimension')
    axes = []
    for r, size in zip(region, shape):
        if not isinstance(r, list) or len(r) != 3 or any(type(x) is not int for x in r) or r[2] <= 0 or not 0 <= r[0] < r[1] <= size:
            raise ValueError('region must be a positive-step nonempty in-bounds slice')
        axes.append(range(*r))
    if math.prod(len(axis) for axis in axes) > 65536:
        raise ValueError('region exceeds event capacity')
    return [sum(x * math.prod(shape[i + 1:]) for i, x in enumerate(coords)) for coords in itertools.product(*axes)]


def raw(value, dtype):
    import tilelang.language as T
    width = UNIFIED_WIDTH[dtype]
    if dtype == 'bool':
        return T.cast(value, 'uint32'), T.uint32(0)
    bits = T.reinterpret('uint' + str(width), value)
    return T.cast(bits, 'uint32'), T.cast(bits >> 32, 'uint32') if width == 64 else T.uint32(0)


def observe(value, pid, counter, coords, ordinals):
    import tilelang.language as T
    from tilelang.language.utils import index_to_coordinates
    from tvm import tirx
    p = active(pid)
    if p['bound']:
        raise ValueError('point constructed more than once in one kernel')
    if isinstance(value, tirx.Buffer):
        scope, shape = value.scope(), [int(x) for x in value.shape]
        if scope not in {'local', 'local.var', 'local.fragment', 'shared', 'shared.dyn', 'global'}:
            raise ValueError(f'unsupported memory scope: {scope}')
        if not shape or any(x <= 0 for x in shape):
            raise ValueError('observation requires positive static buffer shape')
    elif isinstance(value, tirx.PrimExpr):
        scope, shape = 'scalar', []
        if p.get('region') is not None:
            raise ValueError('scalar cannot have region')
    else:
        raise ValueError('observe existing PrimExpr or Buffer')
    dtype = str(value.dtype)
    if dtype not in UNIFIED_WIDTH:
        raise ValueError(f'unsupported dtype: {dtype}')
    indices = region_indices(shape, p.get('region'))
    _, _, _, tid, block = geometry()
    selector = selected_block(p, block)
    if p['thread'] is not None:
        selector = T.And(selector, tid == p['thread'])
    cursor = 0
    for i, s in enumerate(s for s in p['scopes'] if s['kind'] in LOOPS):
        selection = s.get('selection', {})
        if 'iteration' in selection:
            selector = T.And(selector, ordinals[i] == selection['iteration'])
        if 'coordinates' in selection:
            for offset, expected in enumerate(selection['coordinates']):
                selector = T.And(selector, coords[cursor + offset] == expected)
        cursor += len(s['aliases'])
    collective = scope in {'local.fragment', 'shared', 'shared.dyn', 'global'}
    reader, barrier, participants = None, None, p['threads']
    if collective:
        if p['thread'] is not None:
            raise ValueError('whole-buffer observation selects logical elements with region, not thread; use collective.reader only to override its output reader')
        contract = p.get('collective')
        if not isinstance(contract, dict) or contract.get('ready') is not True or set(contract) - {'ready', 'barrier', 'uniform', 'fence_regs', 'reader'}:
            raise ValueError('whole-buffer observation requires explicit collective ready contract')
        if any(s['kind'] in LOOPS | {'branch'} for s in p['scopes']) and contract.get('uniform') is not True:
            raise ValueError('collective path requires explicit participation-uniformity contract')
        groups = [s for s in p['scopes'] if s['kind'] == 'group']
        domain = set(range(p['threads']))
        for s in groups:
            domain &= {t for g in s['groups'] for t in range(g * 128, (g + 1) * 128)}
        if not domain:
            raise ValueError('collective has empty participating domain')
        participants = len(domain)
        requested_reader = frontend_thread(contract.get('reader'), p['thread_extents'])
        reader = min(domain) if requested_reader is None else requested_reader
        if reader not in domain:
            raise ValueError('reader outside group')
        barrier = contract.get('barrier')
        if groups and (type(barrier) is not int or not 1 <= barrier <= 15):
            raise ValueError('group collective requires a reserved named barrier 1..15')
        if not groups and barrier is not None:
            raise ValueError('full CTA collective uses its standard barrier')
        selector = T.And(selector, tid == reader)
    elif 'collective' in p:
        raise ValueError('scalar/local/element observation uses thread, not a collective contract')
    # Reserve E for the independently known frontend domain. A constant start
    # printf can be duplicated onto compiler-added TMA producer threads.
    point_count = len(state._active)
    allowance = p['budget'] // point_count - p['threads']
    emitters = 1 if collective or p['thread'] is not None else p['threads']
    capacity = allowance // (emitters * len(indices))
    if capacity < 1:
        raise ValueError('budget cannot fit end packets and one complete observation')
    p.update(bound=True, scope=scope, dtype=dtype, shape=shape, indices=indices, elements=len(indices),
             capacity=capacity, reader=reader, participants=participants, barrier=barrier)
    state._buffers[pid] = value

    def data(actual, index):
        low, high = raw(actual, dtype)
        return packet(p, 'D', counter[0], index, low, high, coords, ordinals)

    def buffer_data(buffer, position):
        region = p.get('region') or [[0, x, 1] for x in shape]
        sizes = [len(range(*r)) for r in region]
        selected = index_to_coordinates(position, sizes)
        actual = [r[0] + v * r[2] for v, r in zip(selected, region)]
        index = sum(v * math.prod(shape[i + 1:]) for i, v in enumerate(actual))
        return data(buffer[actual], index)

    @T.macro
    def direct(buffer):
        if selector:
            if counter[0] < T.uint64(MAX_COUNTER):
                if counter[0] < capacity:
                    for position in T.serial(len(indices)):
                        buffer_data(buffer, position)
                    counter[1] = counter[1] + T.uint64(1)
                counter[0] = counter[0] + T.uint64(1)
            else:
                counter[2] = counter[2] | T.uint64(1)

    read_valid = T.bool(True)
    if isinstance(value, tirx.BufferLoad):
        for index, extent in zip(value.indices, value.buffer.shape):
            read_valid = T.And(read_valid, T.And(index >= 0, index < extent))

    @T.macro
    def scalar():
        if selector:
            if read_valid:
                if counter[0] < T.uint64(MAX_COUNTER):
                    if counter[0] < capacity:
                        data(value, 0)
                        counter[1] = counter[1] + T.uint64(1)
                    counter[0] = counter[0] + T.uint64(1)
                else:
                    counter[2] = counter[2] | T.uint64(1)
            else:
                if (counter[2] & T.uint64(2)) == T.uint64(0):
                    if counter[0] < capacity:
                        packet(p, 'X', counter[0], 0, 0, 0, coords, ordinals)
                counter[2] = counter[2] | T.uint64(2)
                if counter[0] < T.uint64(MAX_COUNTER):
                    counter[0] = counter[0] + T.uint64(1)
                else:
                    counter[2] = counter[2] | T.uint64(1)

    def sync():
        return T.sync_threads() if barrier is None else T.sync_threads(barrier, participants)

    @T.macro
    def tile():
        # No private counter, thread or iteration selection guards a barrier.
        if scope == 'local.fragment':
            if p['collective'].get('fence_regs', 0):
                T.warpgroup_fence_operand(value, num_regs=p['collective']['fence_regs'])
            scratch = T.alloc_shared(shape, dtype)
            T.copy(value, scratch)
            sync()
            direct(scratch)
            sync()
        else:
            sync()
            direct(value)
            sync()
    if collective:
        tile()
    elif scope == 'scalar':
        scalar()
    else:
        direct(value)
