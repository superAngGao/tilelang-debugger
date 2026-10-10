"""Convert frontend access objects to logical operands, without emitting IR."""
import math
import re

from ..protocols.access import operand_fields


def region(value):
    from tvm import tirx
    if isinstance(value, tirx.Buffer):
        return value, [0] * len(value.shape), list(value.shape)
    if isinstance(value, tirx.BufferLoad):
        return value.buffer, list(value.indices), [1] * len(value.indices)
    if isinstance(value, tirx.BufferRegion):
        return value.buffer, [r.min for r in value.region], [r.extent for r in value.region]
    if isinstance(value, tirx.Call) and str(getattr(value.op, 'name', '')) in {'tl.tileop.region', 'tl.region'}:
        load = value.args[0]
        if isinstance(load, tirx.BufferLoad) and len(value.args) == len(load.indices) + 2:
            return load.buffer, list(load.indices), list(value.args[2:])
    raise ValueError(f'unsupported frontend access region: {type(value).__name__}: {value}')


def operands(selected, other, predicate, spec):
    from tvm import tirx
    if spec['kind'] == 'copy':
        from tilelang.language.copy_op import _normalize_copy_regions
        pair = (selected, other) if spec['side'] == 0 else (other, selected)
        selected = _normalize_copy_regions(*pair)[spec['side']]
    buffer, origin, extents = region(selected)
    shape = list(buffer.shape)
    strides = list(buffer.strides) or [math.prod(shape[i + 1:]) for i in range(len(shape))]
    if any(len(axis) != len(shape) for axis in (origin, extents, strides)):
        raise ValueError('frontend access operand ranks differ')
    values = [predicate, *origin, *extents, *shape, *strides, buffer.elem_offset]
    for expression in values:
        if isinstance(expression, tirx.PrimExpr):
            bad = []
            tirx.stmt_functor.post_order_visit(expression, lambda n: bad.append(n) if isinstance(n, tirx.BufferLoad) and n.buffer.scope() != 'local.var' else None)
            if bad:
                raise ValueError('access operands contain a memory read; bind it to a scalar before observing')
            mutable = []
            tirx.stmt_functor.post_order_visit(expression, lambda n: mutable.append(n) if isinstance(n, tirx.BufferLoad) and n.buffer.scope() == 'local.var' else None)
            if mutable and spec.get('following_effects'):
                raise ValueError('mutable access operands may change during subsequent frontend calls; bind operands in a separate statement')
            if not re.fullmatch(r'(bool|int(8|16|32|64)|uint(8|16|32))', str(expression.dtype)):
                raise ValueError('access operands must be scalar integers fitting signed int64')
            shifts = []
            tirx.stmt_functor.post_order_visit(expression, lambda n: shifts.append(n) if isinstance(n, tirx.Call) and str(getattr(n.op, 'name', '')) in {'tirx.shift_left', 'tirx.shift_right'} else None)
            for shift in shifts:
                count = shift.args[1]
                width = int(re.search(r'\d+', str(shift.args[0].dtype))[0])
                if not isinstance(count, tirx.IntImm) or not 0 <= int(count) < width:
                    raise ValueError('access shift needs a statically valid count; bind its result before observing')
        elif type(expression) not in (int, bool) or not -(2**63) <= expression < 2**63:
            raise ValueError('access operands must be scalar integers fitting signed int64')
    metadata = dict(semantics='source_logical_request', operation=spec['operation'], kind=spec['kind'],
                    buffer=str(buffer.name), memory_scope=buffer.scope(), dtype=str(buffer.dtype),
                    rank=len(shape), fields=operand_fields(len(shape)), stride_semantics='frontend_logical',
                    predicate=spec['predicate'], expression=spec['expression'])
    return metadata, values
