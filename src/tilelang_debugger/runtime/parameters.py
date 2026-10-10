"""Frontend parameter expressions and per-launch bindings, without lowered IR."""
import operator
import re


def truncdiv(a, b):
    quotient = abs(a) // abs(b)
    return -quotient if (a < 0) != (b < 0) else quotient


def integer(value, dtype, *, cast=False):
    if dtype == 'bool':
        return int(value != 0)
    match = re.fullmatch(r'(u?int)(8|16|32|64)', dtype)
    if not match:
        raise ValueError(f'unsupported frontend integer dtype: {dtype}')
    unsigned, width = match[1] == 'uint', int(match[2])
    limit = 1 << width
    if cast or unsigned:
        value %= limit
        return value if unsigned or value < limit // 2 else value - limit
    if not -limit // 2 <= value < limit // 2:
        raise ValueError(f'frontend signed integer overflow: {dtype}')
    return value

OPERATORS = {'Add': operator.add, 'Sub': operator.sub, 'Mul': operator.mul,
             'FloorDiv': operator.floordiv, 'FloorMod': operator.mod,
             'Div': truncdiv, 'Mod': lambda a, b: a - truncdiv(a, b) * b, 'Min': min, 'Max': max}


def expression(value):
    name = type(value).__name__
    if name == 'IntImm':
        return int(value)
    if name in ('Var', 'SizeVar'):
        integer(0, str(value.dtype))
        return {'var': str(value.name), 'dtype': str(value.dtype)}
    if name == 'Cast':
        integer(0, str(value.dtype))
        return {'op': 'Cast', 'dtype': str(value.dtype), 'args': [expression(value.value)]}
    if name in OPERATORS:
        integer(0, str(value.dtype))
        return {'op': name, 'dtype': str(value.dtype), 'args': [expression(value.a), expression(value.b)]}
    raise ValueError(f'cannot serialize frontend dimension expression: {name}')


def evaluate(expr, bindings):
    if type(expr) is int:
        return expr
    if 'var' in expr:
        value = bindings[expr['var']]
        if 'dtype' in expr:
            converted = integer(value, expr['dtype'])
            if converted != value:
                raise ValueError('frontend variable outside dtype range')
        return value
    args = [evaluate(x, bindings) for x in expr['args']]
    if expr['op'] == 'Cast':
        return integer(args[0], expr['dtype'], cast=True)
    if expr['op'] in {'Div', 'Mod', 'FloorDiv', 'FloorMod'}:
        if args[1] == 0:
            raise ValueError('frontend integer division by zero')
        if expr.get('dtype', '').startswith('int') and args[1] == -1:
            integer(-args[0], expr['dtype'])
    value = OPERATORS[expr['op']](*args)
    return integer(value, expr['dtype']) if 'dtype' in expr else value


def parameters(prim_func):
    result = []
    for param in prim_func.params:
        if param in prim_func.buffer_map:
            buffer = prim_func.buffer_map[param]
            result.append(dict(name=str(param.name), kind='tensor', dtype=str(buffer.dtype),
                               shape=[expression(s) for s in buffer.shape], strides=[expression(s) for s in buffer.strides]))
        else:
            result.append(dict(name=str(param.name), kind='scalar', dtype=str(param.dtype), shape=None))
    return result


def bind(parameters, items, roles):
    bindings = {}
    def remember(name, value):
        if name in bindings and bindings[name] != value:
            raise ValueError(f'inconsistent frontend symbol: {name}')
        bindings[name] = value
    for item, index in zip(items, roles):
        param = parameters[index]
        if item['kind'] != param['kind']:
            raise ValueError('snapshot role differs from parameter')
        if item['kind'] == 'scalar':
            if item['dtype'] == 'int':
                remember(param['name'], item['value'])
            continue
        if item['dtype'] != param['dtype'] or len(item['shape']) != len(param['shape']):
            raise ValueError('tensor snapshot differs from frontend parameter')
        for key, actual in (('shape', 'shape'), ('strides', 'stride')):
            for expected, value in zip(param.get(key, []), item[actual]):
                if isinstance(expected, dict) and 'var' in expected:
                    remember(expected['var'], value)
    for item, index in zip(items, roles):
        param = parameters[index]
        if item['kind'] != 'tensor':
            continue
        for key, actual in (('shape', 'shape'), ('strides', 'stride')):
            for expected, value in zip(param.get(key, []), item[actual]):
                if evaluate(expected, bindings) != value:
                    raise ValueError(f'tensor {actual} differs from frontend parameter')
    return bindings
