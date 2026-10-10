"""Source access candidates, operation semantics and evaluation paths."""
import ast

from ..instrument import Unsupported
from .expressions import pure, operand, domain_safe, PURE_CALLS

SCALAR_CALLS = PURE_CALLS | {'T.abs', 'T.exp', 'T.exp2', 'T.log', 'T.log2', 'T.sqrt', 'T.rsqrt',
    'T.sin', 'T.cos', 'T.tanh', 'T.sigmoid', 'T.isnan', 'T.isinf', 'T.isfinite',
    'T.floor', 'T.ceil', 'T.round', 'T.trunc', 'T.uint64', 'T.float16', 'T.bfloat16', 'T.float32', 'T.float64'}
ATOMIC_CALLS = {'T.atomic_add', 'T.atomic_min', 'T.atomic_max'}


def negate(node):
    return ast.UnaryOp(op=ast.Not(), operand=node)


def bind_operands(node, names):
    """Bind known operand names without evaluating frontend calls."""
    result = dict(zip(names, node.args))
    for keyword in node.keywords:
        if keyword.arg in names:
            if keyword.arg in result:
                raise Unsupported('duplicate access operation argument')
            result[keyword.arg] = keyword.value
        elif keyword.arg is None:
            raise Unsupported('access operation needs explicit keyword arguments')
    if not all(name in result for name in names):
        raise Unsupported('access operation requires explicit ' + '/'.join(names))
    return [result[name] for name in names]


def select(statement, point):
    if point['when'] != 'before':
        raise Unsupported('access operands must be observed before the original statement')
    if any(k in point for k in ('region', 'collective')):
        raise Unsupported('access uses the original access region, without a collective read')
    if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr)):
        raise Unsupported('select the access statement, not its enclosing control statement')
    if isinstance(statement, ast.Assign) and (len(statement.targets) != 1 or isinstance(statement.targets[0], (ast.Tuple, ast.List))):
        raise Unsupported('multiple assignment targets need a single-evaluation adapter before access observation')
    wanted = ast.parse(point['buffer'], mode='eval').body
    operation = point.get('operation')
    if operation not in (None, 'read', 'write', 'read_write', 'address'):
        raise Unsupported('access operation must be read, write, read_write or address')
    matches, effects = [], []

    def add(node, role, guards, forbidden, **extra):
        root = node.value if isinstance(node, ast.Subscript) else node
        match = ast.unparse(node) == ast.unparse(wanted) if isinstance(wanted, ast.Subscript) else isinstance(root, ast.Name) and root.id == wanted.id
        if match and (operation is None or operation == role):
            matches.append(dict(node=node, guards=guards, forbidden=forbidden, effects=list(effects),
                                kind='element', operation=role, **extra))

    def walk(node, guards=(), forbidden=False):
        if node is None:
            return
        if isinstance(node, ast.Call):
            walk(node.func, guards, forbidden)
            name = ast.unparse(node.func)
            if name == 'T.copy' or name in ATOMIC_CALLS or name == 'T.address_of':
                names = ('src', 'dst') if name == 'T.copy' else ('dst', 'value') if name in ATOMIC_CALLS else ('obj',)
                try:
                    args = bind_operands(node, names)
                except Unsupported:
                    for arg in [*node.args, *(kw.value for kw in node.keywords)]:
                        walk(arg, guards, True)
                    effects.append((name, id(node)))
                    return
                # Special roles don't change Python's argument evaluation order.
                for arg in [*node.args, *(kw.value for kw in node.keywords)]:
                    role_index = next((i for i, a in enumerate(args) if a is arg), None)
                    is_target = role_index is not None and (name == 'T.copy' or role_index == 0)
                    if not is_target:
                        walk(arg, guards, forbidden)
                        continue
                    if isinstance(arg, ast.Subscript):
                        walk(arg.value, guards, forbidden)
                        walk(arg.slice, guards, forbidden)
                    else:
                        walk(arg, guards, forbidden)
                    if name == 'T.copy':
                        add(arg, 'read' if role_index == 0 else 'write', guards, forbidden,
                            source=ast.unparse(args[0]), destination=ast.unparse(args[1]), side=role_index, adapter='copy', owner=id(node))
                    else:
                        add(arg, 'read_write' if name in ATOMIC_CALLS else 'address', guards, forbidden,
                            adapter='atomic' if name in ATOMIC_CALLS else 'address', owner=id(node))
                if name != 'T.address_of':
                    effects.append((name, id(node)))
                return
            if name == 'T.if_then_else' and len(node.args) == 3 and not node.keywords:
                conditional(*node.args, guards, forbidden)
                return
            if name in {'T.And', 'T.Or'} and len(node.args) == 2 and not node.keywords:
                boolean(node.args, name == 'T.And', guards, forbidden)
                return
            for arg in [*node.args, *(kw.value for kw in node.keywords)]:
                walk(arg, guards, forbidden or name not in SCALAR_CALLS)
            if name not in SCALAR_CALLS:
                effects.append((name, id(node)))
            return
        if isinstance(node, ast.IfExp):
            conditional(node.test, node.body, node.orelse, guards, forbidden)
            return
        if isinstance(node, ast.BoolOp):
            boolean(node.values, isinstance(node.op, ast.And), guards, forbidden)
            return
        if isinstance(node, ast.Compare):
            walk(node.left, guards, forbidden)
            previous, left = [], node.left
            for op, right in zip(node.ops, node.comparators):
                walk(right, (*guards, *previous), forbidden)
                previous.append(ast.Compare(left=left, ops=[op], comparators=[right]))
                left = right
            return
        if isinstance(node, ast.Subscript):
            walk(node.value, guards, forbidden)
            walk(node.slice, guards, forbidden)
            add(node, 'write' if isinstance(node.ctx, ast.Store) else 'read', guards, forbidden)
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            walk(node.value, guards, forbidden)
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                store_target(target, guards, forbidden)
            return
        if isinstance(node, ast.NamedExpr):
            walk(node.value, guards, forbidden)
            store_target(node.target, guards, forbidden)
            return
        if isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Subscript):
                walk(node.target.value, guards, forbidden)
                walk(node.target.slice, guards, forbidden)
                add(node.target, 'read', guards, forbidden)
                add(node.target, 'write', guards, forbidden)
            else:
                walk(node.target, guards, forbidden)
            walk(node.value, guards, forbidden)
            return
        for child in ast.iter_child_nodes(node):
            walk(child, guards, forbidden)

    def store_target(target, guards, forbidden):
        if isinstance(target, (ast.Tuple, ast.List)):
            for child in target.elts:
                store_target(child, guards, forbidden)
        else:
            walk(target, guards, forbidden)
            effects.append(('assignment target', id(target)))

    def boolean(values, is_and, guards, forbidden):
        previous = []
        for child in values:
            walk(child, (*guards, *previous), forbidden)
            previous.append(child if is_and else negate(child))

    def conditional(test, yes, no, guards, forbidden):
        walk(test, guards, forbidden)
        base = list(effects)
        walk(yes, (*guards, test), forbidden)
        yes_effects = list(effects[len(base):])
        effects[:] = base
        walk(no, (*guards, negate(test)), forbidden)
        effects.extend(yes_effects)

    walk(statement)
    if not matches:
        raise Unsupported('no matching source access at selected statement')
    occurrence = point.get('occurrence')
    if occurrence is None and len(matches) != 1:
        raise Unsupported('multiple matching accesses; select operation or zero-based occurrence')
    if occurrence is not None and (type(occurrence) is not int or not 0 <= occurrence < len(matches)):
        raise Unsupported('access occurrence outside matching source accesses')
    chosen = matches[0 if occurrence is None else occurrence]
    node, guards = chosen.pop('node'), chosen.pop('guards')
    if chosen.pop('forbidden'):
        raise Unsupported('buffer-operation semantics need an adapter')
    prefix = chosen.pop('effects')
    owner = chosen.pop('owner', None)
    suffix = [name for name, identifier in effects[len(prefix):] if identifier != owner and name != 'assignment target']
    if prefix:
        raise Unsupported('access follows a potentially side-effecting evaluation prefix; needs explicit scalar binding')
    if not operand(node) or any(not pure(g) for g in guards):
        raise Unsupported('access index/condition needs explicit scalar binding: do not repeat memory reads or side-effecting calls')
    adapter = chosen.pop('adapter', 'element')
    replayed = [node, *guards]
    if adapter == 'copy':
        pair = [ast.parse(chosen[k], mode='eval').body for k in ('source', 'destination')]
        replayed.extend(pair)
        if not all(operand(n) for n in pair):
            raise Unsupported('copy operands need explicit buffers/regions with pure indices')
    elif adapter in {'atomic', 'address'}:
        if not isinstance(node, ast.Subscript) or any(isinstance(n, ast.Slice) for n in ast.walk(node)):
            raise Unsupported('atomic/address access currently requires a single element')
    if guards and any(not domain_safe(n) for n in replayed):
        raise Unsupported('masked access needs explicit scalar binding to avoid evaluating an undefined inactive index or condition')
    chosen.update(kind=adapter, expression=ast.unparse(node),
                  predicate=ast.unparse(ast.BoolOp(op=ast.And(), values=list(guards))) if len(guards) > 1 else ast.unparse(guards[0]) if guards else 'True',
                  occurrence=0 if occurrence is None else occurrence, following_effects=suffix)
    return chosen
