"""Source scope adapters for the version-three observation engine."""
import ast
import copy
import re

from .normalize import normalized
from ..instrument import Unsupported, digest

LOOPS = {'range': 'serial', 'T.serial': 'serial', 'T.Serial': 'serial',
         'T.unroll': 'unroll', 'T.Unroll': 'unroll', 'T.Parallel': 'parallel',
         'T.Pipelined': 'pipeline'}


def called(node):
    return ast.unparse(node.func) if isinstance(node, ast.Call) else ''


def loop(node, prefix):
    sid = f'L{node.lineno}'
    if isinstance(node, ast.While):
        return dict(kind='while', line=node.lineno, id=sid, names=[],
                    aliases=[f'{prefix}_{sid}_ordinal[0]'], ordinal=f'{prefix}_{sid}_ordinal[0]')
    name = called(node.iter)
    if name not in LOOPS:
        raise Unsupported(f'unknown loop adapter: {name}')
    kind = LOOPS[name]
    if kind == 'pipeline':
        manual = node.iter.args[3:] + [k.value for k in node.iter.keywords if k.arg in {'order', 'stage', 'sync', 'group'}]
        if any(not isinstance(a, ast.Constant) or a.value is not None for a in manual):
            raise Unsupported('manual Pipelined schedule arrays need a statement-schedule adapter; automatic num_stages is supported')
    names = [node.target.id] if isinstance(node.target, ast.Name) else [n.id for n in node.target.elts] if isinstance(node.target, ast.Tuple) and all(isinstance(n, ast.Name) for n in node.target.elts) else []
    if not names or kind != 'parallel' and len(names) != 1:
        raise Unsupported('unsupported loop binding')
    if node.orelse:
        raise Unsupported('loop else requires a frontend adapter')
    args = node.iter.args
    if any(k.arg is None for k in node.iter.keywords):
        raise Unsupported('expanded loop keyword arguments need an explicit signature')
    if any(isinstance(a, ast.Starred) for a in args):
        raise Unsupported('expanded loop arguments require an explicit signature')
    if kind == 'parallel':
        if len(args) != len(names):
            raise Unsupported('Parallel arity differs')
        start, step = '0', '1'
    else:
        parameters = ['start', 'stop', 'num_stages', 'order', 'stage', 'sync', 'group'] if kind == 'pipeline' else ['start', 'stop', 'step']
        if len(args) > len(parameters) or name == 'range' and node.iter.keywords:
            raise Unsupported('invalid loop signature')
        supplied = {parameters[i]: f'{prefix}_{sid}_arg{i}' for i in range(len(args))}
        for i, keyword in enumerate(node.iter.keywords):
            if keyword.arg in supplied:
                raise Unsupported('duplicate loop argument')
            supplied[keyword.arg] = f'{prefix}_{sid}_kw{i}'
        if 'start' not in supplied:
            raise Unsupported('loop needs start')
        start, stop = supplied['start'], supplied.get('stop', 'None')
        step = supplied.get('step', 'None') if kind != 'pipeline' else '1'
    aliases = [f'{prefix}_{sid}_c{i}' for i in range(len(names))]
    ordinal = '0' if kind == 'parallel' else f'{prefix}_iteration({aliases[0]}, {start}, {stop}, {step})'
    return dict(kind=kind, line=node.lineno, id=sid, names=names, aliases=aliases,
                ordinal=ordinal, call=name, start=start, step=step)


def prepare(source, config):
    if config.get('schema') != 3 or set(config) - {'schema', 'source', 'points', 'budget'}:
        raise Unsupported('unified configuration requires schema=3, source, points')
    if not isinstance(config.get('points'), list) or not 1 <= len(config['points']) <= 8:
        raise Unsupported('select 1..8 points')
    budget = config.get('budget', 65536)
    if type(budget) is not int or not 1 <= budget <= 65536:
        raise Unsupported('budget must be 1..65536 events per launch')
    tree = normalized(source)
    parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    prefix = '__tldbg3'
    while any(n.startswith(prefix) for n in used):
        prefix += '_'
    points, ids = [], set()
    required = {'id', 'line', 'when', 'buffer', 'block', 'loops'}
    for original in config['points']:
        if not isinstance(original, dict) or not required <= original.keys() or original.keys() - required - {'thread', 'region', 'collective'}:
            raise Unsupported('invalid unified observation fields')
        p = copy.deepcopy(original)
        if not isinstance(p['id'], str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,31}', p['id']) or p['id'] in ids:
            raise Unsupported('invalid/duplicate point id')
        ids.add(p['id'])
        if type(p['line']) is not int or p['when'] not in {'before', 'after'}:
            raise Unsupported('invalid source position')
        block = p['block']
        if not isinstance(block, list) or len(block) != 3 or any(type(v) is not int or v < 0 for v in block):
            raise Unsupported('block requires three nonnegative integers')
        thread = p.get('thread')
        if thread is not None and not (type(thread) is int and thread >= 0 or isinstance(thread, list) and len(thread) == 3 and all(type(v) is int and v >= 0 for v in thread)):
            raise Unsupported('thread needs linear integer or [x,y,z]')
        try:
            expression = ast.parse(p['buffer'], mode='eval').body
        except (SyntaxError, TypeError) as exc:
            raise Unsupported('invalid observation expression') from exc
        allowed = (ast.Name, ast.Subscript, ast.Tuple, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Load,
                   ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd)
        if not isinstance(expression, (ast.Name, ast.Subscript)) or any(not isinstance(n, allowed) for n in ast.walk(expression)):
            raise Unsupported('observe a bound object or element with pure indices')
        if isinstance(expression, ast.Subscript) and (not isinstance(expression.value, ast.Name) or any(isinstance(n, ast.Subscript) for n in ast.walk(expression.slice))):
            raise Unsupported('element indices cannot add memory reads')
        targets = [n for n in ast.walk(tree) if isinstance(n, ast.stmt) and n.lineno == p['line']]
        if len(targets) != 1 or not isinstance(targets[0], (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.For, ast.While, ast.If)):
            raise Unsupported('select an unambiguous computation statement')
        target = targets[0]
        chain, child = [], target
        while child in parents:
            parent = parents[child]
            if isinstance(parent, ast.FunctionDef):
                break
            chain.append((parent, child))
            child = parent
        chain.reverse()
        kernels = [n for n, _ in chain if isinstance(n, ast.With) and len(n.items) == 1 and called(n.items[0].context_expr) == 'T.Kernel']
        if len(kernels) != 1:
            raise Unsupported('observation needs one enclosing T.Kernel')
        kernel = kernels[0]
        scopes = []
        inside = False
        for node, child in chain:
            if node is kernel:
                inside = True
                continue
            if not inside:
                continue
            if isinstance(node, (ast.For, ast.While)):
                scopes.append(loop(node, prefix))
            elif isinstance(node, ast.If):
                scopes.append(dict(kind='branch', line=node.lineno, arm=int(child in node.body), condition=ast.unparse(node.test)))
            elif isinstance(node, ast.With):
                if len(node.items) != 1 or called(node.items[0].context_expr) not in {'T.ws', 'T.WarpSpecialize'}:
                    raise Unsupported('unknown enclosing context adapter')
                args = node.items[0].context_expr.args
                if not args or any(not isinstance(a, ast.Constant) or type(a.value) is not int or a.value < 0 for a in args):
                    raise Unsupported('Group adapter requires explicit nonnegative group ids')
                scopes.append(dict(kind='group', line=node.lineno, groups=[a.value for a in args], group_size=128,
                                   adapter='tilelang-0.1.12/language/warpgroup.py'))
            elif isinstance(node, ast.stmt):
                raise Unsupported(f'unknown enclosing statement: {type(node).__name__}')
        loops = [s for s in scopes if s['kind'] in {'serial', 'unroll', 'parallel', 'pipeline', 'while'}]
        seen = set()
        if not isinstance(p['loops'], list):
            raise Unsupported('loops must be a list')
        for selection in p['loops']:
            if not isinstance(selection, dict) or selection.get('line') in seen:
                raise Unsupported('invalid/duplicate loop selection')
            seen.add(selection.get('line'))
            matches = [s for s in loops if s['line'] == selection.get('line')]
            if len(matches) != 1:
                raise Unsupported('selected loop must enclose the point')
            scope = matches[0]
            if set(selection) == {'line', 'iteration'} and scope['kind'] != 'parallel':
                if type(selection['iteration']) is not int or selection['iteration'] < 1:
                    raise Unsupported('iteration is positive and one-based')
            elif set(selection) == {'line', 'coordinates'}:
                values = selection['coordinates']
                if not isinstance(values, list) or len(values) != len(scope['aliases']) or any(type(x) is not int for x in values):
                    raise Unsupported('coordinate selection arity differs')
            else:
                raise Unsupported('invalid loop selection')
            scope['selection'] = selection
        coordinate_count = sum(len(s['aliases']) for s in loops)
        if 10 + 2 * (coordinate_count + len(loops)) > 32:
            raise Unsupported('observation identity exceeds printf argument capacity')
        # Transfer targets are explicit metadata; no function-wide early-exit ban.
        transfers = []
        pending = [kernel]
        kernel_nodes = []
        while pending:
            n = pending.pop()
            kernel_nodes.append(n)
            pending.extend(c for c in ast.iter_child_nodes(n) if not isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)))
        for n in kernel_nodes:
            if not isinstance(n, (ast.Break, ast.Continue, ast.Return)):
                continue
            if isinstance(n, ast.Return):
                # In fixed eager TileLang this returns from Python frontend
                # construction, NOT from the running GPU function.
                if parents.get(n) is not kernel or n is not kernel.body[-1] or n.value is not None:
                    raise Unsupported('conditional/nested/value return exits frontend construction, not GPU execution; only final bare kernel return is supported')
            ancestor = parents.get(n)
            while ancestor is not None and not isinstance(ancestor, (ast.For, ast.While) if not isinstance(n, ast.Return) else (ast.FunctionDef,)):
                ancestor = parents.get(ancestor)
            transfers.append(dict(kind=type(n).__name__.lower(), line=n.lineno, target=getattr(ancestor, 'lineno', None)))
        p.update(schema=3, engine='unified', scopes=scopes, transfers=transfers, kernel_line=kernel.lineno,
                 prefix=prefix, source_path=config['source'], source_sha256=digest(source.encode()),
                 thread=thread, budget=budget, coordinate_count=coordinate_count, ordinal_count=len(loops))
        points.append(p)
    return points
