"""Source-only rewriting from the prepared control-flow model."""
import ast

from ..source_analysis.normalize import normalized


def inject(source, points, *, enabled=True):
    tree = normalized(source)
    prefix = points[0]['prefix']
    loop_kinds = {'serial', 'unroll', 'parallel', 'pipeline', 'while'}
    loops = {s['line']: s for p in points for s in p['scopes'] if s['kind'] in loop_kinds}

    def statements(text):
        return ast.parse(text).body

    def tuple_code(values):
        return '(' + ','.join(values) + (',' if values else '') + ')'

    def counter(p):
        return f"{prefix}_{p['id']}_counter"

    def end(p):
        return statements(f"{prefix}_finish({p['id']!r}, {counter(p)})")

    class Insert(ast.NodeTransformer):
        def __init__(self):
            self.roots = []

        def visit_FunctionDef(self, node):
            # Only wrap the frontend function that owns these lexical roots.
            kernel_lines = {n.lineno for n in node.body for n in ast.walk(n) if isinstance(n, ast.With)}
            own = [p['id'] for p in points if p['kernel_line'] in kernel_lines]
            decorators = [i for i, d in enumerate(node.decorator_list) if ast.unparse(d) == 'T.prim_func' or isinstance(d, ast.Call) and ast.unparse(d.func) == 'T.prim_func']
            if decorators and own:
                for i in decorators:
                    node.decorator_list[i] = ast.Call(func=ast.Name(id=f'{prefix}_building', ctx=ast.Load()), args=[node.decorator_list[i], ast.parse(repr(own), mode='eval').body], keywords=[])
            old = self.roots
            self.roots = []
            node = self.generic_visit(node)
            self.roots = old
            return node

        def visit_With(self, node):
            roots = [p for p in points if p['kernel_line'] == node.lineno]
            old = self.roots
            if roots and enabled:
                self.roots = roots
            node = self.generic_visit(node)
            if roots and enabled:
                node.body[:0] = [n for p in roots for n in statements(f"{counter(p)} = {prefix}_start({p['id']!r})")]
                node.body.extend(n for p in roots for n in end(p))
            self.roots = old
            return node

        def visit_Return(self, node):
            # The fixed frontend decides whether this return is legal. Do not
            # move/evaluate its value: nonempty returns here need a dedicated adapter.
            if self.roots and node.value is not None:
                from ..instrument import Unsupported
                raise Unsupported('value-return inside kernel needs a frontend return adapter')
            return [n for p in self.roots for n in end(p)] + [node] if self.roots else node

        def visit_For(self, node):
            node = self.generic_visit(node)
            s = loops.get(node.lineno)
            if not s or not enabled:
                return node
            prelude = []
            # Snapshot the original arguments once, preserving left-to-right order.
            # Pipelined scheduling arguments remain in their original call.
            count = len(node.iter.args)
            for i in range(count):
                name = f"{prefix}_{s['id']}_arg{i}"
                prelude.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=node.iter.args[i]))
                node.iter.args[i] = ast.Name(id=name, ctx=ast.Load())
            for i, keyword in enumerate(node.iter.keywords):
                name = f"{prefix}_{s['id']}_kw{i}"
                prelude.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=keyword.value))
                keyword.value = ast.Name(id=name, ctx=ast.Load())
            node.body[:0] = [n for alias, original in zip(s['aliases'], s['names']) for n in statements(f'{alias} = {original}')]
            return prelude + [node]

        def visit_While(self, node):
            node = self.generic_visit(node)
            s = loops.get(node.lineno)
            if not s or not enabled:
                return node
            name = s['ordinal'].removesuffix('[0]')
            before = statements(f'{name} = {prefix}_ordinal()')
            node.body[:0] = statements(f'{name}[0] = {name}[0] + 1')
            return before + [node]

        def visit(self, node):
            out = super().visit(node)
            if isinstance(node, ast.stmt):
                matching = [p for p in points if p['line'] == node.lineno]
                if matching and enabled:
                    def capture(p):
                        scopes = [s for s in p['scopes'] if s['kind'] in loop_kinds]
                        coords = tuple_code([a for s in scopes for a in s['aliases']])
                        ordinals = tuple_code([s['ordinal'] for s in scopes])
                        return statements(f"{prefix}_observe({p['buffer']}, {p['id']!r}, {counter(p)}, {coords}, {ordinals})")
                    body = out if isinstance(out, list) else [out]
                    return [n for p in matching if p['when'] == 'before' for n in capture(p)] + body + [n for p in matching if p['when'] == 'after' for n in capture(p)]
            return out

    tree = Insert().visit(tree)
    pos = int(bool(tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str)))
    while pos < len(tree.body) and isinstance(tree.body[pos], ast.ImportFrom) and tree.body[pos].module == '__future__':
        pos += 1
    tree.body.insert(pos, ast.ImportFrom(module='tilelang_debugger.emitters.unified', names=[ast.alias(name=n, asname=f'{prefix}_{n}') for n in ('start', 'finish', 'observe', 'ordinal', 'iteration')], level=0))
    tree.body.insert(pos, ast.ImportFrom(module='tilelang_debugger.runtime.builds', names=[ast.alias(name='building', asname=f'{prefix}_building')], level=0))
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'
