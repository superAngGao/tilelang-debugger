import ast
import json
from pathlib import Path


def config(checkout, case):
    module = "pool/max_pool2d.py" if case in {"pool", "indices"} else "rope.py" if case == "rope" else "reduction/softmax.py"
    source = Path(checkout) / "src/tileops/kernels" / module
    tree = ast.parse(source.read_text())
    factory = {"pool": "_max_pool2d_kernel", "indices": "_max_pool2d_with_indices_kernel",
               "rope": "_make_rope_neox_position_ids_thd", "softmax": "_softmax_kernel_tiled"}[case]
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == factory)
    points = []
    if case == "pool":
        targets = [("val", "val", "val"), ("max_val", "max_val", "max_val"), ("has_nan", "has_nan", "has_nan")]
    elif case == "indices":
        # Capture state after the original if/elif chain by selecting flat_idx's
        # assignment BEFORE the next update; val itself also samples the inner scope.
        targets = [("val", "val", "val"), ("max_idx", "max_idx", "max_idx"), ("has_nan", "has_nan", "has_nan")]
    elif case == "rope":
        targets = [("pos", "pos", "pos"), ("val", "val", "val"), ("paired_val", "paired_val", "paired_val")]
    else:
        targets = [("full", "tile_f32[i, j]", "tile_f32[i, j]"), ("tail", "tile_f32[i, j]", "tile_f32[i, j]")]
    assignments = sorted([n for n in ast.walk(function) if isinstance(n, ast.Assign)], key=lambda n: n.lineno)
    for name, target, expression in targets:
        matches = [n for n in assignments if ast.unparse(n.targets[0]) == target]
        if case == "softmax":
            node = matches[0 if name == "full" else 1]
        elif case == "pool" and name in {"max_val", "has_nan"}:
            node = matches[-1]
        elif case == "indices" and name == "max_idx":
            node = matches[-1]  # successful strict-greater update arm
        elif case == "indices" and name == "has_nan":
            node = matches[1]  # initial False, before kh/kw loops
        else:
            node = matches[-1]
        points.append(dict(id=name, line=node.lineno, when="after", buffer=expression, block=[0, 0, 0], loops=[]))
    return dict(schema=2, source=str(source.resolve()), points=points)
