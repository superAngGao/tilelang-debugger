"""Mechanism probe, independent of the debugger implementation."""
import argparse
import importlib.util
import json
from pathlib import Path
import textwrap
import traceback


def main():
    import torch
    import tilelang
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    cases = {
        'parallel': ('32', '''for i in T.Parallel(65):
    if i < 61:
        value = x[i] + i
        {observe}
        y[i] = value'''),
        'pipeline': ('32', '''for i in T.Pipelined(5, num_stages=3):
    value = x[tx] + i
    {observe}
    y[tx] = value'''),
        'group': ('256', '''with T.ws(1):
    for i in T.serial(3):
        value = x[tx] + i
        {observe}
        y[tx] = value'''),
        'two_dim': ('(16, 2)', '''for i in T.serial(3):
    value = x[tx] + i
    {observe}
    y[tx] = value'''),
        'dynamic': ('32', '''for i in T.serial(tx % 3):
    value = x[tx] + i
    {observe}
    y[tx] = value'''),
        'while_transfer': ('32', '''i = T.alloc_var("int32", init=0)
while i < 5:
    i = i + 1
    if i == 2:
        continue
    if i == 4:
        break
    value = x[tx] + i
    {observe}
    y[tx] = value'''),
    }
    results = []
    for name, (threads, body) in cases.items():
        item = {'case': name}
        try:
            outputs = []
            for instrumented in (False, True):
                stem = name + ('_debug' if instrumented else '_base')
                init = 'c = T.alloc_local((1,), "uint64")\nc[0] = T.uint64(0)' if instrumented else 'pass'
                observe = ('tirx.call_extern("int32", "printf", "PROBE|' + name + '|D|%d|%d|%llu|%d\\n", tx, i, c[0], value)\n'
                           'c[0] = c[0] + T.uint64(1)') if instrumented else 'pass'
                # Replace the marker preserving the original body's indentation.
                lines = []
                for line in body.splitlines():
                    if '{observe}' in line:
                        indent = line[:len(line) - len(line.lstrip())]
                        lines.extend(indent + part for part in observe.splitlines())
                    else:
                        lines.append(line)
                end = ('tirx.call_extern("int32", "printf", "PROBE|' + name + '|E|%d|%llu\\n", tx, c[0])') if instrumented else 'pass'
                code = 'import tilelang.language as T\nfrom tvm import tirx\n@T.prim_func\ndef main(x: T.Tensor((256,), "int32"), y: T.Tensor((256,), "int32")):\n'
                code += f'    with T.Kernel(1, threads={threads}):\n'
                extent_x = 16 if name == 'two_dim' else int(threads)
                code += f'        tx = T.get_thread_binding(0) + T.get_thread_binding(1) * {extent_x}\n'
                code += textwrap.indent(init + '\n' + '\n'.join(lines) + '\n' + end, '        ') + '\n'
                path = folder / (stem + '.py')
                path.write_text(code)
                spec = importlib.util.spec_from_file_location(stem, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                compiled = tilelang.compile(module.main, target='cuda', out_idx=[])
                (folder / (stem + '.cu')).write_text(compiled.get_kernel_source())
                x = torch.arange(256, device='cuda', dtype=torch.int32)
                y = torch.full((256,), -999, device='cuda', dtype=torch.int32)
                compiled(x, y)
                torch.cuda.synchronize()
                outputs.append(y.cpu())
            if not torch.equal(*outputs):
                raise AssertionError('instrumentation changed output')
            item.update(passed=True, output=outputs[0].tolist())
        except Exception:
            item.update(passed=False, error=traceback.format_exc())
        results.append(item)
        (folder / 'summary.json').write_text(json.dumps(results, indent=2))
        print(json.dumps({k: v for k, v in item.items() if k != 'output'}), flush=True)
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
