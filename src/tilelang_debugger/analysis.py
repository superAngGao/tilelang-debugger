"""Offline reference analysis of complete captures, without TileLang or GPU use."""
import copy
from contextlib import contextmanager
import hashlib
import json
import platform
from pathlib import Path
import sys
import types
import uuid

from .evidence import read, verify_capture, reference_status
from .numerics import compare, tensor_data
from .records import value


def save(path, data):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def manifest(folder):
    return {str(p.relative_to(folder)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def tensors(folder, prefix):
    import torch
    result = []
    for m in read(folder / f"{prefix}.json"):
        raw = bytearray((folder / m["file"]).read_bytes())
        result.append(torch.frombuffer(raw, dtype=getattr(torch, m["dtype"])).clone().reshape(m["shape"]))
    return tuple(result)


@contextmanager
def load_provider(source, path):
    name = "_tldbg_user_reference_" + uuid.uuid4().hex
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
        function = getattr(module, "reference", None)
        if not callable(function):
            raise ValueError("provider must define callable reference(inputs, points)")
        yield function
    finally:
        sys.modules.pop(name, None)


def reference_specs(bundle, points, output_count):
    if not isinstance(bundle, dict) or set(bundle) != {"points", "outputs"}:
        raise ValueError("reference must return points and outputs")
    expected_points, expected_outputs = bundle["points"], bundle["outputs"]
    if not isinstance(expected_points, dict) or set(expected_points) != {p["id"] for p in points}:
        raise ValueError("reference must cover exactly the selected point ids")
    if not isinstance(expected_outputs, (list, tuple)) or len(expected_outputs) != output_count:
        raise ValueError("reference must cover every output in order")
    # Snapshot all returned tensors before doing any comparison.
    def freeze(spec):
        if not isinstance(spec, dict) or set(spec) != {"tensor", "atol", "rtol"}:
            raise ValueError("reference spec requires tensor, atol, rtol")
        return tensor_data(spec["tensor"]), spec["atol"], spec["rtol"]
    return ({key: freeze(spec) for key, spec in expected_points.items()}, [freeze(spec) for spec in expected_outputs])


def markdown(summary):
    def cell(x):
        return str(x).replace("|", "\\|").replace("\n", " ")
    lines = ["# TileLang 数值分析", "", f"分析完成；数值比较：{'全部匹配' if summary['matched'] else '存在差异'}。",
             f"Capture run：`{summary['run_id']}`；驱动原有 reference：{summary['capture_numerical_status']}。", "",
             "| 对象 | shape | 不匹配 / 总数 | 最大绝对误差 | 最大相对误差 |",
             "| --- | --- | ---: | ---: | ---: |"]
    for item in summary["comparisons"]:
        lines.append(f"| {cell(item['label'])} | {item['shape']} | {item['mismatches']} / {item['elements']} | {item['max_abs_error']} | {item['max_relative_error']} |")
    lines += ["", "误差以 reference 为基准；NaN 总是不匹配，同号 Inf 匹配。NaN 的误差为 null，不参与最大误差统计；特殊值计数见 analysis.json。", ""]
    for item in summary["comparisons"]:
        lines += [f"## {cell(item['label'])}", "", f"实际 dtype：{item['actual_dtype']}；reference dtype：{item['expected_dtype']}；atol={item['atol']}，rtol={item['rtol']}。", ""]
        if item["kind"] == "point":
            p = item["source"]
            lines += [f"原始 {cell(p.get('source_path', 'kernel.py'))} 第 {p['line']} 行（{p['when']}），buffer `{cell(p['buffer'])}`，block {p['block']}，循环选择 {p['loops']}，原循环值 {p['loop_values']}。",
                      f"源码摘要：`{p['source_sha256']}`。", ""]
        if item["first_mismatches"]:
            lines += ["前 20 个不匹配元素（按逻辑索引排序）：", "", "| 坐标 | 实际值 | 参考值 | 绝对误差 | 分类 |", "| --- | ---: | ---: | ---: | --- |"]
            for row in item["first_mismatches"]:
                lines.append(f"| {row['coordinates']} | {row['actual']} | {row['expected']} | {row['abs_error']} | {row['kind']} |")
            lines.append("")
    lines += ["完整逐元素数据及双方原始位模式见 elements.jsonl；本报告不推断错误根因。", ""]
    return "\n".join(lines)


def analyze(capture, reference, output):
    if read(Path(capture) / "run.json").get("schema") == "source-samples-v2":
        from .sample_analysis import analyze as analyze_samples
        return analyze_samples(capture, reference, output)
    capture, reference, output = (Path(p).resolve() for p in (capture, reference, output))
    if output == capture or capture in output.parents:
        raise ValueError("analysis output must be outside the capture directory")
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(schema=1, status="running", capture=str(capture))
    save(output / "analysis.json", summary)
    try:
        import torch
        run = read(capture / "run.json")
        evidence = manifest(capture)
        points = verify_capture(capture, run.get("sanitizer"))
        if not isinstance(run.get("run_id"), str) or not run["run_id"]:
            raise ValueError("capture is missing its run identity")
        inputs = tensors(capture / "baseline", "inputs")
        outputs = tensors(capture / "baseline", "outputs")
        records = [json.loads(line) for line in (capture / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        actual_points = {}
        for p in points:
            selected = [r for r in records if r["point"] == p["id"]]
            actual_points[p["id"]] = dict(shape=p["shape"], dtype=p["dtype"], bits=[r["bits"] for r in selected],
                                           values=[value(r["bits"], p["dtype"]) for r in selected])
        actual_outputs = [tensor_data(t) for t in outputs]
        provider_source = reference.read_bytes()
        (output / "reference.py").write_bytes(provider_source)
        summary.update(run_id=run["run_id"], provider=dict(path=str(reference), sha256=hashlib.sha256(provider_source).hexdigest()),
                       environment=dict(python=sys.version, platform=platform.platform(), torch=torch.__version__),
                       capture_numerical_status=reference_status([read(capture / m / "reference.json") for m in ("baseline", "instrumented")], run.get("schema", 1)))
        save(output / "evidence.json", evidence)
        with load_provider(provider_source, output / "reference.py") as provider:
            bundle = provider(tuple(t.clone() for t in inputs), copy.deepcopy(points))
            refs, output_refs = reference_specs(bundle, points, len(outputs))
        jobs = [("point", p["id"], actual_points[p["id"]], refs[p["id"]], p) for p in points]
        jobs += [("output", i, a, spec, None) for i, (a, spec) in enumerate(zip(actual_outputs, output_refs))]
        comparisons = []
        with (output / "elements.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for kind, identity, actual, (expected, atol, rtol), point in jobs:
                result, rows = compare(actual, expected, atol, rtol)
                result.update(kind=kind, identity=identity, label=f"{kind}:{identity}")
                if point is not None:
                    result["source"] = copy.deepcopy(point)
                comparisons.append(result)
                for row in rows:
                    stream.write(json.dumps(dict(scope=kind, identity=identity, actual_dtype=actual["dtype"], expected_dtype=expected["dtype"], **row), allow_nan=False, separators=(",", ":")) + "\n")
        if manifest(capture) != evidence:
            raise ValueError("capture evidence changed during analysis")
        summary.update(status="completed", matched=all(x["passed"] for x in comparisons), comparisons=comparisons)
        (output / "report.md").write_text(markdown(summary), encoding="utf-8")
        save(output / "analysis.json", summary)
        return summary
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        summary.pop("matched", None)
        save(output / "analysis.json", summary)
        raise
