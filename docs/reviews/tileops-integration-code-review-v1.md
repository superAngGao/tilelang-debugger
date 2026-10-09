# TileOPs 外部示例代码独立审阅 v1

**结论：FAIL，1 项阻塞项。** 真实 `run` 入口的已知接入缺口会被错误分类为 failed；必须修正后再执行完整 H200 验收。

- 日期：2026-10-09。
- 范围：`examples/tileops/**`、`tests/validate_tileops.py`、`tests/test_tileops_integration.py` 与 README 新段落。
- 依据：设计 v2、实际 TileOPs 三类 factory、现有 debugger 两个公开入口及数值比较器。
- 独立执行：`C:\Python314\python.exe tests/test_tileops_integration.py`，7 tests PASS。
- 本轮未运行 H200；没有修改实现。当前 CPU PASS 不能覆盖下面的真实入口差异。

## B1：run 的 traceback 源码行与分类器不一致

`tests/validate_tileops.py:52` 对 `run` 和 `trace` 均要求 traceback 包含：

```python
source = driver.with_name("kernel.py").read_text(encoding="utf-8")
```

这个语句只存在于 `access.run`。`capture.run` 的实际代码是：

```python
source_file = driver.with_name("kernel.py")
source = source_file.read_text(encoding="utf-8")
```

因此三个真实 `run` 探测都会得到 failed，默认基线验收也将非零退出；与计划的已知 unsupported 状态不符。`test_failure_classification_does_not_hide_errors` 构造了不存在于 `capture.py` 的语句，故现有测试掩盖此错误。

修复应根据真实入口匹配对应源码行或调用栈，不放宽为任意 FileNotFoundError。分别用实际 `run`、`trace` traceback 验证，并继续覆盖错误路径、错误模块、依赖错误、超时、崩溃、意外零退出。

## 已确认的实现行为

- 普通 Python 包导入，检查已加载与传递加载的 TileOPs 文件位于指定 checkout，保存 Git commit/状态与源码摘要；没有新增产品白名单。
- 21 组合保持真实 factory 调用。Softmax 参考处理正确输出 padding；RMSNorm 保持 padded 输入且分母为原 N；RoPE 使用真正传入的已量化频率表。reference 都在 CPU 上运行。
- 输出记录明确 `instrumented=False`、`planned_not_captured`，README 不宣称取得了中间值或访问记录。`--require-debugger` 对 unsupported 非零退出。
- 独立 worker 的完整进程退出码参与判定；Linux 超时杀进程组，sanitizer 设置错误退出码并校验非零错误摘要。实际导入/API 兼容、精度和 sanitizer 结果仍需 H200 验证。

## 非阻塞改进及验收关注

1. 已撤回：本轮曾误判 README 方案链接少一级。复核 `Resolve-Path examples/tileops/../../docs/tileops-integration-plan.md` 正确解析到仓库 `docs`；原链接正确，无需修改。
2. `baseline_status` 对 `files.json` 缺少必需文件集合检查；空对象会绕过哈希验证。建议要求所有约定证据文件且核对 result 的 case、numerical_status、comparison，以及输出 dtype。这样缺失产物不会因保留一个 passed 标记被接受。
3. `suite_status` 在 probes 为空且 baselines 全通过时默认退出 0。当前主流程会生成探测，但建议空探测显式失败，并在验收汇总校验每类恰有 run/trace 两条记录。
4. `observations` 保留 `applies_to` 字符串但不计算条件，因此 aligned/tail 单例都会列出两个来源锚点。当前标记为计划意图，尚不构成虚假采集；未来通用接入不能直接把全部点视为当前执行路径。
5. H200 验收应保留原预设容差下的失败，不应为获得 PASS 悄悄放宽阈值。CPU 参考的半精度运算与实际生成算术可存在舍入差异，需要实测解释。
