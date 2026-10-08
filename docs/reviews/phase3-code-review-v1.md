# 第三步独立代码审阅 v1

结论：**FAIL，需修复 1 项 P2 provider 加载兼容问题后复审**。已检查的采集/数值状态分离、原始 bits 比较、reference 数学映射和严格验收边界没有发现其他阻塞性代码问题。此结论不是 H200 全量验收通过；本轮 racecheck 五路径尚未全部完成。

审阅日期：2026-10-08。审阅独立于实施 session，先读取 `docs/phase3-plan.md` 和已 PASS 的 `docs/reviews/phase3-design-review-v1.md`，再读取当前 diff、新增实现和测试。审阅没有修改产品实现，没有启动 GPU，仅写本审阅文件，运行本地标准库测试及远程 `CUDA_VISIBLE_DEVICES=''` CPU 测试，并只读复核已产生的真实 capture。

## 必须修正：P2，自包含标准库 provider 不能正常使用 dataclass

位置：`src/tilelang_debugger/analysis.py`，`load_provider` 创建 `types.ModuleType("_tldbg_user_reference")` 后直接 `exec`，未把模块放入 `sys.modules`。

合法的自包含 Python provider 可以使用标准库 dataclasses 和推迟求值的注解。以下代码在普通模块中合法，在当前 loader 中则在定义 Config 时抛出 `AttributeError: 'NoneType' object has no attribute '__dict__'`，尚未调用 reference 就被当作 provider 错误：

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Config:
    atol: float = 0.0

def reference(inputs, points):
    return {"points": {}, "outputs": []}
```

此最小复现只测试模块加载，不依赖返回值的合法性。本机 Python 3.14 和项目远程 PyTorch 环境都独立复现相同异常。dataclasses 解析字符串注解时通过类的 `__module__` 查找模块命名空间，而 loader 没有提供该命名空间。方案明确允许自包含的标准库/PyTorch provider，因此不应要求用户规避正常 Python 模块行为。

建议为每次加载分配唯一模块名，在执行源码前注册 `sys.modules`，至少保留到 reference 调用和全部 tensor 返回值冻结结束，并在成功/异常路径的 `finally` 中清理。保持执行保存的同一份 provider 字节和输入复制隔离。补充含 future annotations/dataclass 的合法 provider 回归，检查成功、异常后清理及连续加载不会串模块。

## 已核对通过的实现边界

- 五个驱动仅替换旧输出断言/JSON 保存为 `check_output`。独立 AST 比较确认原输入、编译参数、launch、reference 数学表达式完全相同。helper 对合法数值失配写 false 并继续 capture；shape/dtype/容差错误仍抛异常；独立运行数值失配仍抛 AssertionError。
- `capture.run` 只增加两份布尔 reference 的校验和 numerical_status 汇总。worker、launch 门禁、输入/输出字节一致性和完整日志解析未放宽。`verify_capture` 保留原证据检查，`verify_success` 继续额外拒绝任一 reference=false。缺失/非布尔 passed 和与 worker 不一致的 numerical_status 被拒绝。
- analyze 根据保存的 sanitizer 模式验证证据，未按当前驱动白名单重新授权历史 capture。观察点 actual 从完整日志对应 records 的 bits 解码，输出 actual 从快照重建，浮点值在主机 double 比较；双方 bits 单独保存，int32 极值未先转 float32。
- NaN 总不匹配、同号 Inf 匹配、其他 Inf 不匹配；正负零数值相等但 bits 保留。reference 零值的 relative error、行优先多维坐标、全部元素计数、前 20 个错误预览和标准 JSON 特殊值表示符合设计。
- provider 输入 tensor clone、points deepcopy，全部返回 tensor 在比较前 detach/contiguous/clone；点 key、输出数量、shape、dtype、CPU/strided/非空以及容差严格检查。保存 provider 原字节/摘要、capture 文件摘要和环境信息，报告输出位于独立目录。
- GELU 使用实际输入切片和独立 FP32 GELU；Sum 同时覆盖 N256/N257 和显式补零；GEMM 使用 iteration 次数而非零基 loop_values 计算 K 前缀；GQA 两 consumer 的 Q 行偏移和固定配置完整 attention 对应原设计。原有 GELU 采集保真验收没有被新数值容差替换。
- `monitor.py`、`ir.py`、`instrument.py`、`records.py`、layout-contracts 和四份 monitor.json 与 HEAD 原始字节一致。四份 kernel.py 与 HEAD 仅受 Windows 工作树 CRLF/LF 表示影响，规范化后完全一致、无 git 内容 diff。contracts 去除 driver_sha256 后与 HEAD 完全相同；只有五个驱动哈希更新。
- 数值负例测试夹具只把 `check_output(out, ref,` 改为 `check_output(out, ref + 100,`。临时驱动哈希仅在测试进程中替换内存契约，无产品绕过开关；未改 kernel、门禁或比较规则。

## 独立运行的 CPU 检查

本地执行 `test_numerics.py` 5 项、`test_evidence.py` 5 项、`test_cpu.py` 7 项，全部通过。远程仓库 `/home/ang.gao/tilelang-debugger-phase3-20261008` 使用 `/home/ang.gao/miniforge3/envs/tilesight-tl012/bin/python`，设置 `PYTHONPATH=src CUDA_VISIBLE_DEVICES=''`，独立复跑 `test_analysis.py` 7 项全部通过。

另外通过真正的 CLI 子进程验证 provider RuntimeError 产生退出码 1、analysis.status=failed、无 matched 字段；独立验证空 sanitizer 日志、非零 hazard 日志和包含 Target application returned an error 的日志均拒绝。已有测试覆盖 CLI 数值失配退出 2、精确错误坐标、NaN payload/负零原始 bits、provider 原地修改隔离和接口错误拒绝。

## 已产生真实产物的只读复核

以下均位于上述远程仓库 `artifacts/`，审阅未生成或改写这些 capture，也未用单元测试模拟数据冒充 GPU 证据。

- `phase3-historical`：五条历史真实路径的分析及错误 reference 分析已完成。对每条原 capture 重跑严格证据检查，重算 manifest，与 evidence.json 一致；保存 provider SHA256 一致；JSONL 完整行数/错误计数和 analysis.json 一致。五条全部 matched，错误 reference 为 completed + matched=false。
- `phase3-racecheck`：GELU、Sum、Sum 无 padding 的严格证据检查通过，记录数分别为 4096、1026、514。当前整体验收 summary=failed；GEMM instrumented worker 被既有 IR gate 拒绝，尚无本轮该路径完整 racecheck 通过产物，GQA 尚待完成。不得把前三条通过或其他模式通过冒充五条 racecheck 全通过。
- `phase3-synccheck`：GEMM 和 GQA 的严格证据检查通过，各有 16384 条完整记录。
- `phase3-numeric-mismatch`：实际两 worker 都是 reference=false，同时 process/一次 compile/一次 launch、sanitizer、完整 capture 证据通过。独立执行 verify_success 确实拒绝，verify_capture 通过；原夹具与偏移夹具仅有上述 reference 表达式差异。保存的 CLI 结果退出 2；正确独立 reference 重新分析 matched=true；capture 的 manifest 与分析时保存的摘要一致。

## 复审和完成条件

修复上述 loader 问题并通过新增 CPU 回归后，可以复审代码 PASS。本轮五路径 racecheck、新 capture 的离线分析及相应精简验收记录仍需完成；已有门禁拒绝产物必须保留，不能通过放宽 monitor/IR/layout 授权来凑齐通过结论。
