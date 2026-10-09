# TileOPs 外部示例验证（2026-10-09）

本次完成 **Softmax、RMSNorm、RoPE 的外部数值基线与接入缺口测试**。没有修改 debugger 产品模块，没有增加源码、算子或 CUDA 白名单。**通用用户 kernel 接入尚未完成，也未在这三个 kernel 上取得中间 tile/runtime 索引。**

## 实测结果

| 项目 | 结果 |
| --- | --- |
| Softmax：N=256、257、513/tile_n=256 × FP16/BF16/FP32 | 9/9 基线通过 |
| RMSNorm：N=256、257 × FP16/BF16/FP32 | 6/6 基线通过 |
| RoPE neox 1D：32×64、16×128 × FP16/BF16/FP32 | 6/6 基线通过 |
| 每类首个 FP16 参数组 × memcheck/racecheck/synccheck | 9/9 通过，零错误/零 hazards 摘要 |
| 三类分别执行公开 `run`、`trace` | 6/6 确认为 `unsupported`，不是采集 PASS |
| `--require-debugger`，Softmax N=257/FP16 | 数值基线通过、debugger unsupported，真实进程退出 1 |
| 新增 CPU 回归 | Windows Python 3.14 和 Linux Python 3.12 各 8 tests PASS |
| 原 `test_cpu.py` / `test_evidence.py` | 7 + 5 tests PASS |

数值容差在 GPU 运行前已固定于 manifest，验证中未放宽。比较完整输出并逐元素保存；Softmax/RMSNorm 的输出 padding 另以零容差验证。reference 在 CPU 上使用实际输入副本，RoPE 使用传入 kernel 的实际 cos/sin 表。sanitizer 验证的对象是未插桩原始上游 kernel，不能外推到未来的采集宏或其他参数组。

## 环境与复现

- TileOPs `95ba6cae78856ec2f610535fde3d44df198d39e8`，从本地 bundle 建立独立 checkout，普通包导入；三类的实际模块及全部已导入 TileOPs 文件均核对路径与摘要。
- NVIDIA H200，GPU 2；Python 3.12.13；torch 2.10.0+cu129；TileLang 0.1.12，沿用前阶段验证环境，关闭编译缓存。
- 原始本地 TileOPs 工作区未修改；已有未跟踪文件保留。

在配置好 CUDA、TileLang 和 TileOPs 依赖的 Linux/H200 上运行：

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=2 python tests/validate_tileops.py \
  --tileops /path/to/TileOPs --output artifacts/tileops/matrix-001 --sanitizers

PYTHONPATH=src CUDA_VISIBLE_DEVICES=2 python tests/validate_tileops.py \
  --tileops /path/to/TileOPs --output artifacts/tileops/strict-001 \
  --case softmax-tail-float16 --require-debugger
```

首条命令实测退出 0，summary 同时明确 `baseline_passed=true`、`debugger_status=unsupported`、`delivery_passed=false`。第二条实测退出 1。输出目录不可复用。

## 缺口证据

`run` 在 `capture.run` 读取 driver 同级 `kernel.py` 时失败，`trace` 在 `access.run` 的同一假设处失败。真实上游模块位于 TileOPs 包中，本仓库没有为迎合接口复制这个文件。

因此这次确认的是**最早的入口限制**；CLI 尚未执行到观察点配置解析、JIT/factory 绑定或 lowering 检查。配置文件中的 `tileops-observation-intent-v1` 仅由测试脚本解析成真实源码行号，不能声称产品 CLI 已支持。即使修复路径入口，固定 source/driver、shape、layout、索引等契约仍需通用化。

## 证据与审阅

- 本地 `artifacts/tileops/matrix-001/`：30 个独立数值/sanitizer 进程及 6 个 CLI 探测，完整命令、退出状态、原始日志和产物。
- `artifacts/tileops/strict-001/`：首次严格验收；`strict-002/` 补充复测，`strict-parent-002/process.json` 保存父进程实际观测到的退出码 1 及命令，避免只凭 summary 的 exitcode 字段作判断。
- `artifacts/tileops/cpu-tests.log`：Linux 8 项 CPU 测试。
- `artifacts/tileops-evidence.tgz`：取回本地的证据包，SHA256 `eec45955c9ae5ae64f27973bab04d0594dc314a1ebea3a825fc6c2800eeb7c5e`。运行产物不纳入 git。
- [设计复审](reviews/tileops-integration-design-review-v2.md)、[代码复审](reviews/tileops-integration-code-review-v2.md)均 PASS，v1 的发现与修正保留。
- GPU 证据的独立复算与核验见[验收审阅](reviews/tileops-integration-acceptance-review.md)。

后续通用入口的实现与外部验收路线见[方案](tileops-integration-plan.md)。这批测试用于暴露并验证原目标中的接入缺口，不替代该目标。
