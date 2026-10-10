# TileOPs 外部示例独立验收审阅

> 历史记录：旧 `trace` 已从当前产品移除；文中 trace 命令、专用测试和兼容性描述仅适用于[移除前提交](https://github.com/superAngGao/tilelang-debugger/tree/e838fc88d96a562ee6230ee154d6593bac6632f8)。保留当时结果，不代表当前产品能力。

**结论：PASS，本次外部基线与接入缺口测试子任务通过，无剩余阻塞项。Debugger 对这三类外部 kernel 仍是 unsupported，完整调试交付未通过。**

- 日期：2026-10-09。
- 证据：`artifacts/tileops/matrix-001`、`strict-001`、`strict-002`、`strict-parent-002`、`cpu-tests.log`。
- 上游：`95ba6cae78856ec2f610535fde3d44df198d39e8`，真实远端 checkout `/home/ang.gao/tileops-debugger-fixtures-95ba6cae`。
- 审阅方式：独立读取本地完整证据；SSH 只读上游源码，并用现有 Python/torch 在 CPU 上重算保存的 tensor。审阅者没有重跑 GPU，也没有修改实现。
- 本地证据包 SHA256 已独立核对：`eec45955c9ae5ae64f27973bab04d0594dc314a1ebea3a825fc6c2800eeb7c5e`，与验证文档一致。

## 独立核验结果

| 项目 | 结果 |
| --- | --- |
| 21 个 shape/path/dtype 数值基线 | 全部通过 |
| 每类一例的 memcheck/racecheck/synccheck，共 9 worker | 全部退出 0、无超时，实际日志错误/风险摘要均为 0 |
| matrix 的 30 worker 证据文件 | 必需产物齐全，逐个 SHA256 与清单一致 |
| 来源 | 保存的上游 commit/干净状态正确；所有导入模块摘要与指定 checkout 实际文件逐一匹配 |
| 编译产物 | 每例保存实际前端/设备 IR JSON、可读 IR、CUDA 和编译配置；环境为 H200、TileLang 0.1.12 |
| CPU 数值复核 | matrix 30 份、strict-001 与 strict-002 各 1 份，共 32 份输入/输出独立重算通过 |
| 公开 CLI 接入 | matrix 六次真实 run/trace 均在读取对应 sibling kernel.py 时失败，正确标 unsupported |
| CPU 回归 | 保存的远端日志为 8 tests PASS；代码审阅阶段已独立重跑同一 8 tests |
| 原有 CPU 回归 | 设置 `PYTHONPATH=src` 后独立执行 `test_cpu.py` 7 tests 与 `test_evidence.py` 5 tests，全部通过 |
| 产品模块 | 本地 `git diff --stat -- src` 无改动 |

逐 worker 检查了真实进程退出码/超时、结果 case、数值通过标记、padding 精确为零、逐元素匹配行数，以及 `instrumented=false`、`debugger_status=not_run`。计划观察点保留 `planned_not_captured`，没有伪造中间值或 runtime 访问记录。

CPU 复核直接读取 `tensors.pt`，用保存的实际输入重新计算 softmax、有效 N 的 RMS normalization 和拆分前后半部的 RoPE，再按执行前固定的 dtype 容差检查实际输出，确认 dtype/shape/有限性。全部通过。较大的绝对差来自 RoPE：FP16 最大 0.001953125，BF16 最大 0.015625，仍满足各自的 `atol + rtol * abs(expected)` 判据；没有放宽阈值。

为检查独立性，RMSNorm 复核使用有效 N 切片的 mean，而原 reference 使用 padded sum/N。最初要求两种 reference 位相同的附加断言在 FP32 不成立；这是不同 reduction 求和顺序的舍入差，最大 9.5367431640625e-7。按预设 1e-5 的数值判据，两种参考及 GPU 输出均通过；这没有被解释为位相同。

## 严格模式的真实退出证据

`strict-001` 内部 summary 为 exitcode=1，但最初没有独立父进程状态文件，因此本审阅没有把这个内部字段当成真实外层退出的唯一依据。

补充的 `strict-002` 由 `strict_process_probe.py` 的父进程执行真实 `--require-debugger` 命令。已直接读取 `strict-parent-002/process.json`，其 `returncode=1`、`timeout=false`，并核对父进程 stdout、严格运行的基线文件摘要、CPU reference 与两个真实 CLI traceback。该次基线通过而外部接入 unsupported，导致完整验收非零退出，符合设计。

## 本次 PASS 的边界

这批例子已经能从指定 TileOPs checkout 直接编译、运行、保存输入输出/编译产物、与独立 CPU reference 比较，并明确暴露现有 debugger 的入口缺口。它们尚不能通过 debugger 获取中间 tile 或 runtime 索引。`summary.json` 正确保留 `debugger_status=unsupported`、`delivery_passed=false`；README 也明确区分基线通过与采集通过。

去除产品源码/driver/布局/索引白名单、接入真实 module/factory 的工作仍未完成。本验收不替代那部分实现和独立审阅。
