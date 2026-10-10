# 第四步独立最终验收复核

> 历史记录：旧 `trace` 已从当前产品移除；文中 trace 命令、专用测试和兼容性描述仅适用于[移除前提交](https://github.com/superAngGao/tilelang-debugger/tree/e838fc88d96a562ee6230ee154d6593bac6632f8)。保留当时结果，不代表当前产品能力。

**结论：PASS。当前固定四例与五条验收路径的交付要求已满足，无剩余阻塞。**

日期：2026-10-09，Asia/Shanghai。本次复核由独立 reviewer 完成，只读源码和本地/远端产物，独立运行 CPU/TIR 测试及证据重算，没有编译或启动 GPU，没有修改产品代码。产品 `access*.py` 与 `access-pins.json` 的摘要仍与 [代码复审 v2](phase4-code-review-v2.md) 一致。

## 独立重验

在固定远端 Python 环境中设置 `CUDA_VISIBLE_DEVICES=''`，按七个独立进程运行 `test_cpu.py`、`test_evidence.py`、`test_numerics.py`、`test_analysis.py`、`test_ir.py`、`test_access.py`、`test_access_ir.py`：分别通过 7、5、5、8、7、5、5 项，**合计 42 项、无 skip**。`tests/run_cpu.py` 的隔离有实际理由：analysis 检查进程未导入 TileLang，而 IR suite 需要导入它；原断言保留，最初统一 discover 的失败日志也保留。隔离没有掩盖测试失败。

对 `access-racecheck-final`、`access-synccheck-final`、`access-memcheck-final` 分别从原始产物重验，五条路径全部通过：

| 路径 | 每次完整记录数 | racecheck | synccheck | memcheck |
| --- | ---: | --- | --- | --- |
| GELU | 4096 | PASS | PASS | PASS |
| Sum N257 | 512 | PASS | PASS | PASS |
| Sum N256 | 1280 | PASS | PASS | PASS |
| GEMM | 2 | PASS | PASS | PASS |
| GQA | 6 | PASS | PASS | PASS |

复核覆盖全部 **30 个 baseline/instrumented worker**：外层正常退出、未超时、pipeline/codegen 各一次且 wrapper 恢复、launch gate、sanitizer 实际命令和零错误摘要（racecheck 同时零 hazards/warnings）、真实输入输出快照字节的长度与 SHA256、两版快照一致、两版输出 reference 的成功状态。逐个重新解析原 stdout，记录与 JSONL 完全一致；baseline/pretrace 比较及擦除 trace 后的实际 codegen 比较均通过；从记录与 descriptor 重新生成的 analysis 与保存结果一致。

旧数值采集 `value-regression-final` 的 GELU/GQA 两例也通过原始 racecheck、进程、快照及记录完整性复核。使用现有 tile verifier 的 CPU 计算逻辑重算 reference，仅在内存中移除其最终文件写入：结果与保存的 validation 完全一致。GELU tile 逐位一致，GQA 两个 QK tile 最大绝对误差均为 `9.5367431640625e-06`。本次旧 GPU 路径回归范围就是这两例，未扩大表述为其他旧 GPU 路径也重新运行。

65,536 条协议极值/错位索引实验沿用代码复审 v2 已完成的独立原日志重放结论。新增保存的 FFI 负实验也已核对：`body-completed.json` 显示四个 descriptor 与 wrapper 恢复，但 `process.json` 为 `returncode=-11, timeout=false`，`validation.json` 为 `passed=false`。最终文档正确区分了内部完成和进程 SIGSEGV；产品没有采用该 hook。

## 包与说明

本地 wheel `artifacts/phase4-wheels/tilelang_debugger-0.1.0-py3-none-any.whl` SHA256 为 `70f5cc1dba7148499e39975801f770d3f89fad3b1cb0ba28200fc88b5ebcba0c`。独立读取 ZIP，五个 access Python 模块、`access-pins.json` 和原 `contracts.json` 全部存在，且逐字节匹配当前源文件。

README 与最终验收文档准确说明固定源码/driver/CUDA 的支持域、CPU suite 隔离、失败产物、输出 reference、runtime 坐标与 host 推导 descriptor 的区别。未把日志解释为事务完成、物理 shared swizzle 或任意 kernel 支持；TMA 负起始 store 的分类与最终实现一致。设计文档已明确最终状态与后续修订的优先级，没有遗留等待实施的顶层状态。

本次受审说明与 runner 摘要：

| 文件 | SHA256 |
| --- | --- |
| README.md | `2662a42724b4e2c3318b05e4a98ad1538a3dcbe3475705a322d0a02bc10d78c4` |
| docs/phase4-plan.md | `7524b6e6af9a339643c49d0ecb11e47a840d817a1d2b768a70aaefb65706c8b6` |
| docs/phase4-validation.md | `98b72818ae849727e13a8d3f1b9041ed0b192335c7a732be73f825c6f2f26ba7` |
| tests/run_cpu.py | `fa303d5db3aad30a798df9e835a5bc50912928b4f925fa71edb86d756b195b18` |

无必须追加的修改或验收。可在最终说明中链接本报告，作为设计、代码审阅之后的独立验收记录；该链接属于非阻塞文档整理。
