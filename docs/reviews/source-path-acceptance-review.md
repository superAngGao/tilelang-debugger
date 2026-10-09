# 用户指定源码路径：独立验收审阅

**结论：PASS，无剩余验收阻塞项。用户指定的源码路径已实际用于 GELU 的值采集和访问采集，原路径进入离线报告；源码/driver 契约与外部包/JIT 限制仍未解除。**

- 日期：2026-10-09。
- 证据目录：`artifacts/source-path/`。
- 证据包 SHA256 已独立核对：`45197454ca3e48dfd465e0834b6adc0d50570cb7641a3a4a5fd2ac6f1fefd26c`。
- 方式：独立读取原始进程/采集/来源证据，重新解析记录与校验 tensor bytes；SSH 使用 CPU 重新执行离线 analyze 和分析测试。没有重复 GPU 运行，没有修改实现。

## 实测和独立复核

| 场景 | 独立核验结果 |
| --- | --- |
| `value-custom` | cwd 相对 `--source` 指定含空格的 `inputs with spaces/custom gelu.py`，覆盖配置中不存在的文件；racecheck 通过，4096 条完整值记录 |
| `access-custom` | 配置目录相对 `../inputs with spaces/custom gelu.py`，driver 位于另一目录；memcheck 通过，4096 条完整访问记录 |
| `value-legacy` / `access-legacy` | 不传 `--source` 的原调用均通过，各 4096 条记录 |
| 原始来源 | 四次运行的 run/source/request/point 元数据一致；custom 两次的路径均为实际用户选择的原文件 |
| 原文件与 baseline 快照 | 读取后的源码文本一致，摘要匹配 run.json；内部别名保持 `source/kernel.py` |
| 进程与正确性 | 四次公开命令及八个 baseline/instrumented worker 均成功；输入/输出二进制摘要与实际文件匹配，baseline/instrumented 位一致，driver reference 通过 |
| 离线数值分析 | 原报告 matched=true；审阅者在 CPU 上重新 analyze，3 个比较全部匹配，报告显示用户原路径及行号 |
| 错误内容显式选择 | run/trace 两个真实命令均退出 1，拒绝修改后的 wrong.py，未回退到有效的 sibling kernel.py |
| TileOPs 六次接入探测 | 命令均显式传入实际上游 `--source`，到达 prepare 的准确 source/driver 契约拒绝；已不再是 FileNotFoundError |
| 分析回归 | 审阅者在远端 CPU 独立执行 `test_analysis.py`，8 tests PASS |

值记录通过 `verify_success` 从原始 GPU stdout 重新解析并与持久化记录逐项比较；访问记录通过 `parse` 校验完整 identity/coverage 后与持久化记录比较，并重新生成 access analysis/Markdown 与保存文件比较。sanitizer 检查使用原 worker 命令、进程返回值和原始日志，未仅依赖顶层 passed 标志。

选定文件原始字节采用 CRLF，Python `read_text` 归一化为 LF 后写入 Linux worker 快照。因此当前 `source_sha256` 是归一化后源码文本的 UTF-8 摘要，原文件与快照文本一致，但不应声称两个文件的原始字节相同。本次 tensor 的输出位一致是单独校验的二进制事实，不受该换行差异影响。

## 接入限制与证据含义

新的 `tileops-probes/summary.json` 明确保留空 baseline、`baseline_passed=false`、`debugger_status=unsupported`、`delivery_passed=false` 和 exitcode=1。六个子进程的实际退出码及 traceback 均经核对；这证明路径入口已推进至内容/driver gate，不证明采集语义或包/JIT 已受支持。

本次未新增任何源码/算子/CUDA 白名单项，未改变同步和布局校验。用户可以指定受审源码的任意名称与目录，worker 用该内容生成内部快照；任意用户源码、修改后的 driver 或外部包执行仍是后续通用接入工作。
