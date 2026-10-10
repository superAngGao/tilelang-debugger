# 用户指定源码路径验证

> 历史记录：旧 `trace` 已从当前产品移除；文中 trace 命令、专用测试和兼容性描述仅适用于[移除前提交](https://github.com/superAngGao/tilelang-debugger/tree/e838fc88d96a562ee6230ee154d6593bac6632f8)。保留当时结果，不代表当前产品能力。

2026-10-09，完成 `run` / `trace` 源码输入修正。实际输入来自用户指定的 `--source` 或配置 `source`，不再查找 driver 同级 `kernel.py`。内部快照命名仍保留以兼容已有受审驱动；没有解除源码内容、driver 或算子契约。

## 路径规则

- 显式 `--source` 优先，相对路径基于 cwd。
- 未显式指定时，配置 `source` 相对配置文件所在目录。
- 原配置、选取方式、用户输入字符串、实际绝对路径保存于 `source.json`；run/point/request/数值与访问报告保存实际路径。旧产物无该字段时仍能显示。
- 缺失输入不回退到另一个文件；未知内容仍由既有契约拒绝。契约早期拒绝发生在输出目录建立前，尚不生成 `source.json`，探测中的原命令和请求配置单独保存。
- `source_sha256` 沿用已有语义：对读取并归一化换行后的源码文本计算摘要。此次原文件为 CRLF、内部快照为 LF，验证的是文本内容一致，不声称两个文件逐字节相同。

## H200 实测

在独立工作区把已有 GELU 源码放到 `inputs with spaces/custom gelu.py`，原驱动内容不变，放在 `drivers/launch.py`；该目录没有 `kernel.py`。配置位于另一个 `configs/` 目录。

| 验证 | 实际结果 |
| --- | --- |
| `run --source` 使用 cwd 相对路径，覆盖 config 内故意不存在的 source | 4096 条数值记录，baseline/instrumented 输出位一致、reference 通过，两 worker racecheck 零 hazards |
| `trace` 使用配置目录相对路径 `../inputs with spaces/custom gelu.py` | 4096 条访问记录，输出位一致、reference 通过，两 worker memcheck 零错误 |
| 原 GELU `run`，无 `--source` | 通过 |
| 原 GELU `trace`，无 `--source` | 通过 |
| 对改名输入的 capture 执行 CPU `analyze` | 通过；report 显示 `custom gelu.py` 的实际路径与源码行 |
| 显式选择追加注释后的源码，原 driver 同级仍有已支持 kernel.py | run/trace 均真实退出 1、报契约拒绝；没有回退运行原文件 |
| 三类 TileOPs 分别执行 run/trace，显式提供真实上游路径 | 六次均到达 source/driver 契约拒绝；不再是缺失 sibling 文件 |

TileOPs 的这次六项测试只运行 `--probes-only`，不重复上一轮已通过的数值矩阵；没有基线，因此整体退出 1。RoPE 的 run 请求仅探测全局 y buffer 的准入，不能表述为已支持无 fragment 的数值采集。三个外部 kernel 的通用接入仍未完成。

CPU：新增路径测试 5 项、更新后的 TileOPs 集成测试 8 项、原源码/记录 7 项、访问 5 项、证据 5 项，合计 30 项在本地和独立代码审阅中通过；Linux 上数值分析测试另 8 项通过。

环境沿用 H200 / TileLang 0.1.12 / torch 2.10.0+cu129。完整命令与真实进程状态在 `artifacts/source-path/*-process/`，采集和分析产物在相应子目录；复现驱动为 `artifacts/source-path/validation-driver.py`。证据包 `artifacts/source-path-evidence.tgz` SHA256：`45197454ca3e48dfd465e0834b6adc0d50570cb7641a3a4a5fd2ac6f1fefd26c`。

[设计审阅](reviews/source-path-design-review.md)、[代码审阅](reviews/source-path-code-review.md)、[独立验收](reviews/source-path-acceptance-review.md)。
