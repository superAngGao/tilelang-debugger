# 第四步独立代码复审 v2

> 历史记录：旧 `trace` 已从当前产品移除；文中 trace 命令、专用测试和兼容性描述仅适用于[移除前提交](https://github.com/superAngGao/tilelang-debugger/tree/e838fc88d96a562ee6230ee154d6593bac6632f8)。保留当时结果，不代表当前产品能力。

**结论：PASS（当前固定四例范围内，v1 的 B1/B2/B3 已关闭，无剩余代码阻塞）。** 本次代码复审及已保存 racecheck 证据通过，不代替仍在进行的 synccheck、memcheck 和旧功能完整回归验收。

日期：2026-10-09，Asia/Shanghai。复审当前 `access*.py`、相关测试/协议实验和既有产物。审阅者没有修改实现、没有启动 GPU；独立运行了纯 CPU/TIR 测试并只读重验远端数据。期间提出的负起始 TMA store 分类问题由实现者修复，以下结论包含该最终修复。

## 受审快照

本地与远端实际 `src/tilelang_debugger` 实现摘要一致。没有使用远端 workspace 根目录误放的同名文件。

| 文件 | SHA256 |
| --- | --- |
| access.py | `c95b4a608b24f34b69d3387443a6c31882d73fd07e8ec9db7343d7bfe597b0b0` |
| access_contracts.py | `e996b6938d9eb70fdbde8a9a9e8d1dcbdb8cd074225e2feb7d921ebf743ba101` |
| access_host.py | `5ba1ea70edc3a32d820d399c0a1d6d721657fe00772ef953bff179509f948d5b` |
| access_ir.py | `0c6db5862963e252971be53fddc18f44c9ba0376574cc7151c76c04450f5e9d2` |
| access_records.py | `fbf4e2101cf58ea8876f10bec82c02cff2a703348a0cb3033639f2f783a46735` |
| access-pins.json | `96bf286c3dc6e2ac434c0abfb9ab7e00d915c08e5c790313a642858521842210` |
| tests/test_access.py | `966260fbfbfcc9b0edb6d6711df3964b3500f22eef04ab4346d782918c64cd1e` |
| tests/test_access_ir.py | `3951e378c71425014d75dc2efb142573ba4899d6b079842eb099aab017a2054d` |

本轮结束时方案 `docs/phase4-plan.md` SHA256：`a274852108a1ec9444232ba9a8c3b42ee503934c63b76f614bee55bf63f95ec8`。

## 阻塞关闭情况

**B1：host 门禁已修复。** 遍历改为允许的语句/调用列表，拒绝构造或 launch 所在的未知控制流；packed args 根、Select discriminator、tensor 参数位置、shape scalar 来源及 Cast、device scalar 与 launch dimensions 均增加检查。descriptor metadata 对实际 constructor 的结构核对、stack allocation、实际参数位置、未审调用/额外用途限制仍保留。v1 的零次构造循环、两次 launch 循环、错误 Select 条件三个反例现在均被拒绝；错误 scalar Cast 和 descriptor 参数也有拒绝验证。

**B2：racecheck 解析已修复。** 按工具识别不同成功摘要，检查 tool command、外层进程退出、pipeline 恢复和 launch gate。已保存的真实干净 racecheck 日志可通过，非零进程状态仍失败。

**B3：TMA 区域解释已实现。** 事件按 descriptor 身份关联，区域来自记录坐标加实际 host 绑定推导的 box，随后转换为原 tensor 轴顺序；shared offset/barrier 来自记录，swizzle 不伪装成逐元素物理地址。缺 descriptor 拒绝，来源显式标为推导。复审中发现的负起始 store 误标合法丢弃已修成 `invalid_store_origin`；非负起始的尾部越界继续分类为 `store_oob_discard`，相应 CPU 断言通过。

## IR 与来源检查

普通/vector 原访问节点和原外层分支仍被保留；日志取自原索引/调用实参，不复写为验收公式，不额外加载目标数据。新增 gi_prod/eff、guard/election 验证与有限源码、baseline CUDA 摘要、pipeline 同边界比较共同限定当前支持域。

Var-keyed descriptor metadata 的规范化只作用于比较副本，要求唯一键名及 key 与参数中 descriptor Var 同一身份，完整参数值仍参与比较。实际模块未修改。新增真实 codegen 输入上的 trace 擦除比较，检查 printf 格式、参数 ABI 和静态 site 数，并与 baseline codegen IR 比较，补足仅验证 pipeline 边界的不足。

以上结论针对固定契约和节点形式，不意味着提供任意 host/device IR 的通用验证器，也不意味着 printf 不影响执行时序或物理内存事务。

## 独立验证结果

使用远端 `/home/ang.gao/miniforge3/envs/tilesight-tl012/bin/python`，先导入 TileLang，设置 `PYTHONPATH=src` 与 `TLACC_FIXTURE=artifacts/access-gqa-03`，独立执行 `test_access*.py`：**10 项通过**。负起始 store 修复后再次执行，仍为 10 项通过。仅加载已有 IR，没有编译或启动 kernel。

对 `artifacts/access-racecheck-final` 的五条路径独立重验：worker 退出/恢复/tool 摘要、输入输出实际快照字节及摘要、原 stdout 重解析与 JSONL 一致、baseline/pre-trace 等价、实际 codegen trace 擦除等价、analysis 重算与保存结果一致，均通过。

| 路径 | 完整记录数 |
| --- | --- |
| GELU | 4096 |
| Sum N257 | 512 |
| Sum N256 | 1280 |
| GEMM | 2 |
| GQA | 6 |

对 `artifacts/access-protocol-03` 原始 stdout 独立重放 **65,536 条**记录，逐项核对唯一身份、INT64_MIN、负数、超过 2^31/2^32 的值、UINT64_MAX 和实际错位 load 索引，全部通过；独立保存的外层 process 状态为 returncode=0、timeout=false。最终分类修复后另重算 GEMM/GQA 已保存区域结果，仍一致。

## 非阻塞后续

完成方案已要求的 synccheck/memcheck 和旧回归，保留失败尝试及外层退出状态。最终交付应将本代码 PASS 与各项验收结果分别列明；不要把一次代码复审自动表述成全部 GPU 验收完成。
