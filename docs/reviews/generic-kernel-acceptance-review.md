# 源码数值引擎：独立验收审阅

**结论：PASS，本轮源码数值采集目标完成，无剩余验收阻塞项。** Softmax/RMSNorm 已通过真实包/JIT 接入取得中间值，RoPE 已取得只读输入。通用 trace、RoPE 中间 scalar 和任意异步 kernel 不属于本次完成范围。

- 日期：2026-10-09。
- 本地证据：`artifacts/source-engine/` 下的 `release-source`、`release-source-edges`、`release-source-cpu`、`release-source-reviewed`、`release-source-mismatch`。
- 远端工作区：`/home/ang.gao/tilelang-debugger-source-engine-20261009`。
- 审阅方式：原始日志/记录/二进制快照重新校验，CPU 独立复算中间点与最终输出，独立重跑隔离 CPU suite，核查代码/包摘要。审阅者未重复执行 GPU kernel。

## 正式矩阵

独立确认矩阵完成后才作本结论：27/27，包括 21 组基础组合和 6 组 racecheck/synccheck 组合，共 54 个 baseline/instrumented worker。

| 类型 | 实际观察 | 每次记录数 |
| --- | --- | ---: |
| Softmax N256/N257 | row_max、row_sum | 2 |
| Softmax N513、tile_n256 | 原 serial 第 3 次 tile_sum，实际循环值 2 | 1 |
| RMSNorm N256 | sumsq、rrms、normalized | 258 |
| RMSNorm N257 | sumsq、rrms、normalized，含 padding | 514 |
| RoPE 两种 shape | 只读输入 x | 2048 |

覆盖 FP16/BF16/FP32、block `(1,0,0)`、对齐/非对齐及多 tile 路径。对全部 27 次采集，审阅者：

- 用当前 verifier 从设备 stdout 重新解析完整记录，校验重复/缺失/身份/位宽，重新校验输入/输出 snapshot 文件字节及摘要、两个 worker 的实际退出码、执行恢复状态和输出 reference。
- 核对 6 组指定 sanitizer 的 12 份实际 worker 命令和完成日志，均为零错误；其余 21 组未被冒称跑过 sanitizer。
- 从原位记录独立恢复 tensor，在 CPU 上另外编写数学计算复算全部 **49 个观察点、19,745 条记录**及全部最终输出，按预设容差全部匹配。RMSNorm 使用有效 N 的求和，RoPE 使用真实传入的量化 cos/sin 表。
- 核对 Softmax 第三轮的源码 ordinal 和实际 loop value，而非只看记录条数。
- 核对实际前端/设备 IR、CUDA、来源文本摘要及 worker request；示例目录的全部文件摘要与清单一致。baseline 标记未插桩，instrumented 标记已插桩，示例 helper 未自行冒称完整 capture 验收。

本次只用实际 Buffer/launch 元数据和源码插入宏；受审核心文件与代码复审 v2 SHA256 一致，没有新增 lowering/layout 分析。

## 普通接入、失败与错误值

九项边界证据经独立读取：

- package、renamed、same_file、wrong_math 四个真实成功采集各 128 条记录；包内相对 import、改 buffer 名/空行和同文件 driver 均成立。没有 reference 的原 capture 为 not_checked，离线分析独立得出匹配或不匹配。
- wrong_math 正常采集完成，离线分析退出 2；不是把执行异常当成数值差异。
- driver_error 保留真实 RuntimeError；missing_import 为 0 compile/0 launch 并失败。
- dynamic_loop 的实际错误来自线程相关 Add 表达式不能转成静态整数；out_of_range 为明确 ordinal 越界。两者插桩阶段均 0 compile/0 launch，hook 恢复为 true。
- global_alias 为实际 storage 重叠拒绝，发生于代理调用原 kernel 之前。execution 中 launches=1 是代理调用尝试计数，不能将它解释为该插桩 GPU kernel 已执行。

另外，真实 TileOPs RMSNorm 副本把平方计算每元素加 1，仍采集完整 **514 条记录**，两版输出位一致，driver 数值状态为 failed；示例继续完成逐点分析并最终退出 2。原参考得到 sumsq/rrms/normalized/output 的不匹配数分别为 1/1/252/1014。审阅者又按“平方加 1”的实际错误数学计算独立复算三个观察点，均与采集值相符，证明记录的是修改后的计算，而非仅产生了一个 mismatch 标志。

## 回归、CPU 与包

原 reviewed GELU/GQA 两次采集的四个 worker racecheck 均通过。重新校验完整设备日志和 tensor 快照后，审阅者还在 CPU 上分别重新运行离线 reference 分析，两个样例的各 3 个比较均匹配。

审阅者独立重跑 10 个隔离 suite，**64 tests 全部通过、无跳过**。审阅时首次误传旧 GELU 路径给要求 GQA 的 access fixture，导致缺少 frontend.json、该 suite 0 tests；这是审阅命令的 fixture 错误，已保留在远端 `artifacts/reviewer-source-cpu`。改用正确 `access-memcheck-final/gqa` 后完整 10 suite 通过，记录位于 `artifacts/reviewer-source-cpu-correct-fixture`。未删除这次失败，也未计为产品测试通过。

独立核对证据包 SHA256：`8dd3f1126113ae5658b68d8185fe035f596a510b4dd28a01efb48220bed1aa91`。

独立打开 wheel，将其中 21 个产品文件逐字节与当前 `src` 比较，全部一致。wheel SHA256：`eeb0c460e99e31eadfa21effa830f2aab540ff8f1c8e61bdc73a0e620aa4e540`。未发布包索引。

## 结论范围

当前 source 引擎确实不再靠用户源码/driver/布局摘要白名单接纳这批 kernel；真实包加载、JIT、源码变更和错误计算都取得了记录。打印沿用 Python fragment→shared→同步→原位 printf→同步，IR/CUDA 仅作为实际编译产物保存。

README 和验证记录正确区分了 source、显式 reviewed、受审 trace，明确 RoPE 仅输入打印、没有 reference 时不声称数值正确，也不把采集成功当成原算法/同步正确的证明。本 PASS 不扩大这些边界。
