# 第四步独立代码审阅 v1

> 历史记录：旧 `trace` 已从当前产品移除；文中 trace 命令、专用测试和兼容性描述仅适用于[移除前提交](https://github.com/superAngGao/tilelang-debugger/tree/e838fc88d96a562ee6230ee154d6593bac6632f8)。保留当时结果，不代表当前产品能力。

**结论：FAIL（针对下列受审快照）。** 普通访问插桩的基本路径合理，但 host 门禁存在已复现的错误接受，racecheck 验证器会拒绝真实干净日志，TMA 区域解释尚未落地。

日期：2026-10-09，Asia/Shanghai。对照已通过的方案第 12/13 节，阅读 `access*.py`、pins、例子配置、capture helper refactor 和 CLI。本轮没有修改实现、没有运行 GPU；在远端固定 TileLang Python 环境仅加载已有 IR，运行 CPU/TIR 验证器负例。

实现者在本次审阅过程中根据即时反馈继续修改代码。因此本报告固定首次读取/测试的快照，不将后续修改自动视为已复审。尤其 `access_host.py`、`access_ir.py`、`access_records.py` 在落盘前已发生修改，需下一轮确认修复。

| 文件 | 受审原始字节 SHA256 |
| --- | --- |
| access.py | `b6e1f095a72454ee55dd16e51f975192e0ccf2ca40574065dfac23c3d35a0bc5` |
| access_contracts.py | `e996b6938d9eb70fdbde8a9a9e8d1dcbdb8cd074225e2feb7d921ebf743ba101` |
| access_host.py | `7e5d541e4d1a9a0e74af644b92ae21281ee08c9764c70084c067a9719d6f9636` |
| access_ir.py | `7ef5b6c997b0f689ae528bfba9852d5447f5b3222505679ea39a3734034d1e3b` |
| access_records.py | `da6be73ec74ffdf62be5fad6480a1012594c0afd38cd8c5c145bf56eccccff47` |
| access-pins.json | `96bf286c3dc6e2ac434c0abfb9ab7e00d915c08e5c790313a642858521842210` |
| capture.py | `4721bf274a53be126cecd0b2fccf22ddd8154b92a10b69d9e1de75f5962e59d9` |
| cli.py | `8f27aa27af5b237a0c3ac30f0a5025b6ad09085406b43f6f83d9bc6a188cf8c7` |

## B1 / P1：host 门禁没有证明其声称的构造支配、单次 launch 和 tensor 身份

位置：受审 `access_host.py:47–60`、`:78–84`、`:105–128`、`:140–157`。

`visit()` 对任意带 `body` 节点直接递归，故会把循环体中的语法调用当成无条件一次执行。所谓构造支配只检查共同 SeqStmt 中的位置，不排除零次循环。`handle_index()` 只识别 Select 的两个分支，未验证条件或 packed args 根；`scalar()` 对 shape load 外的 Cast 只取 shape 数值，没有验证/执行该 Cast，也未完整验证 shape 提取链。

使用已有 `artifacts/pipeline-probe-gqa/pipeline.json`，在 CPU 上构造等形状的原 PrimFunc 参数映射，原模块 `HostBindings` 通过。随后每次只改 host、保持 device 不变，得到：

| 负例 | 当前受审验证器结果 |
| --- | --- |
| 将 Q descriptor 构造 Evaluate 包进 extent=0 的 For | ACCEPTED，仍声明 `constructor_dominance=True` |
| 将唯一目标 launch Evaluate 包进 extent=2 的 For | ACCEPTED，仍声明 unique launch/dominance |
| 将 Q_handle 的 Select 条件改为常假，保留两分支 | ACCEPTED |

这些验证未编译或执行 kernel。问题在于预启动证明不成立，不能据此信任 descriptor 推导或声明只有一次实际 launch。固定 baseline CUDA 摘要无法补足：三例完全不改变 device CUDA；baseline/pre-trace 相等也只能说明两次 host 相同，不能说明它符合授权构造路径。

最小修复：host statement 使用当前 packed ABI 所需的显式允许列表，拒绝 For/未知控制流；核对真实 packed args 根、类型 discriminator 与 Select 条件、tensor data/shape 提取索引及所有相关 Cast。验证无隐藏构造/launch/descriptor 写入，且 device 动态 scalar 的 launch 实参与推导所用 shape 绑定一致。保留上述三例为拒绝测试，并加入错误 shape 来源/Cast、descriptor 重写或交换负例。不要只为使现有模块通过而删除验证。

## B2 / P1：racecheck 的成功摘要被错误解析

位置：受审 `access_records.py:60–72`。

所有 sanitizer 都要求 `ERROR SUMMARY: 0 errors`。现有真实 `artifacts/phase3-racecheck-final/gqa/baseline/racecheck.log` 只有：

```text
========= COMPUTE-SANITIZER
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

因此正常 `trace --sanitizer racecheck` 也会失败，当前实现不能完成方案要求的 racecheck 验收。这是解析逻辑缺陷，不是缺少新的 GPU 运行。

最小修复：按 tool 分别验证 racecheck 的零 hazards/errors/warnings 和 synccheck/memcheck 的零 errors；同时核对保存的 command 与请求 tool 一致，保留 process exit、timeout 和 pipeline 恢复检查。用已保存的干净/非零摘要验证，不需要为修这个解析器重新跑 GPU。

## B3 / P2：TMA 报告还没有实现区域与边界解释

位置：受审 `access_records.py:75–87`，关联 `parse()` 的 TMA 分支。

TMA records 只保留 coordinates/shared offset/barrier index；`report()` 将 descriptors 原样附上，然后只统计条数和 active/masked。没有把每条事件连接到对应 descriptor，使用实际 coordinates、box、mode 顺序和 shape 展开逻辑区域或判断边界。报告却整体标记 complete，limitations 中还写着 TMA regions are derived，容易让读者以为已经生成该推导。

最小修复：仅对已受审 rank2/rank4 tiled 配置，按本条事件实参和已核实 descriptor 生成区域起止、logical tensor 轴映射及方向相关边界结果；标明 derived provenance，shared swizzle 仍只显示原参数。未知 descriptor/配置拒绝；无法解释的项目明确 unknown。分别覆盖 GQA 两个输出区域及 GEMM 第二轮 K 区域。不要求逐元素物理地址追踪或任意 kernel 扩展。

## 已接受部分与复审注意

- 原访问语句仍作为插桩 SeqStmt 的原 child 保留；日志从原 BufferLoad/Store 的索引或 TMA call 实参提取，不加载目标数据。Sum 的 active 条件来自原 if_then_else，日志位于原语句前；原 vector load/store 没有拆成 scalar 数据访问。移除新增节点后检查同边界 IR 等价，方向正确。
- source/driver 摘要、有限 AST 规则、固定 baseline CUDA 摘要和额外局部 loop key 能限制来源映射。受审初版对外层 guards、GEMM gi_prod 和 GQA eff 仅保存/粗放接受，未完成方案第 12.2 节证明；实现者已开始补充，下一轮必须检查这些规则与相应拒绝负例，不能仅确认函数存在。
- `tma_descriptor_args` 的 Var-keyed Map 在序列化后可能导致模块结构比较不稳定。可在独立比较视图中将其规范化为唯一稳定键及完整参数列表，但须先验证原 Var key、metadata 内 descriptor、实际 constructor 和 launch 的身份一致；实际 IR 不得修改，完整值必须继续比较。不能删除该属性或仅比较键名。
- capture helper refactor 保留旧入口的 cache/environment 设置和默认 `_worker`，CLI 对 trace 独立分派；未发现这部分引入的数值 capture 语义回归。
- 新增日志经过后续 LowerIntrin/Simplify/HoistBroadcastValues 后仍须核对实际 codegen IR/CUDA。当前导出这些产物是正确基础，不能把 `ir_erasure_verified=True` 当作最终 CUDA 访存/同步语义已验证的替代品。

本轮不把尚在进行的 GEMM/GQA GPU 接入、测试数量或文档完善另列为阻塞。下一轮以修订文件和上述针对性证据复审，不自动沿用本次 FAIL 或把即时修改自动升级为 PASS。
