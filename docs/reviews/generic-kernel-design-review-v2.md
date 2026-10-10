# 用户 kernel 数值采集 v2：独立设计审阅

> 历史记录：旧 `trace` 已从当前产品移除；文中 trace 命令、专用测试和兼容性描述仅适用于[移除前提交](https://github.com/superAngGao/tilelang-debugger/tree/e838fc88d96a562ee6230ee154d6593bac6632f8)。保留当时结果，不代表当前产品能力。

**结论：FAIL，仅有两个源码支持域规则需补齐。复用现有 Python 打印宏的方向可实施；不再要求已撤销的 layout、lowered buffer remap 或 late IR erasure。**

- 日期：2026-10-09。
- 受审方案：`docs/generic-kernel-plan.md` v2，SHA256 `9bbcfedd413c474b3031ccfa9c5bfaa6aab5ff70f46d22b7bbd97dfba3a8049d`。
- 已核对 `monitor.py` 与 `docs/phase1-validation.md` 的宏结构/同步正负对照记录，以及既有 worker、TileOPs helper、source 路径语义。
- 本轮不修改实现、不运行 GPU。v1 的寄存器直读相关阻塞项随方案撤销，不继续沿用。

## B1：全 CTA 位置需要约束整个 serial 迭代域

规则 3 已拒绝 if/Parallel/while/pipeline/ws 包围观察点，但“常量 start/step”仍允许 stop 依赖线程。比如 `for k in T.serial(0, tx + 1)` 内的第 2 次采集，部分线程不进入宏，其余线程到达全 CTA 同步，仍会造成不完整参与。外层没有 if 并不能排除这个问题。

首版可采用最简单的规则：每层包围 serial/range 的 start、stop、step 在构建期均为静态整数，且同一 CTA 所有线程一致；不能确认则拒绝该观察点。拒绝会绕过观察点的线程相关提前退出/循环控制；循环 ordinal 必须在真实静态域内。参数化的静态整数闭包允许，但不能把线程 Var 当成常量。

这个要求只是新增集体同步的源码参与条件，不是要求证明原算法正确，也不需要读取或解析 lowering pass。应有一个线程相关 loop bound 的负测试，确保 launch 前拒绝。

## B2：global 只读输入的确认不能仅靠 out_idx 和显式 buffer 名

不在 out_idx 中只说明返回约定，不能证明内存只读。规则 5 的“源码直接写或 copy 目标拒绝”还漏掉别名/match_buffer、指针实参、用户 helper/macro 内写入同一输入。若其他 CTA 正在写，全局打印会额外引入读写竞争。

无需做通用内存分析。首版可以把 global 输入观察限定到明确可检查的源语法：真实前端参数 Buffer 的身份可绑定，所有写目标可解析且均与该输入无别名；遇到未知 helper、alias、指针/extern 写法拒绝 global 观察点。也可利用未 lowering 的 PrimFunc 中 Buffer/data identity 和可识别读写语义确认，但不能引入被用户撤销的 lowering 分析。要明确选择哪种保守规则。

应增加一个非 out_idx 输入经别名写入的负测试。RoPE 简单输入读取应继续能通过，不要求为了支持一般 alias 写法扩大本轮范围。

## 已接受的其他设计

- 默认 run source 引擎取消用户源码/driver/布局哈希准入，reviewed 显式保留，trace 保持原实现；边界清楚。
- fragment→shared→全 CTA 同步→单线程原位打印→同步，复用已验证 Python 宏。既有打印负对照说明内部同步位置有意义，无需为本轮重新设计另一套打印机制。
- Buffer/KernelLaunchFrame 的实际静态 shape/dtype/grid/threads 足以提供本轮必要元数据；不需要为了“通用”恢复底层 layout。
- 正常包加载和相对导入保持原环境，来源按实际 origin；compile/JIT 双绑定、禁用缓存、一次编译/launch、finally 恢复和异常传播已有明确规则。仍需代码与实际 H200 验证。
- 无 reference 为 not_checked，数值不匹配正常采集后退出 2，任意 driver 异常不能伪装成 mismatch；旧 schema 证据校验不放宽。
- RoPE 仅输入打印，不冒充中间 scalar 或访问索引；新 trace 通用化留后续，这是用户最新范围下可接受的明确限制。
- 既定三类真实采集、选点 reference、sanitizer、改名/空行/参数变化/错误计算、无 reference/异常 driver 与 reviewed 回归验收保留。

补齐 B1/B2 后即可复审进入实施，不要求恢复 v1 技术路线。
