# 阶段 1 独立设计审阅 v1

结论：**FAIL**。方案方向可行，不需要新增 lowering pass；但现在还不能把“证明安全后运行”交给实现者自由解释。以下 B1–B3 必须写入方案并复审通过后实施。

审阅日期：2026-10-08。审阅者为独立 reviewer session，未参与方案编写。本轮只读检查指定方案、README、实验 README/代码、四个交付 kernel，以及本地 `tilelang-v0.1.12-source` 的相关打印、缓存、WGMMA 和 ThreadSync 源码。未运行 GPU，未修改作者文件。本地源码只能说明需要核实的实现机制，不能代替目标 H200 安装版本的证据。

## 已成立的设计选择

- 源 AST 定位、临时源码插桩、原始/插桩独立编译、从实际执行对象导出 IR，边界合理。
- 单目标 compile/launch、逐元素完整记录、严格完整性校验，显著降低 launch 身份和日志拼接歧义。
- 明确区分默认 ThreadSync 实验、关闭同步时的竞争报告、可见错值，以及完整 GQA，现有实验的结论没有夸大。
- 不把 shared barrier 当作 WGMMA/TMA 完成等待；monitor staging 前后都使用相同参与域的 barrier，方向正确。
- 保留原始 compile 配置、完整算法与 producer/consumer 协议是必要条件，不能为了通过采集而随意关闭或开启 ThreadSync。

## B1：安全点的接受规则仍未落地，四例与规则之间存在具体缺口

第 5 节说由 AST 和基线 IR “核实”域、就绪和资源，但没有给出支持的判断规则、失败条件或四例的具体锚点。只看到一个同名 `wait(0)`，不能证明选中 buffer 是它覆盖的 accumulator，也不能证明整个 tile 已初始化、点被所有参与线程一致执行，或 shared 尚未被其他组释放复用。最终 IR 里也可能有重名/重写后的 buffer，不能仅按变量名配对。

具体例子：

1. GEMM 的 `wait_wgmma(0)` 和 `warpgroup_fence_operand(c_local, num_regs=64)` 位于 `ki → s → if stage == s` 内。第二轮观察需要明确选择 `ki` 与 `s`，证明 `gi_cons`/`stage` 组内一致；或者在整段 ring dispatch 后选一个已证明的锚点。不能同时要求逐层选全部循环又只在示例中写“第二轮”。
2. GQA 两个 consumer 的 QK prologue 都只有显式 `wait_wgmma(0)`，没有该 wait 后的显式 operand fence。检查的本地 WGMMA emitter 会在异步发射代码尾部生成 accumulator fence，但它出现在用户后续 wait 之前；`wait_wgmma` helper 自身仅调用 `cute::warpgroup_wait`。这不支持直接把“wait 后必要 fence 已存在”当作事实。本审阅不据此断言原 GQA 错误，但方案必须消除自己提出的安全前提与实际目标点之间的缺口。
3. Sum 在某个行循环尚未完成时，whole `x_f32` 可能只初始化部分行。允许整个 `T.Parallel` 后采整 buffer 并不自动安全。

最小修订：列出四例准确的语句锚点、before/after、buffer 词法作用域、完整 enclosing loop 选择、线程区间/leader、初始化证据和异步就绪证据。给出有限的可机械验证规则，未知调用、未知写入/别名、未知循环一致性或 buffer 生命周期一律拒绝；不必建设通用程序验证器。GQA 应明确采用哪一种可核实策略：证明实际生成代码满足所需 compiler operand ordering，或在监视宏中采用已有 operand-fence intrinsic（不增加 wait、不改变原 barrier 协议），并明确寄存器覆盖范围从哪里取得。不能以一个 kernel 名或 buffer 名作为安全证明。

## B2：实际编译和验证的时序、缓存策略不足以保证 IR 与运行一致

第 7 节“独立 worker 隔离全局状态及缓存”对磁盘缓存不成立。本地 `tilelang.compile` 进入持久化 cache，`JITKernel.from_database` 可以恢复 adapter 而不产生 `artifact.device_mod`。独立进程仍可命中相同用户缓存，导致承诺的实际 device IR 缺失。不得重新 lower 一个副本再冒充本次 artifact。

另一个问题是插桩会改变 ThreadSync 的插入。只扫描基线已用 ID 无法保留新增自动同步所需 ID；方案说编译后核查，但没有明确这是 launch 之前的强制门禁，也没有说明怎样区分 monitor barrier 与自动生成同 ID barrier。两个 worker 的顺序与基线 manifest 交接也应明确，避免在资源检查所需证据尚不可用时执行插桩程序。

最小修订：

- 在 import TileLang 前设置每个 worker 的独立空 cache/temp 目录，或使用目标版本受支持的禁用缓存机制；明确要求本次真实编译返回非空 device IR，缺失则在 launch 前失败。
- baseline 编译导出资源/配置 manifest；插桩 worker 读取该 manifest，构建、编译并核查实际插桩 IR/CUDA，然后才允许目标 launch。baseline 可先完成一次运行，但插桩不能绕过门禁。
- 对 named barrier 采用保守唯一分配，每个 point 一个 ID 即可，无须引入并发分析框架。记录完整参与范围、ID 和 count；在实际插桩 IR 中核实所有新增及原有 barrier 的使用者，不是只比较 ID 集合。碰撞或无法唯一归属时拒绝，保留编译产物。
- 明确两个版本实际 target、execution backend、pass_configs、compile_flags、out_idx 的差异清单；任何影响同步的差异必须显式判定，不能混入“采集相关配置”。

## B3：数值保真承诺尚缺最小设备端协议与验收

第 6 节的位模式方案可以实现，但已有实验只验证 int32 的原版格式。CPU bits 还原测试并不能验证 GPU 路径未把 fp16/bf16 先转成 float、未改变 NaN payload/负零，也不能验证 `printf` 参数类型与格式匹配。计划声称所有四种 dtype 原位保真，验收却只写一般性的“元素数量与值”。

最小修订：明确记录的固定 schema/version 和合法 point ID 编码；在 shared 原 dtype 上先做同宽 bit reinterpret，再将 uint16 零扩展成与 `%u` 匹配的 uint32，所有身份与该值在一次 printf 调用中输出。给出运行期 block/loop 实参如何进入打印宏，以及 points.json 如何从该宏收到的实际 buffer 对象按唯一 point ID 绑定，不能依靠重名 buffer 或输出顺序。限制 printf 参数个数/支持的循环嵌套深度，明确预算覆盖格式和参数开销。

增加一个最小 H200 端到端保真验收：四 dtype 的已知原始位模式从输入经 fragment staging 到记录逐位相等，包含正负零、Inf、至少不同 NaN payload、极端整数；同一值编码也覆盖直接 shared 路径，或将 shared 支持推迟。输入快照在 launch 前保存，比较两次输入的 dtype、shape、stride/必要布局及原始位模式；对 NaN 不使用普通数值相等。最终数值 reference 的容差、baseline/instrumented 比较规则必须预先写明，不能让相同的过宽容差掩盖 monitor 改变计算。

## 非阻塞精简建议

- 保持一个相邻 kernel.py 的限制；不需要插件框架，也不需要在这一阶段复刻 tilesight 的完整 actor/synchronization 类型系统。
- 默认拒绝 T.Pipelined/T.unroll 内部采集，先完成四例，不要以“验证过的简单形式”形成没有列举的开放入口。
- 将“发布可复用状态前”的 shared 观察约束写成明确拒绝规则；四例若只验证 fragment，就不要提前将任意 shared 宣称为支持。
- README 的广义路线图可以保留，但链接本阶段文档并明确 reference 产品接口、访问诊断和报告不属于本轮交付，避免范围混淆。
- 实验 matrix 自身不汇总 exit status 的已知限制已有说明；正式验收入口应对任一子项失败返回失败。

复审通过标准：B1 的具体锚点与有限证明规则、B2 的真实编译及 launch 前检查时序、B3 的协议和设备端保真验收都成为明确方案内容。无需本轮先实现这些功能或运行 GPU 才能做设计 PASS；但不能只再增加一句“实现时检查/验证”。
