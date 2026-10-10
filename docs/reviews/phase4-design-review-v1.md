# 第四步设计独立审阅 v1

> 历史记录：旧 `trace` 已从当前产品移除；文中 trace 命令、专用测试和兼容性描述仅适用于[移除前提交](https://github.com/superAngGao/tilelang-debugger/tree/e838fc88d96a562ee6230ee154d6593bac6632f8)。保留当时结果，不代表当前产品能力。

结论：**FAIL（方案需小范围补全后复审）**。pipeline 末端追加日志的方向具有可实施性；阻塞在源码关联/覆盖契约和 TMA 证据链尚未具体化。不是要求实现通用 kernel 追踪架构，也不是因为尚未完成 4A 实验而判失败。

- 审阅日期：2026-10-09，Asia/Shanghai。
- 受审文件：`docs/phase4-plan.md`，草案，未实施。
- 受审文件原始字节 SHA256：`10d21bfee9d043467f046a629160e2aef2e10f3252f4ba2f39ffc8cb7dc93479`。
- TileLang checkout：`2d63708c8ad57196051c4636a1167c5453c73a48`，已通过 `git rev-parse HEAD` 核对；源码树无本次修改。
- 本次为独立审阅 session，根据文件和产物重新建立判断，未借用以往审阅结论。
- 本次只新增本报告；未修改原方案或产品代码，未启动 GPU，未执行 TileLang 编译。

## 1. 范围与证据

阅读了方案、`src/tilelang_debugger/{capture,ir,evidence,instrument}.py`、四例源码及相关驱动，并核查 `artifacts/phase3-racecheck-final` 中 GELU、Sum N257、Sum N256、GEMM、GQA 的 baseline device IR/CUDA、编译配置、环境信息和现有数值采集结构。

重点核对 TileLang 的 `tilelang/cuda/pipeline.py`、`tilelang/backend/pass_pipeline/pipeline.py`、`tilelang/engine/lower.py`、`src/op/copy.cc`、`src/cuda/op/copy.cc`。补充核对了 CUDA codegen、`lower_hopper_intrin.cc`、`src/cuda/runtime.cc`、JIT adapter 和 election helper。

现有五组产物的执行 backend 均为 `tvm_ffi`。本地 examples 与 baseline 保存源码按 UTF-8 文本读取、统一换行后完全一致；原始文件字节不同是换行差异，不能声称两者原始字节 hash 相同。现有 artifacts 证明原数值路径曾运行，不证明新 access 插桩已可用。

CPU 检查仅包括上述身份/文本核对与已显示索引域枚举：GELU 每 block 的 global 访问分量为 2048 个且覆盖无重复；Sum N257 所选一行的 512 个候选分量包含 257 个 active、255 个 masked。当前 Windows Python 未安装 `tvm`/`tilelang`，没有运行 TIR 构造/变换验证。此限制不影响以下源代码与设计层面的结论，也不把 4A 的待验证假设当成已验证事实。

## 2. 阻塞 B1：固定源码映射和事件域仍是目标，尚不是受审契约

**位置：方案 94–100、125–127、143–146 行（第 6、8、10 节），关联第 4 节表达式限制。**

第 6 节列出了 buffer 身份、方向、索引结构、循环、分支、descriptor 来源等判断依据，但未列出首版实际支持的源码点、对应 lowered site、对象重建/共享内存别名规则、序列化稳定身份、事件基数或失败条件。“先对已有真实 IR 建立有限映射”仍留在将来。对于本方案的核心正确性，这不能只留到实现者自行决定。

末端 IR 已经历 buffer flatten、storage rewrite、shared allocation merge；跨 lowering 不能直接把 Python 前端 Buffer 对象身份当作末端身份。现有 `instrument.inject()` 自己就注明 lowering 会克隆 Var，并通过词法作用域中的唯一重命名处理两个 GQA fragment。新路径又不打算借用该插桩，故应明确自己的有限匹配依据。只靠名称或顺序虽已被禁止，但替代规则尚未落到具体点。

实际产物有以下足够小、可直接写入方案附表的受审基础；这里的公式仅用于核对 site 和事件域，运行时日志必须继续取原节点表达式。

| 原源码位置 | 现有 baseline lowered 证据 | 需要明确的事件域/身份 |
| --- | --- | --- |
| GELU `kernel.py:87`、`:90` | `gelu/baseline/device.py:19` 开始：输入/输出均为宽 8 的 vector，局部 `i` 为 0..1；global 分量偏移 `bx*2048+i*1024+tx*8+lane` | 输入 read、输出 write 各 2048 分量/block；若同时观察 local 端，另计其 site，不能把两端混成一条事件 |
| Sum `kernel.py:35`，外层 serial `:33` | `sum/baseline/device.py:24` 附近：`i_1=0..3`，`tx=0..127`，条件 `i_1*128+tx<257`，offset `bx*514+i*257+i_1*128+tx` | 选定一个原始 `i` 后 512 个候选，257 active、255 masked；主键必须含 lowering 的 `i_1` |
| Sum `kernel.py:42`、`:47` | `sum_unpadded/baseline/device.py`：global→shared 宽 4；shared→local_cast→float32 fragment 宽 2；shared 指针由合并分配基址偏移 0 导出 | 前一个 copy 共 512 个 global/read 与 shared/write 分量；后者每个原始 `i` 为 256 个 shared/read 分量；明确 cast 临时 buffer 的角色和配对范围 |
| GEMM `kernel.py:93`、`:98`，loops `:70`、`:77` | `gemm/baseline/device.py:45–56`：producer `tx<128`，`stage==s`，同一个 elect 分支内 A/B `tma_load`，`ki*64`，shared `s*8192` | 原 `ki`、`s` 与源行选择如何绑定；第二轮 `ki=1` 的有效 slot 为 1；各 transfer 一个组级事件 |
| GQA `kernel.py:114`、`:115`、`:120`、`:128` | `gqa/baseline/device.py:102–118`：Q 两个静态 site，K/V 各一循环 site，`256<=tx<288`，elect(32)，slot `k%2` | Q 两个 site 不合并；K/V 每个选中 `k` 各一个事件；当前实际 KV head 坐标为 0，不能用未经验证的泛化 head 公式冒充日志 |
| GQA `kernel.py:293`、`:455` | `gqa/baseline/device.py:315`、`:517`：两处 `tma_store`，consumer 组不同，Os 起始 offset 0/4096 | 两个不同静态输出 transfer site，每个 consumer 组各一事件；不能把 shared 上的分量 store 误认作 global 输出 store |

覆盖还涉及已有的可变局部标量。GEMM `stage` 来自 `gi_prod_1[0]`，后者是 `local.var` BufferLoad，且每轮递增；GQA producer 循环界为 `eff_1[0]`，由动态 `seq_len_kv` 算出。这不是未来任意 kernel 才有的情况。方案应说清“既有标量绑定”的允许范围、支配关系与本次 shape 参数来源，否则“含 BufferLoad 拒绝”和“必须通过 4C/4D”可能在实现时冲突，或为通过案例而错误放宽为任意数据依赖。

**最小修改：** 在方案增加一份仅涵盖上述固定源码点的映射/覆盖附表及规则：源文本/driver hash、AST 作用域锚点；末端参数位置或 allocation/alias 链、访问角色、结构路径和允许的索引骨架；确切 site 数、原循环映射、额外局部循环、参与线程/组及事件计数；零匹配、多匹配、未认识别名/节点的拒绝条件。说明稳定 ID 由受审 source/site 身份构造，而不是内存地址或偶然遍历顺序。对 GEMM 的计数器递推和 GQA 的 shape-derived `local.var` 给出有限验证规则，只读取已存在的纯标量绑定或在原安全作用域中使用它，不泛化支持 tensor 数据索引。

补充明确：索引骨架核对和事件身份验证不得用“期望坐标替换实际坐标”；负例 fixture 应在自己的受审映射中允许该错误索引被匹配、记录并显示。无需通用跨 pass provenance。

## 3. 阻塞 B2：TMA descriptor 真实来源、操作方向与实参解析缺少闭环

**位置：方案 22–26、54、86–92、113、133、145–146 行（第 2、3、5、7、9、10 节）。**

方案正确要求从实际 descriptor 构造信息关联 shape/stride，但没有说明在当前 backend 哪里取得、如何保留、如何绑定本次 launch。仅保存现有 frontend/device IR/CUDA 不够：

1. `src/cuda/op/copy.cc:1836–1872` 构建 descriptor 的 mode 顺序、byte stride 和 box，并调用 `create_tma_descriptor`。该布局不应从原 tensor shape 的直觉顺序猜测。
2. `src/cuda/transform/lower_hopper_intrin.cc:43` 保存 `tma_descriptor_args`，`:182–199` 将 device call 内的构造调用替换为 descriptor Var；`:253–261` 构建 host 侧 `tvm_call_packed` 初始化。
3. 当前 `tvm_ffi` 路径最终由 `src/cuda/runtime.cc:540–550` 的 `__tvm_tensormap_create_tiled` 调用 `cuTensorMapEncodeTiled`；实际参数是 `globalStride + 1`，IR metadata 中的完整 stride 数组与 Driver API 接口数组不能混淆。`jit/adapter/wrapper.py` 的 C++ wrapper 也能说明参数格式，但不能把它当作这批 `tvm_ffi` 产物实际执行的 host 路径。
4. 现有 `ir.export()` 只写 frontend/device 与默认 kernel-only CUDA；`JITKernel.get_kernel_source()` 默认 `kernel_only=True`。现有 GEMM/GQA device 参数只是 `grid_constant` opaque descriptor，不含构造详情。历史产物不能支持对真实 descriptor shape/stride/box 的完整证明；本次方案必须承诺增加 host 证据，而不是凭 device Var 名称重建它。

还存在 load/store 实参语义差异。实际 GQA 的两处 global 输出为 `tma_store`，不是普通 BufferStore。方案的 TMA 字段仅明确“shared 目标偏移、barrier 身份”，这适用于 load；store 的 shared 是源、global descriptor 是目标，而且没有 load 的 mbarrier 参数。`operation=transfer` 可以保留，但 manifest 至少需要明确方向、两端身份以及 barrier 可空。

固定 checkout 的 CUDA codegen `src/cuda/codegen/codegen_cuda.cc:2524–2558` 将 `tma_load` 最后一个参数解释为 eviction policy；`:2595–2611` 将 `tma_store` 最后两个参数解释为 reduction flag 与 eviction policy。故 GQA IR 中 load 尾部的最后一个 0、store 尾部的最后两个 0 都不是 tensor 坐标。方案目前没有定义这一受审参数 schema，容易把 rank 或区域算错。barrier 参数虽然在 IR 中表现为 BufferLoad，也不能为打印“身份”去读取 barrier 内容，应记录 allocation 身份和索引。

**最小修改：** 增加当前 `tvm_ffi` 的 TMA 小节：在 pipeline 返回模块中保存未修改 host IR/descriptor 构造 metadata；沿 host launch 参数与 device 参数位置建立 descriptor 关联，绑定实际输入/输出 tensor 的身份、shape/stride 及动态 scalar 值，核对构造路径和运行成功；把静态元信息、本次 launch 实参、离线推导分别落盘。对本批固定 tiled descriptor 明确 dtype/rank、mode 顺序、byte stride、boxDim、elementStrides、interleave、swizzle、OOB 和调用方向；实现可限制为现有配置，不需要解码任意 opaque descriptor。缺构造证据或不支持配置时拒绝区域解释/该点，不能假装知道。

同时给出 `tma_load(desc, barrier, shared_ptr, coords..., eviction)` 与 `tma_store(desc, shared_ptr, coords..., reduction, eviction)` 的受审 schema，store 保留原 arrive/wait、不新增 barrier；明确 GQA 输出两点属于 store transfer。descriptor ID 应是本次构造与形参绑定身份，不是“某个地址值就是 descriptor 内容”。动态字段可以由已保存的真实 launch 绑定求值并标为推导，不必新增 descriptor 内存读取或 C++ lowering。

TMA 方向相关边界解释的现有原则正确。NVIDIA 文档说明 descriptor 不透明，包含 dimensions/strides/box 等构造参数；load 的越界部分可填零，store 对负起始坐标有不同限制。继续依据方向和配置解释，不要把 `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE` 简化成“所有越界均非法”。参见 [Tensor Map API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TENSOR__MEMORY.html) 与 [CUDA 13.2.1 async copies](https://docs.nvidia.com/cuda/archive/13.2.1/cuda-programming-guide/04-special-topics/async-copies.html)。

## 4. 已接受的设计及非阻塞实施建议

### 4.1 pipeline 边界可行，但保证必须分层验证

`PassPipeline` 注册可替换，`engine/lower.py` 在 `pipeline.lower()` 返回后才 Filter host/device。CUDA pipeline 已完成布局、vectorize、storage rewrite、host/device 拆分和 ThreadSync。独立 worker 中临时包装 CUDA registry、调用原 lower 一次、只改目标 device body、finally 恢复，符合本任务范围。4A 验证 Python IR 构造能否通过编译，是合理的实验，不要求本轮先实现才能通过设计审阅。

需把第 92 行的等价性边界写清：`erase(trace_at_pipeline_exit) == pre_trace_at_pipeline_exit`；baseline/pre-trace 另作同边界比较。后续 `_prepare_device_codegen_mod()` 还依次执行 LowerIntrin、Simplify、HoistBroadcastValues，`artifact.device_mod` 是此前的模块，不能把它直接称作最后送 CUDA codegen 的 IR。要保存该真实边界，需单独捕获该模块；跨不同 pass 边界不直接要求结构相等。

移除日志结构相等只能证明原 IR 未改，还要验证每条日志的索引、guard、vector component 来自所关联节点，以及新增求值本身安全。方案已要求最终 CUDA、输出逐位、sanitizer 检查，方向正确。不得声称能够保证原物理内存请求顺序、原寄存器分配或原性能；方案已作恰当限定。

### 4.2 普通访问、惰性分支与 predicate

GELU/Sum 的受审点可在包含访问的原语句之前插入纯索引日志，保留原 BufferLoad/BufferStore 对象结构和 vector width。对 Sum 可以先记录纯整数候选 index 和 `i_1*128+tx<257`，再执行原来的 `T.if_then_else`，不复制其中的 load。不要把 RHS 整体求值来取得 index，也不要把多条标量 load 当成原 vector。

建议把遍历限定到实际表达式树中的访问 occurrence，记录 enclosing statement、`if_then_else` 分支极性、Bind 作用域和外层控制流；不能直接复用当前 `walk_statements()`：它跳过 Bind，并把 BufferStore 当整体事件，不提供所需的表达式内分支上下文。区分用于 address/barrier identity 的 BufferLoad 形态与普通值读取，拒绝未受审 call 上下文。

固定 CUDA codegen `:4927` 和 `:5024` 对残留 BufferLoad/BufferStore predicate 明确 `ICHECK(!predicate.defined())`。因此“检查 scalar/vector predicate”应在首版落实为：末端仍有这种 predicate 就明确拒绝；只有已降低为受审 guard/if_then_else 的形式才接受。无需为了测试 predicate 而增加 C++ lowering，也不应宣称原样保留末端 predicate 就能编译。

对 signed overflow 的措辞应保守：保留原 dtype/cast 顺序是必要条件，但不能承诺任意溢出的 C++ 表达式具有可靠的数学 wraparound 语义。固定域应验证不溢出；64 位测试同时检查 sign/zero extension 与大偏移，而不把 host 高精度重算当实际求值。

### 4.3 election 与日志完整性

方案采用组级计数、不假定线程 0 正确。实际 helper `src/tl_templates/cuda/intrin.h:92` 中 elect(32) 每 warp 选一个，elect(128) 只在每四个 warp 的首 warp 选一个。当前域因此可进一步核对 elected tx：GEMM producer 为 0..31；GQA producer 为 256..287；两个 GQA output consumer 分别为 0..31、128..159。不要只检查 tx 在整个 producer/consumer 组内就算合法。

日志应放在原 election 分支内部、原 TMA call 前。不要为日志再调用一次 elect，不把 election 当普通可重复求值的 mask，不对所有未 elected 线程输出伪造 masked TMA 事件。原 wait/expect/arrive 相对关系保持不变。主键纳入所有局部循环以及对未知动态域拒绝的设计合理；B1 要补的是具体域证明。

### 4.4 64 位 printf ABI 与预算

独立 `TLACC1`、单事件一次 printf、不复用 int32 数值协议的决定合理。实施前固定字段类型/数量，检查格式符与最终生成 C++ 类型的匹配；这里 int64 scalar codegen 输出 `int64_t`（`:992`），不能仅因为 TIR dtype 是 64 位就默认任何 `%ld`/`%lld` 都正确。Linux/H200 限定允许采用已验证 ABI，无需扩大到 Windows GPU。

至少测试负数、`2^31` 以上、`2^32` 以上和 unsigned 大值往返；每种记录把 format 外参数数目限制为 32 以内。CUDA printf 的 FIFO 存的是参数及内部元数据，不能以终端文本长度直接证明容量；65536 条和 64 MiB 是候选限额，按最长 schema 做容量测试并保留缺失/重复/截断失败检查。以上为 4A/后续验收可落实的事项，不单独作为设计阻塞。参见 [CUDA 13.2 printf 限制](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/05-appendices/cpp-language-support.html#formatted-output)。

### 4.5 原 capture 与第三步兼容

新命令、配置、records validator 独立，保留 baseline-first、单 compile/launch、真实输入输出快照、进程失败即失败，设计合理。`capture.run()` 当前会分配 monitor barrier 并进行源码注入，worker 会运行 fragment 专用 gate；只能提取或复用其中通用设施，不能给 access 路径直接套该流程。`evidence.verify_capture()` 也依赖数值 records/points，已明确另写 validator 是正确的。

现有 `evidence.verify_capture()` sanitizer 允许项只有 racecheck/synccheck。新增 access memcheck 验收时明确使用自己的验证入口或兼容扩展，不改变旧数值契约含义。跨 run 关联用实际输入、源码/driver/config 摘要并明确其证据限度，符合方案目标。

## 5. 复审通过条件

只需修改方案并复审两个阻塞：补足固定四例的 source→site/事件域附表及有限规则；补足当前 backend 的 host descriptor→launch→device call 证据链与 load/store 参数契约。其余建议在方案或 4A 验收细则中明确即可。

本审阅不要求现在编写产品实现、执行 GPU、证明任意 kernel、建立通用源码追踪器或新增底层 intrinsic。修订后的方案若仍把上述关键关系留给“实现时再决定”，不能据此宣称设计已通过。
