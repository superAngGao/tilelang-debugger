# 阶段 1：H200 定点采集方案

状态：v2/v3/v4 设计、v3 代码及最终阶段1限定范围验收均已通过独立审阅。后面的修订节优先于前文。本文件保留原始讨论边界及其明确修订，当前用户使用范围以 README 为准。验收见 [phase1-validation.md](phase1-validation.md) 与 [最终独立复核](reviews/phase1-acceptance-review.md)。

## 1. 目标与边界

用户通过原始源码行号、buffer 名、block 坐标、串行循环迭代序号选择观察点。工具在临时源码副本中插入 monitor，运行 H200 kernel，保存可还原的 tile 数值、观察点上下文以及原始和插桩版本的 IR/CUDA。

本轮合并原阶段 1、2 的采集工作。reference 产品接口、误差诊断、访问诊断、HTML 报告及分层交互留到后续阶段。样例的 reference 仅用于验证工具，没有自动生成任意 kernel reference 的承诺。

不新增 TileLang lowering 指令/pass，不修改安装文件。仅在独立工作进程中使用 Python 宏和编译入口薄封装。数值格式和同步通过进程内局部替换打印 helper 实现，退出编译作用域后恢复。

## 2. 验证依据

已有 `experiments/print_sync_probe.py` 在 H200、TileLang 0.1.12 上验证：默认 ThreadSync 在普通 CTA 和 T.ws(1) 都通过 racecheck；关闭 ThreadSync 时原打印及外围同步都报告 staging write/read 竞争；在 fragment→shared 后和打印结束后使用组内 barrier 的局部 monkey patch 通过相同检查。本次所有打印值正确，不能把 racecheck 结果表述为已观察到错值。

该实验不证明完整 GQA 可直接使用固定 barrier 14，也不证明任意控制流都可插入 collective。

## 3. 目录与入口

```text
src/tilelang_debugger/
  __init__.py, __main__.py, cli.py
  instrument.py   # 配置校验、AST 选点、源码变换
  semantics.py    # 支持范围内的执行域、就绪点及 barrier 检查
  monitor.py      # 局部宏与原值编码
  capture.py      # 子进程运行、编译/launch 记录、产物比较
  records.py      # 严格解析和完整性校验
  ir.py           # 从实际编译对象保存 IR/CUDA
examples/{gelu,sum,gemm,gqa}/{kernel.py,run.py,monitor.json}
tests/            # CPU 单测及 GPU 验证入口
docs/phase1-plan.md
docs/reviews/     # 独立设计和代码审阅原文
artifacts/        # 不提交的大体积运行证据
```

命令：`python -m tilelang_debugger run examples/gelu/run.py --monitor examples/gelu/monitor.json --output artifacts/gelu-001`。

每个样例的 run.py 正常准备输入、调用 build_kernel、tilelang.compile 和编译结果。不设计插件框架。阶段 1 支持一个相邻 kernel.py、一个目标 PrimFunc、一次 compile、一次 launch；额外 compile/launch 明确报错。输入为 PyTorch GPU tensors，编译调用显式设置 out_idx，使输出作为返回值可采集。复杂多文件导入与多 kernel 编排暂不支持。

CLI 启动两个独立子进程，分别执行原始/插桩的脚本副本；固定同一 seed。工作目录包含 run.py 与原始/变换后的 kernel.py，保持可 inspect 的源码文件。原文件不改。两次运行的实际输入值逐一核对，不能仅凭 seed 声称同输入。

## 4. 观察配置与源码定位

配置顶层指定 `source: kernel.py` 和 points。每个 point 包含唯一 id、原始 line、buffer 标识符、when: before/after、block 三维坐标，以及 enclosing serial loop 的 `{line, iteration}` 列表。iteration 从 1 开始。

line 必须唯一定位支持的 statement；多行调用按其开始行选择，不通过包含关系猜测。允许在整个 T.Parallel 语句完成后观察其输出 buffer；不允许在 T.Parallel 内部加入整个 tile 的 collective。同一源码行多个 statement、宏内部、动态生成源、异步分支无法解释时明确拒绝。

串行范围支持 `range`/`T.serial`/`T.Serial`，常见常量或编译参数起点、终点、步长；将迭代序号转换为原循环变量条件，保存原始变量值。嵌套串行循环必须逐层选择。T.unroll/T.Pipelined 只在验证过的简单形式开放，不能仅因名称相似就默认支持。

插桩使用 AST 生成临时源代码并保留源码摘要、point→原始位置映射。lowering 后不反推原始行号。变量命名避免与用户局部变量冲突。

## 5. 执行域与同步

参考 tilesight-for-tilelang 的 actor partition 与 synchronization 数据结构，但不依赖性能建模包。阶段 1 只证明有限的线程域形式：整个一维 CTA；`T.ws(整数常量)`；以 threadIdx.x 为变量的固定连续 warp-aligned 区间分支。线程域内 monitor 条件只能依赖 block 坐标和已证明组内一致的串行循环变量，不能依赖各线程不同的 runtime 值。

源码 AST 负责定位，未插桩的前端/最终 device IR 负责核实线程数、buffer 存储范围及同步资源。`T.ws`/线程区间必须与实际 lower 后的参与域一致。不能由 producer/consumer 名称或 kernel 名猜测。

fragment 采集宏：原结果就绪 → group 内 T.copy 到私有临时 shared → group barrier → leader 逐元素输出 → group barrier → 后续计算。后一个 barrier 保证 scratch 读取完成后再复用。整个 CTA 与子分组都使用显式参与人数，避免在 consumer 内加入全 CTA barrier。

barrier ID 从基线编译的实际 device IR/CUDA 已用 named barrier 集合中选取，取剩余 ID；自动同步保留的资源也计入。动态 ID、无法辨认的同步内联代码或无可用 ID 时拒绝自动插桩。多个可能并发的 monitor 分配不同 ID。补丁编译后再次核查新旧 barrier，不能把碰撞当作成功。

异步就绪不由打印 barrier 替代。本轮四例的异步观察点必须紧随明确的 wait(0) 与必要 operand fence，或位于后续已同步的计算后。GEMM 与 GQA 对不同 outstanding group 的 wait(1) 不作为任意 buffer 就绪保证。无法证明安全点时拒绝，报告具体缺少的证明；不隐式加入可能改变原始异步协议的 wait。

初始支持 whole fragment 和 shared buffer；普通 scalar 可选后补。shared 直接读取同样要求生产完成且尚未发布可复用状态。未验证的全局跨 CTA 快照不支持。

## 6. 数据记录与保真

用已有 call_extern/printf 路径输出每元素一整行结构化记录。一个记录含 point ID、launch ID、block、被选循环的原变量值、扁平元素索引和原始位模式；不能把 index 和 value 拆成两次输出再按顺序拼接。

第一版 dtype：float16、bfloat16、float32、int32。位重解释输出 uint16/uint32 的十进制值，主机用 dtype 还原；不通过 `%f` 做精度依据。NaN/Inf/负零保留位模式。维度、dtype、选择器和预期元素数量在构建 instrumented IR 时从实际 buffer 对象记录到 points.json。

一个 worker 只允许一次目标 launch，launch_id=0；run_id 与 worker 版本由输出目录及主机元数据关联。此范围内无需给 kernel 增加 launch 参数。每个点选择一个 block 和各 enclosing loop 的一次迭代，因此预期恰好一个完整 tile。

每个 point 应恰有 shape 乘积条记录，index 覆盖完整，无重复。未到达观察点、缺失、重复、错误 point、意外 block/迭代、非法 dtype/位宽、截断或进程错误均令本次采集失败，保留原始日志，不呈现“完整成功”。记录排列不依赖 stdout 顺序。

## 7. 编译、运行与产物

目标版本先固定已验证的 TileLang 0.1.12 环境，记录实际安装路径、版本、关键 helper 源码摘要、CUDA 和 GPU 信息。新版本必须重新验证 helper 结构，不能静默套用兼容补丁。

包装实际 tilelang.compile 调用，保存其输入 PrimFunc 和结果 artifact.device_mod、get_kernel_source；不另编一个仅用于展示的“相似 IR”。保持用户 pass_configs、compile_flags、target、out_idx；仅添加经过声明的采集相关配置。所有 monkey patch 在 try/finally 中恢复，独立 worker 隔离全局状态及缓存。

产物包括 run.json、配置快照、points.json、原始/插桩源码、各自 frontend/device IR 和 CUDA、stdout/stderr、records.jsonl、input/output tensor 快照及验证结果。新输出目录必须不存在，拒绝覆盖。仅保存目标程序及本次必要数据，不复制其他工作区内容。

launch 后显式 CUDA synchronize 再结束 worker，以捕获完整设备 stdout。设置明确 printf 容量/采集预算，超预算拒绝或要求缩小选择范围；不声称不会丢记录。子进程与进程组有超时终止，GPU 异常立即报告，不继续吞掉错误。

## 8. 四个样例及验收

从 tilesight-delivery-docs-20260923/examples/kernels 复制独立 kernel，保留 LICENSE.TileOps 和来源说明。保持算法、线程分工和已有 barrier 协议；缩小问题尺寸但至少保留两个 block 及足够串行迭代验证过滤。

- GELU：观察 x_reg 或 y_reg；验证 fragment 数据与输入或最终输出对应。
- Sum：观察转换后/归约后 fragment；验证值与原输入或结果对应，包含正常 padding 分支。
- GEMM：观察 consumer 中 wait(0)/fence 后 c_local，选择第二轮；测试端用前两轮 K 范围计算参考；核对完整 kernel 输出。
- GQA：先观察一个 consumer 的 QK prologue 完成点，核对相应 Q/K tile 的 matmul；完整输出与独立 attention reference 比较。随后至少验证另一个 consumer 的选择和打印线程，不把最小 ws 实验代替完整 GQA。

CPU 测试：源码选点、嵌套循环、命名/行号、拒绝线程分歧或未知域、barrier 冲突、bits 还原、重复/缺失/乱序/错误身份及配置预算。

H200 测试：四例 baseline/instrumented 输出一致性和独立 reference、每点元素数量与值、实际 CUDA barrier 位置；两个 consumer 覆盖。对新增 staging 路径运行 racecheck；若原程序有 racecheck 报告，应保留 baseline 对照并定位，不能忽略新增报告。适用时用 synccheck 核实 collective。原有打印最小实验作为回归。

完整验收以前不能将四例标成已支持，也不能只通过 CPU 单测宣称 H200 采集完成。

## 9. 审阅与实施顺序

1. 独立 session 只读审阅本方案及已有实验，输出 PASS/FAIL 与可执行意见。
2. 修订直至没有阻塞问题，保留每轮审阅原文。
3. 实施 CPU 定位与记录、GPU monitor/运行、四例验证。
4. 独立代码审阅，修复并重跑受影响验证，直至通过。
5. 更新 README 的实际支持状态，提交代码、方案、审阅与精简验证结论；大日志留 artifacts。

## 10. v2 实施约束（优先于前文较宽的描述）

首版以四个完整样例的受审源码契约为接受范围。契约包含完整源文件 SHA256、限定构建参数、精确 AST 锚点及以下证明；修改源码后须重新审阅契约，不能仅凭 kernel 名放行。并不声称支持任意 TileLang 程序。通用 AST 定位/循环选择保留，但未知函数、别名、写入、分支或生命周期不因能解析 AST 就获得运行资格。初版仅 fragment；shared 直接读取、T.Pipelined/T.unroll 内采集暂不开放。局部编译参数、buffer shape/scope/dtype、实际线程域仍须在构建及实际 IR 中检查。

四例具体锚点如下（生成配置时从原文件 AST 获取行号，不手填漂移的行号）：

| 样例 | 锚点和 buffer | 选择与参与域 | 初始化/就绪证明 |
| --- | --- | --- | --- |
| GELU | after `T.copy(..., x_reg)` 观察 x_reg；after 完整 `T.Parallel` 观察 y_reg | block x=1，CTA 128，leader 0，无 enclosing serial loop | 整个输入 tile copy 完成；整个输出 Parallel 覆盖全部元素；N 是 tile 大小整数倍 |
| Sum | before `T.reduce_sum(x_f32, acc, dim=1)` 观察 x_f32；after reduce 观察 acc | block x=1，CTA 128，leader 0，无 enclosing loop | 选点在整个 pad/non-pad 分支之后，两条分支均执行完全部 block_m 行；M 是 block_m 整数倍；分别测 N=256 和 N=257 |
| GEMM | after consumer 的整个 `for s in range(num_stages)`，观察 c_local | block (1,0,0)，外层 ki 的第 2 次；consumer tx=[128,256)，leader128；s 已退出，不是 enclosing loop | gi_cons 初始0且每轮统一+1，stage=gi_cons%num_stages，s 完整覆盖且恰一分支；分支里的同一 c_local wgmma 后 wait(0) 和 fence；第二轮之前 first clear_accum 成立 |
| GQA | after 每个 consumer prologue 的首个 `T.wait_wgmma(0)`，观察该词法作用域 acc_s | block (1,0,0)，无 enclosing serial loop；分别 ws0=[0,128)、ws1=[128,256) | 同一作用域 wgmma clear_accum=True 写满 acc_s；q_bar/kready 已等待；在释放 kfree 前。monitor 先发出已有 `warpgroup_fence_operand(acc_s,num_regs=64)`，不新增 wait；8192 个 fp32 元素/128线程=64，并核查实际 CUDA accumulator 每线程分配覆盖范围及 fence 在 wait 之后 |

GEMM 验收固定 tile=128×128、Ktile=64、stages=3、M=128,N=256,K=256；GQA 固定 B=1,H=2,Hkv=1,D=64,Q=256,KV=384、noncausal、softcap=0，原有 producer/consumer/同步和循环保持。GELU 固定 threads128,npt16,N4096；Sum 固定 M4,block_m2,threads128。更改这些布局参数须新契约；seed 可以变。观察的 block/合法迭代可选择，但不能选择未知安全语句或尚未完成的部分 tile。

运行严格顺序：baseline worker → 导出实际 IR/资源 manifest、一次 launch → instrumented worker 读取 manifest → 构建与编译 → 检查产物及资源 → 一次 launch。每个 worker 在 import 前设置 `TILELANG_DISABLE_CACHE=1`；若实际返回对象没有 `artifact.device_mod`，在 launch 前失败。逐项保存并比较实际 compile 的 target、execution_backend、pass_configs、compile_flags、out_idx；这轮不允许改变影响同步的配置。编译调用计数及 launch 计数必须恰为1。

barrier 核查：每个 point 独占一个基线未用的 named ID（0保留），从15向1分配。解析实际 device IR 的调用节点，同时保存 CUDA 证据；动态 named ID、未知同步 asm、无法识别的调用拒绝。插桩编译后，在 launch 前逐点核对恰有两处该 ID/count，且位于对应线程域/观察条件及 staging store→printf→后续操作的位置；原有同步调用/参与域保留。新增自动同步不得使用 monitor ID，原有ID的用法发生不可解释变化也拒绝。有限四例契约允许对实际结构作专门核查，不建设通用并发证明器。

记录格式 v1：固定前缀 `TLDBG1|`，point id 限 ASCII 字母数字下划线、长度≤32；一次 printf 输出 point、launch=0、block xyz、最多2层 loop 原变量值、扁平index、bits，共≤9个数值参数。标识整型显式转 int32（%d），bits 在原 dtype 的 shared 元素上先 reinterpret uint16/uint32，再 cast uint32（%u），不先转 float。points.json 在前端宏接受实际 buffer 对象时绑定唯一 point id，记录原名、scope、dtype、shape、选择条件、域和barrier；同 point 在多个不一致作用域构建直接失败。布局/shape不从日志猜测。

每次采集预算≤65536元素，按每条最多1024字节设备 printf FIFO 开销计，worker 显式请求至少64MiB FIFO且检查CUDA调用返回状态；超过预算拒绝。仍以完整性校验检测截断，不能用预算代替结果核验。run.py 与 kernel.py 是可信用户代码，工具不是沙箱。

端到端保真验收另有一个最小只做 input→fragment→monitor→output 的 kernel，固定 threads128，两block。对float16/bfloat16/float32/int32各测原始bits；浮点包括正负零、正负Inf、多个quiet NaN payload、有限极值；整数含0、-1、INT_MIN/MAX。输入和输出快照均存原始字节以及dtype/shape/stride，输入在 launch 前保存；NaN不以数值相等比较。四例 baseline/instrumented 输出要求逐位相同；如果不相同，视为阻塞并定位，不能自动放宽。独立数值 reference 使用预定容差：GELU bf16 atol=0.03125,rtol=0.02；Sum fp16 atol=0.03125,rtol=0.002；GEMM fp16输出 atol=0.0625,rtol=0.002，fp32中间tile atol=0.001,rtol=0.0001；GQA fp16输出 atol=0.01,rtol=0.01，QK fp32同GEMM中间tile。输入均为固定seed的标准正态数，reference使用fp32，禁用TF32；GELU采集输入/输出、Sum采集x_f32逐位核对。每项实际最大误差随验收保存。

## 11. v3 lowering 澄清（已通过审阅）

实际 Sum unpadded 编译表明：插桩可以改变 fragment 布局，使原有跨线程 shared 读写变为同线程读写，原来的自动 CTA barrier 随之消失。原方案门禁已在 launch 前拒绝该变化。拟只对无手工同步、无异步操作的受审 GELU/Sum CTA 契约，允许 ThreadSync 在保持启用的条件下改变自动 CTA barrier，记录完整差异；原有 named barrier 和所有 WS 契约的同步调用及参与域仍严格保留。编译器自动同步的变化须由该固定契约的 baseline/instrumented racecheck 和逐位输出验收证明；不扩展到未知源文件。

归约结果存在复制布局，TileLang 的 fragment→shared copy 会只选择代表线程存储。因此 staging writers 允许是 barrier 参与域的非空子集；实际 lowering 的索引、循环与线程坐标须枚举证明整个 tile 每元素恰有一个 writer。printf 从同一实际 shared allocation 读取，两个 group barrier 包围读取；GQA fence 的指针必须与该 staging 实际读取的 accumulator 指针一致，不能依靠同名变量配对。

## 12. v4 样例修正与 operand fence（已通过审阅）

H200 上原 GQA 在无 monitor 的 racecheck 运行出现数值错误。隔离实验 `experiments/gqa_epilogue_probe.py` 中，原版连续3次 reference 失败；对两个 consumer 在 `T.copy(acc_o, Os[group,:,:])` 后增加各 writer 的 `T.fence_proxy_async()`，再各用独立 `T.sync_threads(3/4,128)`，随后保持原来的 shared→global TMA copy，连续3次通过，max_abs_error=0.00017371773719787598。原与修正版 racecheck 均报告0 hazards，不能以此说 sanitizer 捕获了该缺陷。

原 CUDA 在所有线程写 Os 后，直接由 leader 发起 TMA store，没有组内同步；已有 proxy fence 也仅位于 leader 分支。新增顺序符合 [NVIDIA TMA shared→global 写出协议](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)。将原版保留为 `examples/gqa/kernel.original.py`；正式 example 仅作这四行同步修正，并在基线和插桩两边共同使用修正版。协议原本的named1/2与mbarrier、算法、线程划分和等待不变。新的3/4计入monitor资源扫描。原版不宣称通过，不静默覆盖失败证据；实验可从保留原件重复生成对照。

另，GEMM 的 c_local 实际每线程128个fp32寄存器，源样例已有wait后fence64不能当作全buffer compiler ordering的证明。monitor在受审ring dispatch之后增加现有operand-fence intrinsic，覆盖实际128个寄存器；检查实际被staging读取的同一accumulator指针、offset0及完整allocation大小。GQA相应为64。没有新增GPU完成wait或lowering指令。原始异步/同步协议的实际IR调用与loop/guard上下文须保留，逐点只剔除经过验证的monitor barrier及其新增operand fence再比较。

## 13. 代码审阅后的实现细化

lowering 会克隆 buffer 的 Var，因此不能假定前端和最终IR之间对象身份不变。插桩副本将选中buffer及其同一TileLang词法作用域内的全部引用改为唯一内部名；前端宏记录实际buffer对象。最终IR必须恰有一个对应内部名的allocation，随后staging source、operand fence和该allocation用对象身份核对。GQA两组同名acc_s分配不同内部名；对用户展示仍是原始名称与原始行号。

staging检查目标地址全覆盖、每个writer的源local索引完整且不越界，拒绝额外数值算术；printf循环、读取地址与记录index必须一一对应。原值编码必须是同宽reinterpret后零扩展，拒绝转换后再编码。未知调用用固定受审intrinsic/extern集合拒绝；AllReduce模板helper的同步ID1/2始终保留，相关header固定摘要，CUDA显式named barrier与IR交叉核对。

实际 GEMM 的既有 InjectFenceProxy pass 因新增generic shared写入，在循环中额外插入一处 `fence_proxy_async`。允许这一明确的附加内存顺序fence，要求参与线程域可证明、记录差异，并保留所有原协议调用及其guard/loop顺序；不允许以它代替原来的完成wait，也不允许新增未知named同步。此变化由代码复审和H200复验共同核实。

monitor采用局部T.macro及已有call_extern printf/同步intrinsic，直接复用T.print的fragment→shared实现路径，不再替换公共T.print helper；唯一进程级替换是临时compile包装，finally恢复。避免为不同格式在用户打印上引入额外全局行为。

第二轮代码复审指出：仅分别证明source/destination全覆盖仍允许完整置换。现将 `(逻辑index, writer线程, 源local index)` 的完整配对摘要纳入每个观察点布局契约，未知摘要拒绝；诊断提取时不launch插桩程序，且要求生成CUDA逐字等于已通过独立数值验收的产物，候选摘要经review后才进入契约。布局改变不自动学习或自动放行。另要求两个monitor barrier的完整loop上下文严格等于受审enclosing serial loops，printf仅多一个全元素循环；线程相关或额外collective循环一律拒绝。
