# 第四步方案与修订记录：运行时访问索引、边界与布局观察

> 历史记录：旧 `trace` 已从当前产品移除；文中 trace 命令、专用测试和兼容性描述仅适用于[移除前提交](https://github.com/superAngGao/tilelang-debugger/tree/e838fc88d96a562ee6230ee154d6593bac6632f8)。保留当时结果，不代表当前产品能力。

状态：设计v3和代码v2独立审阅均PASS，已实施并完成H200验收。第13/14节优先于第12节及前文，保留早期草案用于解释修订；最终结果见phase4-validation.md。沿用原阶段编号，第一、二步合并完成，第三步reference与数值分析已完成。

## 1. 目标

用户在原始源码选择一次copy或访存所在语句、block和原始循环迭代，查看该次操作在运行时使用的索引、偏移、访问条件及目标区域。将这些信息关联到源码、实际device IR和生成CUDA，用于解释数值错误可能来自哪次数据访问。

核心路线：**编译期找到原访问；运行时采集该访问自己的索引和条件。** 不另写一套正确索引打印出来冒充原程序行为，也不为记录索引额外读取目标数据。

交付继续聚焦Linux/H200、已核查TileLang源码及四个固定样例。首版不做硬件内存事务追踪、性能统计、自动根因归因、完整并发竞态检测或任意kernel支持。

## 2. 两种实际访问分别记录

### 普通load/store

从选中操作lowering后的BufferLoad/BufferStore提取实际buffer、索引和条件。在同一运行上下文中记录block、原循环值、线程、向量元素位置、元素/字节偏移以及访问条件。

向量访问保留原load/store的向量宽度；仅将索引表达式的各向量分量输出，不把原访存改写成多条标量访存。日志中的lane字段指向量分量，另存thread/warp lane，避免混淆。

条件为真表示该IR访问路径被选中；记录在访问之前发出，不作为内存请求完成、缓存命中或物理总线事务的证据。编译器和设备仍可能调度指令，报告只描述插桩程序的IR访问及实参。

### TMA调用

GEMM/GQA的主要输入搬运是TMA，记录实际调用点的descriptor身份、运行时坐标、shared目标偏移、barrier身份、执行线程/组和原始循环值。tensor shape、stride、boxDim、swizzle及越界填充设置从实际descriptor构造信息建立关联，区分静态常量与运行时参数。

TMA是一条搬运调用，不输出虚构的“每个线程加载了哪个global元素”。可根据本次实参和实际descriptor离线展开逻辑区域，但必须标注为 `derived_from_runtime_operands`，而不是逐元素runtime trace。shared swizzle无法可靠解码时保留原始参数并注明未解释，不能套用普通row-major偏移。

NVIDIA文档说明TMA的descriptor包含shape/stride/box/swizzle/OOB配置，因此“坐标超出逻辑shape”不能单独等同于非法访问，必须结合操作方向和填充策略解释。见 [CUDA Driver API tensor map](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TENSOR__MEMORY.html) 和 [CUDA 13.2.1异步搬运说明](https://docs.nvidia.com/cuda/archive/13.2.1/cuda-programming-guide/04-special-topics/async-copies.html)。

## 3. 第一版的用户入口

拟新增独立 `trace` 命令，复用现有baseline/instrumented worker、快照和产物基础设施。与数值capture使用分开的配置和记录schema，避免将访存事件误当成完整数值tile：

```bash
python -m tilelang_debugger trace examples/sum/run.py \
  --access examples/sum/access.json --output artifacts/sum-access-001
```

配置草案（字段在实施前由review确定）：

```json
{
  "source": "kernel.py",
  "accesses": [{
    "id": "sum_input",
    "line": 35,
    "operation": "read",
    "buffer": "x",
    "block": [1, 0, 0],
    "loops": [{"line": 33, "iteration": 1}]
  }]
}
```

line采用原始语句开始行；例子选择Sum归约前加载语句的第一行数据。串行循环仍使用1起的iteration，输出另存原变量值。T.Parallel所映射的线程/向量元素由工具枚举，不要求用户为每个线程填配置。

对于TMA，operation为transfer、buffer标识源buffer；同一源码语句产生多个静态访问点时，列出映射并全部记录，不能随意取第一个。存在多个不能消歧的源级操作时要求选更具体的已支持位置。

首版每观察点仍选一个block和受支持的串行迭代，采集该操作的全部已知参与线程/向量分量。不增加任意线程筛选、复杂条件表达式或追踪整个kernel的默认模式。数值capture和access trace先分别运行，不默认叠加两种插桩。

## 4. 编译期与运行时各做什么

| 信息 | 编译期确定 | 运行时采集 |
| --- | --- | --- |
| 源码位置、buffer身份 | 原源码摘要、AST锚点、词法作用域 | 仅输出稳定site ID |
| 访存种类、dtype、向量宽度、地址空间 | 实际device IR | 静态项只在manifest保存 |
| block、原循环值、线程 | 确定表达式及对应关系 | 记录本次值 |
| 索引、字节偏移 | 从原访问节点提取表达式 | 对本次实参求值并记录 |
| 条件/mask | 提取具有正确分支语义的访问条件 | 记录true/false |
| TMA区域 | 关联实际descriptor及调用点 | 记录坐标、目标偏移等实参 |
| 元素范围及布局展示 | 建立已验证的转换规则 | 主机解释runtime实参；明确标注推导项 |

第一版只处理四例中的纯整数索引和条件：常量、线程/block、串行变量及已存在的标量绑定。若索引含数据依赖BufferLoad、带副作用的call或无法保证安全求值的表达式，先拒绝该位置。后续支持gather等动态数据索引时，应复用原计算已经得到的值，不能重复加载索引tensor。

保留原始计算位宽和signedness；先取得原表达式结果，再扩展为64位记录。不能先把整个表达式换成64位运算，以免掩盖原有32位溢出。原始求值结果与主机高精度的边界推导若同时展示，必须分开命名。

## 5. 推荐插桩位置与TileLang接入

已检查当前固定checkout `2d63708c8ad57196051c4636a1167c5453c73a48`：

- `tilelang/cuda/pipeline.py` 先LayoutInference/LowerTileOp，再LegalizeSafeMemoryAccess/LowerAccessPtr，之后FlattenBuffer、向量化、StorageRewrite、host/device拆分及ThreadSync等。
- `tilelang/backend/pass_pipeline/pipeline.py` 提供PassPipeline、get_pipeline/register_pipeline；`engine/lower.py` 调用pipeline.lower后分离device模块并进入codegen。
- `src/op/copy.cc` 的LowerNormalCopy及 `src/cuda/op/copy.cc` 的CUDA copy分发说明：一个源级copy并不保证最终一定是普通BufferLoad/Store。

优先方案是在独立worker中临时包装已注册CUDA pipeline：调用原pipeline完整lowering一次，对返回模块里的目标device函数追加access日志IR，再交还原编译流程。退出作用域恢复原pipeline；原pipeline执行异常也必须恢复。只处理目标一次compile，不更改安装文件，也不复制整套pipeline到本项目。

这样观察的是经过布局、flatten、向量化和同步处理的真实访问；插桩后不重新跑LayoutInference/ThreadSync。codegen前仍有LowerIntrin等步骤，必须保存该边界及最终生成CUDA，验证日志节点正常降到既有printf。

这里需要新增**Python侧IR插桩变换**，但不新增GPU指令、TileLang intrinsic或C++ lowering规则。能否在该位置正确处理scalar/vector访问与现有artifact导出，是第一个实施实验的必验项；不能仅凭接口存在就宣称已可用。

保存三个阶段：baseline device IR、插桩编译内pre-trace device IR、插桩后实际送codegen的device IR/CUDA。移除新增日志节点及其仅用于日志的绑定后，必须与同次pre-trace IR结构等价；原buffer读写、控制流、同步、分配和调用实参不变。同时对照baseline的原访问和协议，无法解释的编译差异仍拒绝。

## 6. 源码到lowered访问的关联

沿用源码行选择和源文件/驱动摘要；增加受审 `source point → lowered sites` 关联。首版针对四例的固定结构做局部匹配，依据buffer实际对象/分配身份、访问方向、索引结构、循环及分支上下文、TMA descriptor来源共同判断，不能只靠buffer同名或日志顺序。

一个源语句可能对应多个lowered site，每个site分配稳定ID；对于直接搬运，只有源/目标关联可证明时才给出成对映射。普通read/store分别记录自己的索引，不因在同一语句或时间相邻就配成一次copy。

先对已有真实IR建立有限映射，再在编译时核对。不能将“主机推导出了合理坐标”视为成功关联；映射失败应给出无法支持的具体节点。本轮不承诺通用的跨任意pass源码追踪器，不依赖span总会保留。

## 7. Mask与边界语义

Sum padding是首个重点：其条件位于 `T.if_then_else` 表达式内部，不能只收集外围IfThenElse语句。必须保留条件分支求值语义；同样检查BufferLoad/Store predicate、向量predicate和包含它们的外层分支。

对可安全计算的候选索引，在原mask分支之前、仍位于原外层执行上下文内发出记录，区分：

- `active_in_bounds`：条件成立，索引在已知合法范围。
- `masked_out`：条件不成立，没有该次读取；报告候选索引但不称实际访问。padding值来自常量时注明fill，不伪造load。
- `active_out_of_bounds`：条件成立但越界，可疑；不额外读取地址，也不自动修改原程序来“保护”它。
- `unknown`：只有偏移可见但逻辑维度、descriptor或边界不能确定，禁止强行判定。

TMA使用单独的区域边界解释，合法OOB填充与非法请求分开。不能用普通load的判定规则覆盖TMA。

不能为了记录mask=false而把原来受保护的危险索引计算（例如除零或间接读取）提到分支外；无法安全取得的候选索引标为不可用或拒绝该点。被编译器完全消除的访问没有runtime事件，只能在静态说明中解释。

不绕过现有baseline先运行的检查。若原程序在baseline就非法访问/退出，第一版不会自动继续插桩重跑；记录已知失败，不宣称能可靠获取faulting kernel的完整轨迹。专门的故障采集模式留待后续单独设计。

## 8. 日志与完整性

复用既有外部printf调用，但使用独立协议，例如 `TLACC1`。由执行访问的线程直接打印索引/条件；无fragment搬到shared，无新增collective/barrier。实际输出会影响寄存器、时序和性能，不用于原程序性能推断。

每条完整记录一次printf；建议核心字段：run/launch关联、source point、static site、block、原循环值、线程、向量分量/宽度、访问条件、原位宽、偏移或TMA坐标。静态buffer/dtype/shape/表达式/地址空间保存在 `access-points.json`，不重复打印长字符串。64位值使用匹配的printf ABI，不截断成现有数值协议的int32。

记录主键包含static site、block、所有影响事件身份的动态循环值（包括lowering产生的局部循环）、线程及向量分量。不能仅靠用户选中的外层循环识别重复。TMA election只记录实际发起线程；对于已证明“一组恰有一个发起者”的事件，以组级身份核对次数，不假定elect固定选择线程0。

首版只接受可在固定四例域内验证事件覆盖的选择。编译期为所选block/迭代构造预期事件域，运行后检查缺失、重复、非法身份、截断及完成状态。mask=false也有候选事件时单独计数。静态无法确定覆盖条件的动态控制流暂不支持，不能把未知数量当作日志完整。

按实际字段/事件数重算printf容量预算；先沿用不超过65,536条记录、至少64MiB FIFO作为上界约束，但必须验证新协议单条开销和容量是否足够，不能机械继承数值采集证明。不支持截断后标记成功。

## 9. 文件产物与第三步衔接

新增 `access-points.json`（源码/site/表达式/descriptor映射）、`access-records.jsonl`（原始运行事件）、`access-analysis.json`（覆盖和边界统计）、`access-report.md`（可读索引/范围/条件摘要）。两版源码、IR、CUDA、快照、日志和进程状态继续保留。

报告将“运行时记录”“根据运行时实参推导”“未知”显式分开。可按源码点、buffer、block、迭代及错误坐标筛选，不默认展示数万行。

第三步数值报告与第四步访问报告可按源码版本、实际输入摘要和观察上下文交叉查看。两次运行的run_id不同，必须验证输入及配置一致，并注明是跨运行关联；不能声称某次访问直接导致另一run的某个错值。本轮不自动建立完整数据依赖图。

## 10. 实施顺序与四例验收

| 子阶段 | 目标 | 主要验收 |
| --- | --- | --- |
| 4A 插桩入口实验 | 包装原pipeline；保留原访存和同步；正确编译printf日志 | 单点scalar/vector索引原样输出；移除trace后IR等价；异常时恢复pipeline |
| 4B GELU与Sum | 普通访问、copy、向量分量、mask及补零 | GELU输入/输出访问范围；Sum N257的有效列与255列padding；N256的global→shared→fragment路径 |
| 4C GEMM | producer TMA调用与ring slot、迭代索引 | A/B第2轮K范围、shared slot、实际descriptor坐标/stride；原有wait/arrive不变 |
| 4D GQA | Q/K/V搬运与两个consumer输出store | Q两块区域、KV head映射、K/V循环及各自slot、两个输出区域；不得混淆TMA调用与逐线程元素读取 |

先实现4A、4B再扩展TMA，最终第四步完成要求4C、4D也通过。每扩一类真实访存形式，明确新的受审访问契约；不自动扩大原数值capture的支持范围。

测试必须覆盖：

1. CPU/TIR：源点消歧、索引/条件表达式一致性、if_then_else惰性分支、vector分量、符号和64位偏移、错误映射、重复/缺失/截断、TMA派生标识和未知descriptor拒绝。
2. H200：四例两版输入输出逐位一致、第三步reference继续匹配、记录覆盖符合预期；racecheck/synccheck检查新增行为，memcheck检查普通访问及测试用例的边界。工具零报告不替代索引语义验证。
3. 负例：在专用受审测试fixture内制造仍在分配范围内的错行/错列，runtime记录必须反映错误索引而不是期望索引；mask=false越界候选不得触发额外读取；错误descriptor关联、日志截断及未知IR必须拒绝。真实非法访问仅在明确隔离的sanitizer测试中验证失败状态，不作为完整采集正例。
4. 逐次查看最终CUDA，确认无额外目标数据读取，无新增barrier/wait，原访存/同步参数未变；检查本轮是否因日志造成codegen形式变化，任何无法解释差异留证并拒绝，不能重试后抹掉失败。
5. 原阶段1和第三步回归继续通过。方案经独立session审阅通过后才实施，实施再经独立代码复审、修复及真实H200验收。

## 11. 建议的代码落点

新增 `access.py`（选点和受审site关联）、`access_ir.py`（作用域内pipeline包装、IR变换/前后核对）、`access_records.py`（解析、覆盖与边界报告）；复用capture/evidence的进程和快照能力，但access事件采用独立验证器，不能调用数值records.parse或现有fragment专用IR检查来假装已经验证。

四例增加access.json；tests增加CPU/TIR单测、H200访问验收和独立负例。独立维护access契约，现有数值capture契约及第三步provider接口不改变。

本草案最先需要review确认的是：最终lowering边界的包装方法、源点到lowered site的有限关联策略、mask=false的安全记录方式、TMA descriptor的真实来源及新记录协议完整性。若4A实验推翻某项假设，应修订方案并复审，再扩展到完整样例。

## 12. v2：固定映射、事件域及TMA证据链（优先于前文）

### 12.1 固定源码到访问的契约

复用既有四例源文件/驱动摘要及固定构建参数，但独立维护access契约。契约同时固定已经验收的baseline CUDA摘要；若baseline生成不同CUDA，拒绝本次trace，不自动学习。当前真实baseline产物作为映射建立的依据，不用数值monitor版的CUDA。pipeline内pre-trace模块须与本次baseline保存的同边界模块结构等价；两个条件共同保证只对受审原程序结构做局部插桩。

静态site通过源文件摘要、AST行/作用域、操作角色共同编号，用户ID仅为显示标签，不使用Var指针值或全局遍历次序做身份。匹配同时检查：实际参数/分配对象、alias链、所属语句角色、分支和loop结构、向量宽度、索引骨架及以下唯一site计数。baseline CUDA摘要和pre-trace等价性使有限匹配有依据；命中0个或多于受审数量即拒绝。

| 源码点/方向 | 末端site及对象约束 | loop/线程域与预期数量 |
| --- | --- | --- |
| GELU :87 read x；:90 write y | 唯一global参数x/y对应输入/输出；宽8；local x_reg/y_reg的copy语句；index=bx*2048+i*1024+tx*8+lane | lowering i=0..1，tx=0..127，lane=0..7；每点2048记录 |
| Sum N257 :35 read x，选原:33的i | local x_f32 store的if_then_else真分支唯一global x load；index=bx*514+i*257+i_1*128+tx；condition=i_1*128+tx<257 | 选i=iteration-1；i_1=0..3，tx=0..127；512候选=257active+255masked |
| Sum N256 :42 read x / write shared_buf | 宽4直接copy；global index=bx*512+tx*4+lane；shared index=tx*4+lane；shared_buf由shared.dyn基址+0别名而来，size512 FP16 | tx=0..127，lane=0..3；两端分别各512记录 |
| Sum N256 :47 read shared_buf，选原:45的i | shared_buf→local_cast宽2；index=i*256+tx*2+lane；后续cast到x_f32不额外当作shared读取 | i=0..1中选一次；tx=0..127，lane=0..1；256记录 |
| GEMM :93 transfer a；:98 transfer b | a_desc/b_desc的唯一tma_load；shared a_smem/b_smem；shared element offset=s*8192、extent8192，FP16；coords A=[ki*64,0]，B=[ki*64,bx*128] | 原:70 ki、:77 s全部选择；默认ki=1,s=1；在stage==s的原elect(128)内各1事件，tx允许0..31 |
| GQA :114/:115 transfer Q | 同Q_desc的两个静态load，shared Qs offsets0/4096、extent4096；coords=[0,bx*128(+64),by,0] | 无原loop；producer原elect(32)内，各1事件，tx=256..287；不按顺序合并两个site |
| GQA :120 transfer K；:128 transfer V | K_desc/V_desc唯一load；shared Ks/Vs，offset=(k%2)*8192、extent8192；coords=[0,k*128,0,0] | 原:117 k选一次，合法0..2；各1事件，tx=256..287 |
| GQA :293/:455 transfer Os | O_desc两处store；shared Os offsets0/4096、extent4096；coords=[0,bx*128(+64),by,0]；barrier=null | 两个consumer原elect(128)内，各1事件，tx分别0..31/128..159 |

GQA shared.dyn别名链固定：Ks/Vs/Qs/Os分别从同一98304字节allocation的0/32768/65536/81920字节偏移而来。GEMM shared分配/alias以实际metadata核对；同名但不同对象不能匹配。参数角色以原PrimFunc buffer_map和host launch绑定核对，不依赖设备参数恰好保留原始顺序。

表中公式用于审阅/验证映射，打印参数来自被选节点原表达式；不以公式替换实际表达式。错误索引负例采用专用fixture及独立受审的错误骨架/摘要，验证错误offset原样出现，不为产品添加绕过契约开关。

### 12.2 标量、条件与覆盖规则

原loop行号从AST确认，并在受审作用域内映射到唯一末端loop Var。主键包括该点所有末端loop值（最多4层）和vector lane；未使用位置填0。普通访存逐tx核对覆盖；TMA覆盖按组级事件计数且检查实际elected tx属于上表首warp，不能按固定thread0配对。

GEMM gi_prod是local.var[0]：必须验证初始化0、producer外层ki每轮末尾恰一次+1、无其他写者/未知call修改，stage=gi_prod%3。只在原stage分支和原elect分支内部打印ki、s和call参数，不重新读取gi_prod、不重新调用elect。用户选择不可能的ki/s组合在编译/配置阶段拒绝。

GQA eff local.var[0]只允许支配循环的一次赋值ceildiv(seq_len_kv,128)，无其他写；seq_len_kv/seq_len_q由实际K/V、Q/O tensor shape绑定并交叉核对为384/256。对其loop覆盖用实际launch绑定求值，不在GPU额外读取eff来输出日志。拒绝其他local.var递推和tensor数据索引。包含这些已验证标量的外围guard不重复求值；普通masked候选只支持Sum纯整数if_then_else条件。

末端BufferLoad/Store仍带predicate则拒绝（当前CUDA codegen不接受）；只支持已降低成上述guard/if_then_else的条件。不把原RHS求值当作索引，不复制任何目标数据load。固定域验证索引算术不溢出；不承诺任意C++ signed overflow的wraparound。

### 12.3 TMA构造、host launch和device call闭环

限定实际backend=tvm_ffi，descriptor类型仅现有tiled配置。保存pipeline返回的完整host/device IR（含JSON）及tma_descriptor_args属性，沿host kernel launch的实参位置绑定device形参；匹配metadata中的descriptor Var、tensor data Var和真实输入/输出tensor角色。不能把wrapper.py的另一个backend当作实际路径证据。

独立worker在编译/launch作用域内临时包装已有FFI函数 `__tvm_tensormap_create_tiled`，保留原函数对象：记录每次构造的实参及原函数返回值，原样转调原函数；finally恢复。此hook不解引用descriptor或global数据内存，不替换cuTensorMapEncodeTiled，不修改返回值。若该固定backend无法可靠捕获，4A实验停止并修订方案，不用猜测的descriptor继续。

FFI schema由当前runtime.cc固定：共8+4*rank个参数，为map指针、dtype enum、rank、globalAddress、globalDim[rank]、globalStrideRaw[rank]、boxDim[rank]、elementStrides[rank]、interleave、swizzle、l2Promotion、oobFill。记录 `cudaGlobalStrides=globalStrideRaw[1:]`，不能把第0项也交作CUDA API stride。map/globalAddress只用作本次关联，稳定ID由构造site和device参数绑定产生。

调用前记录真实输入tensor data_ptr/shape/stride；输出由原adapter分配，返回后记录output data_ptr/shape/stride。构造调用的globalAddress必须与对应tensor data_ptr唯一匹配，所有dtype/rank/shape/stride/box及设置必须与host metadata和受审配置一致；构造返回0、真实launch成功且事件完整后，才将descriptor关联标为已核实。缺构造记录、重复歧义、未知动态字段或地址不匹配令整个trace失败，禁止生成猜测区域。

device call schema：`tma_load(desc, barrier, shared_ptr, coords[rank], eviction)`；`tma_store(desc, shared_ptr, coords[rank], reduction, eviction)`。load方向global→shared，store方向shared→global；store barrier为空，原tma_store_arrive/wait完全保留。barrier记录实际allocation身份及下标，不读取barrier内容。尾部policy/reduction字段不算坐标。暂只接受当前eviction=0、store reduction=0和已出现的descriptor设置，其他值拒绝。

TMA runtime事件记录coords、shared元素offset和barrier index；shape/stride/box来源为实际host构造实参，区域展开标注derived_from_runtime_operands。bounds首先在descriptor mode顺序下逐维检查，logical tensor坐标由已验证mode映射转换；shared swizzle仅显示参数，不声称解析每元素物理shared地址。不存在逐线程global元素trace。

### 12.4 实际编译边界、记录协议及完成条件

pipeline末端新增日志结构移除后，与同次pre-trace模块结构等价；baseline/pre-trace在同一pipeline边界比较。另临时包装 `_prepare_device_codegen_mod` 保存原函数返回的真正codegen输入IR，finally恢复；artifact.device_mod仅称pipeline后device IR。host IR、原分配、原同步及访问节点不改。对原函数对象的包装和恢复需CPU异常路径测试及实际compile次数检查。

固定 `TLACC1` 每条22个printf数值参数，全部ABI用int32/uint32和%d/%u，避免int64_t与long/long long格式不匹配：bx/by/bz/tx、四个loop值、vector lane、active共10个int32，后面六个64位值各拆成uint32低/高两半，共12个uint32。普通事件v0=原元素offset，其他v1..v5为0；TMA v0=shared offset，v1..v4=coords（rank2余位0），v5=barrier下标（store=-1）。先按原位宽求值再扩展，signedness在manifest保存，主机重组；用负数、2^31/2^32以上及unsigned极值做往返测试。site ID和launch0置于每site固定format内，不额外占参数。

每点manifest记录shape/dtype、地址空间、实际索引/guard表达式、source/site身份、loop元数据、原位宽、vector宽度和TMA关联。预期key来自上表已知域，不依赖stdout次序；缺失、重复、未知key、wrong mask或descriptor关联错误均失败。首先验证覆盖/身份，再解释实际offset；错误但完整的offset应保留并呈现诊断，而不是改成预期值。

新access记录独立validator覆盖racecheck/synccheck/memcheck，数值capture旧入口不改。四例access输出和baseline最终输出逐位一致，沿用第三步reference的最终输出验证。新增64位协议/FIFO容量实验使用最长22参数schema和65536记录，必须实际无截断；不能仅按文本长度估算。失败日志和非法访问的partial状态保留，不标完整。

## 13. v3：4A实测后的descriptor证据方式修订

第12节FFI hook实验已在H200实际编译运行GQA，能捕获四次真实构造及返回0，但恢复hook后进程在解释器退出时因TVMFFIPyCallbackClosure析构发生segfault；增加显式GC仍复现。probe内部validation.json仅标记函数返回前的状态，不代表整个worker成功；两次进程均退出1，保留为失败实验。不得用os._exit或吞掉退出码绕过。该结果推翻FFI hook作为产品路径的可靠性假设，本节替代第12.3节相关包装做法，待独立复审后方实施。

产品路径不替换FFI构造函数，不读取opaque descriptor；使用实际host IR、真实tensor快照绑定、原构造路径成功执行的证据链：

1. 保存实际pipeline完整模块、真实codegen输入和原host函数。提取host tma_descriptor_args，并逐项验证其与host内实际 `tvm_call_packed(__tvm_tensormap_create_tiled,...)` 的参数结构相等；每descriptor恰有一次构造，构造支配该次device launch、没有重写。查找host内唯一目标kernel launch，逐个device参数位置验证descriptor Var身份。未知host控制流或非tiled构造拒绝。
2. host全模块/构造metadata必须与本次baseline模块结构等价，baselineCUDA继续固定授权。实际输入shape/stride与固定样例契约一致；GQA seq_len_kv/seq_len_q分别从K/V与Q/O shape获取并交叉一致，其他动态scalar拒绝。原tensor data Var由原PrimFunc的参数/buffer_map和固定host packed参数索引绑定，不按任意同名Var猜测。
3. 对构造参数中的纯整数表达式，使用本次真实shape绑定求值；保存原表达式、绑定值和结果，明确来源 `derived_from_host_ir_and_launch_bindings`。globalAddress只记录tensor角色和本次data_ptr，构造数值不声称由runtime hook读回；不记录未观测的descriptor指针或原始字节。
4. 当前受核查runtime.cc对cuTensorMapEncodeTiled非成功结果会LOG_FATAL，构造语句支配launch；因而已证明构造路径+实际launch与worker成功可以支持“按这些参数成功构造”，不伪造回调返回值。固定runtime.cc/LowerHopperIntrin及pipeline关键源码摘要；缺任何证据则整个TMA trace失败。
5. 与核查过的固定dtype/rank/globalshape/stride/box/settings逐项核对；GEMM两rank2 descriptor，GQA四rank4 descriptor。GQA输出的shape/stride在返回后由真实output tensor补齐核对，任一不一致不生成成功报告。字段globalStrideRaw与cudaGlobalStrides[1:]保持区分。

TMA日志仍由GPU原call参数直接提供coords/shared offset/barrier index，报告将这部分标为runtime；descriptor数值和区域展开标为host绑定推导。无需修改原host函数、访问或调用。4A仍须证明pipeline/codegen包装恢复正常、两版host结构不变、构造元数据/launch参数关联正确；此次仅替换存在实际退出故障的FFI拦截途径。


## 14. ???????????

TVM ? Var ?? tma_descriptor_args Map ? JSON roundtrip ???? structural_equal=False?????????? Var ??? descriptor ?????????? Var ?????????????????? host constructor/launch ??? Var ???????????????????????????????? codegen ?????????? baseline ??????

FFI probe ???????? returncode??????????????????????? 64 ????????????????? 65536 ?????????????? phase4-validation.md?
