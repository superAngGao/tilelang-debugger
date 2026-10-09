# 源码观察策略：统一嵌套控制流设计

状态：系统设计，尚未实施。2026-10-09。

目标：用户在源码选择位置和对象；在 Python 宏展开时根据实际对象选择打印方式。循环与分支统一为有序的嵌套作用域，不为“循环里套分支”等组合分别写适配。不解析 lowered IR，不查询 layout 或寄存器映射，不新增 lowering；保留实际 IR/CUDA 导出。现有源码导入、独立 worker、reference 和证据基础设施继续复用。

## 1. 系统分工

```text
用户选择位置/对象/block/迭代/线程
  → AST 建立作用域树，取得该位置的上下文路径
  → 在原位置插入 capture(value, point_id, context_values)
  → Python 宏展开：识别 Buffer / PrimExpr、scope/dtype/shape
  → 组合选择条件、打印方式、记录方式
  → TileLang 正常编译与执行
  → 按执行实例整理记录，校验完整性，比较用户 reference
```

AST 只处理源位置、名字的词法绑定、控制流结构及有限表达式依赖。原循环、分支条件及求值次序保持不变。运行期变量的值由插入的表达式带出；不能重新执行有数据读取或副作用的条件来推断走了哪条分支。类型在宏展开时从真实对象取得，不在 AST 上猜 dtype。

## 2. 一棵作用域树，任意层组合

规范化节点如下；有嵌套就递归，不再使用 in_loop/in_branch 两个布尔量。

| 节点 | 静态描述 | 执行时坐标 |
| --- | --- | --- |
| Kernel | 源位置、函数身份、grid/线程域 | launch、block |
| Group | 已声明的线程组及可用同步实现 | group id |
| Loop | loop_id、种类、绑定变量、start/stop/step | 原 induction value；必要时其 ordinal |
| BranchArm | branch_id、then/elif/else 路径、原条件 | 所在动态父实例中实际选择的 arm |
| Point | point_id、before/after、观察对象 | 打印记录 |

`scope_id` 由本次原始源码摘要、函数词法路径、节点位置产生；它是一次 capture 内的身份，不承诺编辑源码后保持不变。变量按 binding_id 标识，不按变量字符串识别；内外层都叫 i 仍是两个绑定。Parallel 多维坐标属于一个 Loop 节点的坐标元组。

绑定身份必须连同值一起保留：每层 loop 入口保存其原 induction value 到唯一调试别名，或在宏上下文中保留该层真实绑定表达式。最内层打印使用这些绑定快照，不按当前同名 i 重新查找外层坐标。快照只保存已经存在的循环值，不重复内存读取；它也避免循环体内的同名重新赋值改变迭代身份。

每个观察点的静态上下文是 root 到 Point 的路径，动态实例是这条路径上的坐标。嵌套深度由资源预算限制，不再固定最多两层。

示例：

```python
for k in T.serial(K):                  # L0
    if enabled:                        # B0.then
        for i, j in T.Parallel(M, N):  # L1
            if mask[i, j]:             # B1.then
                val = f(i, j, k)
                # P0: observe val
```

P0 的路径为 `Kernel / L0 / B0.then / L1 / B1.then / P0`；记录带 `(k, i, j)` 和实际线程。分支再套循环、循环再套分支使用完全相同的规则。打印留在原分支中，mask 不额外求值，也不搬到分支外读取 val。

## 3. 递归规则：坐标、到达条件和参与域

遍历作用域树时携带三项上下文：`coordinates`（有序循环绑定）、`control_path`（分支路径）、`participation`（对某个协作线程域是否一致）。

### 循环

- 进入循环时追加该 loop 的绑定和类型；退出时恢复父上下文。记录使用原始逻辑变量，不使用 lowering 后的循环名或展开顺序。
- 对 counted loop，用户 iteration 从 1 起；实际值为 start + (iteration-1)*step。嵌套选择分别绑定具体 loop_id，不按“第一个同名循环”匹配。无选择的层按已声明预算收集所有实例。
- 对 Parallel，记录每一维逻辑坐标；物理线程仅为另一个字段，不能把线程数当元素数。用户可按逻辑坐标或线程筛选。
- 对 Pipelined/unroll，仍沿用原逻辑迭代身份。各类打印与这些循环组合分别验证，不因函数含该构造而拒绝所有观察点；未验证的特定组合不宣称已支持。
- 集体打印要求选定协作线程域内的迭代边界、迭代选择和循环到达路径一致。逐线程表达式打印不强加这条要求。
- 第一轮实施覆盖有界 counted loops 的多重嵌套。while、提前退出、隐藏在不透明 helper 中的重复执行先保留明确能力项；在没有 visit/exit 协议前不能伪造 ordinal 或完整覆盖。限制定位到相关作用域/观察点，不禁用无关位置。

### 分支

- 进入 arm 时追加 branch_id/arm_id，继承已有坐标和参与域；退出恢复。elif 规范化为实际条件链，保留短路和条件求值次序。
- 一个 runtime 分支是否命中，以该次执行的原控制流为准。对于编译期消失的 arm，前端构建清单标为 inactive_at_build，不能当成设备日志丢失。
- 内层一致条件不能修复外层已经分歧的参与域。各层规则组合后，才能决定一个需要协作的打印是否适用。
- 到达性与数据就绪分开：所有线程到达不能证明异步写入已经完成。已有打印 patch 的等待/同步前提仍须满足。

### 对协作域的一致性

仅对需要协作的打印计算 `uniform / divergent / unknown`。相对于指定 CTA/Group，常量、编译参数、block id 和已证明一致的 serial 坐标为 uniform；thread id、Parallel 坐标一般为 divergent；buffer load、未知 helper 的结果为 unknown。简单纯运算组合这些属性，未知保持未知。局部赋值按源顺序维护绑定，分支合流保守合并；遇未知数据流不尝试构造通用程序证明器。

一致性必须相对于 emitter 的实际协作域判断，不能只保存一个全局布尔量。例如 `tx < 128` 对 256 线程 CTA 分歧，但对已声明的线程组 `[0,128)` 一致。切换到 Group 时，要对整条祖先控制路径在新域上重新判定；既不能直接沿用 CTA 的分歧结论，也不能仅清空父层状态。

普通表达式/local 观察在 divergent/unknown 路径仍可逐执行实例打印。fragment 集体采集只有整个路径对实际参与域一致时，才选择对应集体实现。只报告该点与该策略不兼容，不再因为函数中出现 gemm/ws/if 就整体拒绝。

## 4. 对象策略与上下文组合

| 对象 | 读取/打印实现 | 参与约束 | 结果单位 |
| --- | --- | --- | --- |
| PrimExpr（包括已存在的 scalar 变量） | 原位置直接打印，附逻辑坐标和线程 | 不新增集体同步 | 一个执行实例的值 |
| local Buffer | 各被选线程打印自己的元素 | 不新增集体同步 | 线程 × 本地元素 |
| local.fragment | 复用已验证的 fragment→shared→同步→打印→同步 | 参与域完整且原值就绪 | 逻辑 tile |
| shared Buffer | 复用相应 shared 打印及已知可见性约定 | 需要哪组同步由该策略明确声明，不能默认所有线程已完成写入 | buffer 的选定元素 |
| global Buffer/元素 | 指定读取者在原位置读取；保持原程序可见性语义 | 不添加跨 block 全局 barrier，不宣称全 grid 原子快照 | buffer 元素/所选区域 |

策略解析为 `selector + emitter + record_schema`。循环/分支不是另一套 emitter；它们只提供上下文并检查 emitter 声明的参与约束。

thread selection 对表达式/local 指数据来源线程；fragment 是逻辑 tile，选择最终打印者不会把结果变成该线程拥有的寄存器子集。前端必须显示这一差异；当前设计不提供需要寄存器映射的“某线程持有的 fragment 子集”。buffer 的 reader 选择与协作参与域分开，不能让只有 reader 进入 collective。

共享同步资源、组内等待等由已有受验证的 Python 打印实现承担。线程组身份、资源占用和就绪前提不能仅从 memory scope 推测；新组协议的可用性需明确适配并测试，不退回源码/算子哈希白名单。

整 Buffer 打印与表达式读取均是主动观察。不得为了判定路径重复读取 mask；访问追踪另有不额外读取目标地址的约束，本轮不把它混入数值打印。

## 5. 统一记录格式和完整性

保留 TLDBG1 读取兼容；新增版本区分 tile 与 samples，不让旧 parser 把线程私有值当重复 tile 元素。

静态 manifest 保存 point/scope/binding id、源位置、arm、变量名、对象策略、dtype/shape、坐标编码及选择规则。每条设备记录至少含：版本、事件类型、point/scope id、launch、block、必要的线程/组、动态循环坐标，以及元素索引/原始位模式。循环坐标长度由 manifest 决定，不再固定 l0/l1。shape、源码字符串等不逐条重复。

同一静态桩的不同实例用上下文坐标区分。scalar/local 的 thread 是数据身份；tile 的 reader_thread 是来源信息，不改变 tile 的逻辑身份。每个数据记录自带完整可关联身份；不依赖先打印一条标题，也不依赖线程间日志顺序。

完整性分两层：

1. **实例完整性**：一次实际采集有 BEGIN/DATA/END，声明预计元素数及实际输出数；检查键集合、重复、位宽和截断，不能只检查 count。固定单值可以合并成一条有固定格式的完整事件。
2. **选择域覆盖**：是否所有应到达的实例都出现。静态有界域从原源码/前端已知参数确定；分支通过必要的作用域见证记录证明所走路径。BEGIN/END 不能证明整个实例没有一起丢失。

选择域覆盖进一步区分 `logical_coverage` 与 `execution_coverage`。静态 Parallel 边界只能给出逻辑坐标集合，不能据此推导实际 `(thread, coordinate)` 或 replica 的执行集合。逻辑坐标都出现，不等于所有线程执行实例都已采到；不同线程报告同一逻辑坐标也不自动算重复。需要完整 samples 执行覆盖时，须有独立的执行域见证闭合；没有则 execution_coverage=unverified。reference 可以比较已经收到的样本，但不得把这种比较成功升级为所有执行实例完整。此区别不要求读取布局；具体见证协议在实施方案中确定并测试。

作用域见证只针对被观察路径：在原分支各 arm 入口记录已选 arm（无 else 时可加仅用于调试记录的 else），不重新求值条件；父 scope 的闭合域决定应有哪些子实例。没有闭合根域、循环执行证据或必要路径见证时，coverage 标为 unverified/incomplete，不能仅凭无 DATA 宣称 not_executed。嵌套假分支可以据已验证的父 arm 记录判定整棵子树未到达。

该协议须在实施方案中落实到可运行的有限域实例：第一轮先对静态有界 nested counted/Parallel 域及条件路径做严格覆盖验收。无法确定的域仍能保存实际 samples，但不能标成完整 tile 或严格验证成功。动态 while 的完整覆盖等协议另行支持。

元数据和 DATA 都计入输出预算。预算不足在可预估时先拒绝；运行时截断/超时保留失败证据，不标记成功。输入/输出位一致和 sanitizer 结果继续单独记录；不匹配不能被完整日志掩盖。

## 6. 数值格式、分析与状态

dtype 决定编码：复用 FP16/BF16/FP32/int32 原位输出，表达式常见 bool/int64 需有显式编码与测试后加入支持表。向量化不得把多个逻辑值当一个未知 C++ 对象打印；用逻辑坐标覆盖测试实际输出，不读取 lowered layout。

tile reference 继续按 shape/index 对齐。samples reference 显式给出执行键→期望值，或逻辑坐标→期望值及覆盖域；不得把缺失样本补零后 reshape 为完整 tensor。旧 reference(inputs, points) 契约保留，新增 samples 结果需要版本化适配并验证，不悄悄改变旧 provider 的参数/返回值。

状态分别表示：execution（运行是否完成）、capture_integrity（记录是否完整）、coverage（选择域是否闭合）、numerical_status（有无 reference 及是否匹配）。未提供 reference 仍是 not_checked；未执行由路径证据判断；错误数值可完成采集。未证明覆盖不以 passed 完整采集交给旧严格分析入口。

## 7. 验收矩阵与实施顺序

规则测试不枚举任意深度的笛卡尔积。验证每条递归规则、必要的维度交互，再用三层以上混合嵌套确认组合成立。

| 测试组 | 必须覆盖 |
| --- | --- |
| 嵌套身份 | 3+ 层 loop/branch 混合、同名变量遮蔽、非零 start/step、多点同一行 before/after |
| 路径 | then/elif/else、无 else、短路、外层 false、零次循环、编译期 inactive、scope 见证整体缺失 |
| 线程/对象 | scalar、local、fragment、shared、global；全部/指定线程；逻辑坐标不等于线程坐标 |
| 同步 | uniform 和 divergent 分支、内层 uniform 外层 divergent、组内/CTA、单 reader 不缩减参与域 |
| 循环实现 | serial、Parallel、展开、Pipelined 组合；原始迭代键、向量化后逐元素覆盖 |
| 记录 | 跨线程乱序、多桩交错、丢 DATA/BEGIN/END/整个实例、重复、截断、预算耗尽 |
| 数值 | 原位 dtype、正确 reference、故意错误计算、未提供 reference、samples 对齐失败 |

优先端到端：RoPE 原 val/rotated 的 Parallel 内采集 + 分支 scalar + 三层循环/分支混合；同时复跑现有 Softmax/RMSNorm/GELU/GQA。随后验证 shared/local 和已知 group/pipeline 策略，不使用新增算子白名单。

实现落点：source_instrument 的作用域树与上下文、monitor 的对象分派、records/evidence 的版本化记录、analysis 的 samples 对齐，source_capture 继续复用执行生命周期。先独立系统设计审阅；具体协议、支持表与可执行测试闭合后再进行产品修改、代码审阅和 H200 验收。
