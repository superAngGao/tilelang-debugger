# 统一源码采集：独立设计审阅

后续修订指引：正式 TLDBG3 已采用 **D/E/X**，以独立的前端线程域和每线程唯一 E 核对完整性；线程身份为 `frontend`，不承诺等同于物理 CUDA 线程。`point.thread` 筛选 scalar/local/元素的线程执行，整块采集用 `region` 选择逻辑元素，输出线程由可选的 `collective.reader` 单独控制。详见[自动流水线根协议修订](automatic-pipeline-root-design-review.md)和[最终代码审阅](unified-core-review.md)。下文保留原审阅历史，其中 R/E 和物理线程描述属于当时的候选设计，不代表正式接口。

日期：2026-10-09。产品基线：`11cbc13`。最终结论：**PASS（设计，可进入机制实验和实现）**。复审计划 SHA256：`b75ed856472956f8c6fb94f37cf0e4017264ab7a1a66875912679679633eba04`。不表示机制实验或功能已经完成。

审阅范围：`docs/unified-capture-plan.md`、前一份 `docs/source-control-flow-design.md`，并核对现有 `capture_state.py`、`source_capture.py`、`emitters/samples.py`、`emitters/tiles.py`、`protocols/samples.py`。本审阅不修改实现、不运行 GPU，不以阅读源码代替机制实验。

## 协议简化判断

每 point/thread 的 R、独立 attempted 计数、D、最终 E，可以替代逐 scope 的 enter/branch/transfer 事件，证明**满足选择条件的采集事件是否完整**。不要求原程序走遍静态潜在路径，适合零次循环、提前退出和动态分支。算法与路径正确性仍由 reference 和测试确认。

但 attempted=0 只证明“无符合筛选的到达”，不证明源码观察点从未执行。比如点实际执行了 10 次，但用户选择第 11 次，结果也为零。报告应命名为 `no_selected_visits` 或明确说明 `not_executed` 指选定条件。协议保证范围应写进元数据，不能将事件完整性包装成 Parallel 物理分工正确性。

## 首轮意见及复审结果

首轮结论为 REVISE。以下五项现均已在计划中补齐：全 CTA 的固定 R/E 根域、零选中访问的准确含义、预算预留及部分日志丢失判定、collective 同步独立于私有预算、动态 ordinal、独立构建 session 和规范化别名证据。最终复审还确认物理线程坐标的固定线性化规则、顺序 host launch 的时序边界，以及 TLDBG3 明确替代旧设计逐 scope 见证设想。未发现阻止按该计划进入机制实验/实现的剩余设计阻塞。

以下保留首轮问题的具体说明，作为实现和代码审阅检查依据，**不代表这些设计问题仍未解决**。

### 1. 根生命周期和预期域

现计划要求检测整个线程 R/E 丢失，但尚未规定如何在 Group/分支未到达时独立建立预期域。建议每个已编译 Kernel 的 point/selected block/expected physical thread 只在根处产生一对 R/E，计数器也在根初始化；观察点的祖先 Group、分支和循环只控制 D 与计数。参与线程集合由 launch 和已验证的 Group adapter 元数据确定，不能从收到的 R/D/E 反推。

如果某前端无法在根域读取/保存组内私有状态，需要机制实验证明另一种有外部闭合见证的设计；不能仅在 group 内写 R/E 后假设缺失表示组没执行。合法 return 需在真正的函数边界收尾，macro return 不能提前结束整个 kernel 计数。正常尾部和提前返回路径不能重复 E。明确用户线程选择用物理三维坐标还是组内相对坐标，禁止混用。

### 2. 预算、计数与输出丢失

在 launch 前预留所有预期 R/E 的最小预算；连根记录都放不下时拒绝配置，不在运行中静默省略收尾。剩余预算分配给完整 visit 的元素组；用明确公式检查 attempted、admitted、selection cardinality、D key 集合、overflow 和 truncated 标志。

“整次输出或者不输出”只保证 emitter 的准入决策原子，不保证多条 printf 物理上不会部分丢失。准入后缺任一元素必须报告 incomplete，不能重新解释为正常预算截断。建议 admitted 使用 visit 序号前缀，便于独立计算预期 D；如果不是前缀须明确定义可核对的编号规则。溢出采用饱和加 overflow 标志，不能依赖回绕后的计数。

### 3. collective 控制必须对参与域一致

协作采集不能用各线程独立的 visit 计数/剩余预算决定是否进入 barrier。必须规定共同的采集准入条件和共同 visit 身份；reader 的私有预算只能抑制打印，或者通过已验证的组内一致方式广播准入。所有参与线程仍须按同一顺序到达所需同步。

同样，对动态循环次数、分支、用户迭代筛选和线程筛选的组内一致性做源级保守检查。不能从“拥有 Group 祖先”推断整条路径一致，不能用选择 reader 的条件包住搬运/同步。计划应给出接受的明确声明字段与其信任边界；异步生产的 wait 与寄存器 fence/普通 barrier 不可互相替代。

### 4. 多 compile/launch 与运行证据

现有共享 session 是一次性构建状态，并且 emitter 拒绝 point 二次构建。多 compile 必须按 frontend build/compile 分配独立 point 实例与绑定快照，再与 launch 清单关联；重复调用已编译 kernel 不重建，两个 kernel 对同一源点构建不得互相污染。明确不包含所选点的 compile 是透传并登记，还是给诊断，不能错误要求每个 compile 构建所有点。

baseline/instrumented 的比较需核对完整 compile/launch 清单、参数种类/标量 dtype 与位值、输出树及 tensor 别名拓扑。别名用每次调用内的 storage group/offset/shape/stride 记录，不能拿跨进程设备地址直接比较，也不能只比各 tensor 数值。保留传参对象和原有别名，快照只读取；多次原位 launch 的前后链条必须对齐。日志边界包含异常/失败情况，缺末边界不得把尾数据归给下一 launch。

调用前后 device synchronize 会串行化执行。因此本轮应明确支持顺序 host launch，不承诺原有跨 stream 并发/异步 host 交互时序不变；并发使用给明确诊断而非悄悄等价声明。标量输入需按 frontend 参数类型编码，浮点 NaN/负零及 64 位整数不能被 JSON 数字转换损坏。

### 5. 动态选择身份

区分 `value/coordinate` 选择和“第几次迭代”选择。对于负步长、while 重复值、外层重复进入，必须明确第几次相对于哪一父实例。per-point visit 足以唯一标识已选事件，但不能替代用于筛选的每层循环 ordinal；continue 之前也要增加该层 ordinal，重新进入子循环按父实例重置。不能由数据值或实际收到的日志推导这些身份。

## 机制实验与验收门槛

上述修订后，设计可通过并进入实现。Parallel/Pipelined/Group 中私有计数器是否保存真实每线程事件数，仍必须实验；验证多于一个物理线程、多轮相同 pipeline slot、尾部掩码、线程不均匀到达、零次和多次外层重入。二维 CTA 要验证真实 thread 坐标及总数。

collective 至少测试：选择一个 reader 但多人搬运、低预算下同步不分歧、外层分歧与组内一致、异步生产后合法 wait、scratch 复用。R/E/D 负例包含整线程删除、完整 visit 删除、部分元素删除、budget 标志伪造、序号重复和 launch 边界删除；不要求计数协议防御同时篡改所有数据与计数的恶意伪造。

计划通过仅批准实施；独立代码/安装 wheel/H200 验收通过后才能标记能力完成。新的 R/D/E 保证需明确替代上一版方案的逐 scope 事件要求，并保留旧 TLDBG1/2 解码行为。
