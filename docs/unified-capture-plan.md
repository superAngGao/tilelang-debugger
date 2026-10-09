# 统一源码采集实施计划

2026-10-09。起点 `11cbc13`。用户已授权：完善计划、独立审阅、修改至通过后实施，再独立代码与验收审阅。计划状态与实现状态分开记录。

## 目标与范围

在 H200、固定 TileLang 0.1.12 前端上统一源码选择、控制流、对象打印及运行身份。继续使用 Python 源码插桩和现有 printf/搬运/同步宏；不增加 intrinsic，不读取 lowered IR/layout 来决定数值观察策略。IR/CUDA 仍原样保存。旧 schema 1/2 及受审样例兼容。

本轮包括：多重 serial/unroll/Parallel/Pipelined、分支和明确 Group，动态边界/while/合法 break/continue/return；scalar/local/fragment/shared/global 明确选区；共同 dtype 编码；多维 CTA；scalar 参数、多 launch、in-place 的运行证据。访问 trace 泛化、跨 block 一致快照、不连续 tensor、未知前端语义和任意程序 reference 自动生成不在本轮。

## 组件职责

- source_analysis：一次建立 Kernel/Sequence/Loop/Branch/Group/Transfer/Point 模型，解析位置、绑定、祖先和真实跳转目标。循环种类通过 adapter 描述，同一递归算法组合。Pipelined 第三位置参数是 num_stages，保持原调度参数。原边界/条件只在原位置求值；观察记录不得重算有副作用的表达式。
- instrumentation：根据模型原位插入观察调用、计数器初始化及正常/提前返回收尾。break/continue 不需要跳到调试收尾，不改变其目标。kernel 之外的函数体不能全局触发拒绝。保留完整原控制流。
- emitters：前端确认真实对象、dtype、memory scope、shape、launch；scalar/local 原位输出；fragment 使用已有 shared 中转；shared/global 直接读取指定区域。统一原始位编码。reader 筛选与协作参与条件分离。
- protocols：纯 CPU 解码、身份/计数/选区检查；不加载 TileLang，不从接收到的数据反推应有数据量。
- runtime：每次 compile/launch 的身份、独立元数据、输入/输出和原位参数快照，临时 hook 恢复。离线 reference 按 launch 对齐。

## 已审协议：TLDBG3

最终格式须通过设计审阅和前端实验后固定。核心是每个 point、选中 block、实际执行线程维护独立的访问计数：D 数据、E 收尾、X 非法读取。主键包含 run(产物目录)、compile、launch、point、block、thread、visit、element；D 另带原循环坐标，group 来源存静态元数据。visit 独立于坐标，能区分 while 重复坐标、外层重复进入和多 launch。

本方案的 TLDBG3 D/E/X 正式取代旧控制流设计中新增逐 scope enter/arm/transfer 事件的设想，旧 TLDBG1/2 不变。thread 配置保留线性整数，固定为 x + extent_x * (y + extent_y * z)，记录也保存 extent 与可逆前端线程坐标；可接受 [x,y,z] 并在前端按同一规则校验转换。

访问计数在所选观察点实际到达并满足采集筛选后增加，独立于是否输出成功；预算耗尽后仍继续计数，原计算继续。每个完整访问必须输出整个选区或一个元素都不输出，不能把半块当完整块。E 包含尝试次数、已尝试输出的完整访问数及预算状态；计数不依赖收到的 D。计数宽度/溢出需明确检测，不能静默回绕。

E 生命周期固定为每 kernel launch 的每 point、所选 block 内的全部前端 CTA 线程各一次，不受 point 的外层 Group/分支/循环筛选。thread 选择只影响 D，根域仍独立已知。attempts=0 的产品语义是“没有满足选择条件的观察到达”，不是原语句绝对未执行。编译期未构建对象另标 inactive_at_build。若编译器改变根 CTA 域，须从明确前端/launch信息处理并实验验证，不能从收到的 E 枚举参与者。数据域的完整性不等于算法路径正确性。

每次 launch 先保留所有 E 预算，再将剩余数据预算按 point/参与域分配；容量不足以容纳一次完整访问时报配置错误。输出尝试不保证 printf 成功，丢失任何 D（含预算最后一次访问的一部分）由序号/元素集合/E计数检测为 incomplete，而不是正常 truncated。完整截断前缀与丢失日志分别报告。计数器使用 uint64 饱和计数，达到上限标 overflow/incomplete，原程序继续。

协作采集不能用各线程不同的 visit/预算/筛选控制 barrier。协作同步位于满足原参与域一致条件的路径，预算和 reader 选择只控制同步之间的输出；输出访问计数由 reader 维护，参与线程仍完成必要同步。这样本地计数不需要跨线程一致。循环筛选若可能分歧，同样只控制 reader 输出，不能包住 collective。

动态循环配置保留 iteration 为该父实例内从 1 开始的第几次；新增 coordinates 显式选择原循环变量值。while 用循环体入口 ordinal 选择，入口递增不能被 continue 跳过；唯一 D visit 是每 point/thread 整个 launch 单调累计，不因重新进入内层循环而清零。原循环坐标与 ordinal 独立记录，不混用。

零次循环、未进入分支、continue/break 跳过点，允许 E 计数为零。return 需要在真实前端允许的函数边界收尾；返回表达式必须只求值一次，不得改动其副作用顺序。没有收尾证据不能称“未执行”。完整性按线程独立核对，允许跨线程乱序；不要求所有线程重复输出所有 Parallel 元素。整条 end 缺失必须由预期参与域发现，不能只验收到的线程。

预算按 run/launch 的配置分配到 point/thread，计入 E 和 D。预算耗尽报告 truncated，与缺记录 incomplete、数值 mismatch 分离。旧协议严格校验不变。对动态域只用计数证明采集事件完整，不宣称证明原程序算法正确。

Parallel/Group 中的线程私有计数器是否能按预期编译和执行，是实施前必须完成的机制实验。若前端会复制/重排计数器，修正 emitter/协议并复审，不通过降低完整性标准宣布完成。Group 的前端线程身份和根/收尾参与域也需实验核对，不将编译器重新映射后的硬件线程编号与前端组坐标混为一谈。

## 对象和协作策略

scalar/local 不插 collective barrier；分支/线程筛选只控制自身输出。选区使用静态的每维 [start, stop, step] 或明确元素坐标，先做形状边界检查；普通 buffer 元素读取须在原位置处理，打印是新增读取而不是访问追踪。

fragment/shared 的协作顺序是参与线程采集/就绪 → 必要同步 → reader 输出 → 必要复用同步。global 只承诺当前 block 下有明确生产完成条件的区域，不做跨 block 快照。同步策略基于明确前端作用域及声明；组内一致性保留所有祖先条件，不猜测异步完成。对全 CTA 的普通同步点和已验证组内同步模式提供实现；无法满足契约的观察点说明缺少哪项条件。未知情况下不能自动加全 CTA barrier。

共享 dtype 位编码/解码覆盖 bool、int8/uint8/int16/uint16/int32/uint32/int64/uint64、FP16/BF16/FP32/FP64（固定前端实际可用者）；整数比较保持精确。额外浮点格式如 FP8 不声称覆盖。

## 运行接口

编译元数据和每次 launch 分目录保存；重复调用同一已编译 kernel 不能再次构建观察点。多 launch 日志需无歧义分段（调用前后 CUDA 同步和 host 标记），不能把异步尾日志分给下一次 launch。原始设备事件 launch 标识可由 launch 日志边界绑定，但验证必须同时检查完整边界和 worker 调用清单。

本轮运行契约是顺序 host launch；不声称保留跨 stream 并发或异步 host 交互的时序。驱动显式要求并发执行时诊断不支持该采集方式，不能偷偷串行化后声称时序等价。打印本身亦不保证性能不变。

参数模型区分 tensor 与 scalar，保存标量类型和值并与 baseline 对齐。无 out_idx 的原位调用保存调用前/后的参数 tensor，不能误当无输出；不破坏别名关系，不为调试偷偷 clone 传入参数。baseline/instrumented 逐 launch 比较输入、输出和参数变更。编译和 launch 失败保留证据，不把部分运行标为通过。

每次前端构建使用独立 observation session，编译时冻结该构建元数据，compiled proxy 保留独立副本，不能累计 bound 标志或复用可变字典。运行清单包含 compile_id/launch_id、源码身份、参数角色、标量位值和 tensor storage别名组/offset/stride；baseline/instrumented 比较规范化别名结构，不比较跨进程物理地址。只要同一 source 中某些选点不属于该次 kernel，就在该次构建标 inactive，不能将其他 kernel 的点冒充缺失；整次运行必须交代每点是否曾构建及被选择。

## 实施顺序和验收

1. 机制实验：Pipelined/Parallel/Group 下私有计数器、二维线程身份、提前退出前端、协作宏。先保存原代码/插桩代码、reference 与 sanitizer 结果，再固定协议。
2. 源模型、TLDBG3、scalar/local 和统一嵌套；支持正负/非单位步长、依赖外层的边界、while 重复值、前置跳转、零次/未执行、选线程/迭代。
3. 同一路径接 fragment/shared/global 选区、类型统一和多维 CTA。覆盖多桩、同名绑定、组内 reader 与参与线程分离、pipeline slot 复用；检验采集值和最终输出。
4. scalar 参数、多 compile/launch、in-place 和离线 analysis/evidence 对齐；旧入口继续可用。
5. 独立代码审阅与 H200 验收：旧 80 项套件和四例回归；TileOPs pool/indices/rope/softmax/RMS；新机制矩阵；畸形/缺失/重复/乱序/截断/整实例删除/整线程删除/计数篡改等协议负例；reference 真实错误不能通过。

每项证据区分 frontend 是否支持、打印功能是否通过、具体组合测试覆盖。Parallel 是补测试和协议验收项，不能仅因不解析 layout 将其列为功能缺失；同样不能用几项测试声称任意程序线程分工已被证明。

完整验收后更新 README、矩阵和独立审阅记录，构建 wheel 检验实际安装后路径与 GPU 工作进程版本，提交并 push。若真实前端限制导致某一目标无法实现，必须保存复现并在最终状态明确列出，不能静默缩减范围。

## 进度

- [x] 独立设计审阅通过
- [x] 机制实验及必要协议复审
- [x] 统一模型/协议/scalar/local
- [x] 协作对象/选区/dtype/多维 CTA
- [x] 参数/多 launch/in-place/离线分析
- [x] 独立代码审阅、H200 验收、文档
- [x] 提交并 push 最终版本


## 实验后修订：前端线程身份与输出者

真实 TMA pipeline 的源码 CTA 为 32，编译器自动添加 128 个搬运线程。常量开始标记 R 留在自动分工之外，而 DATA/END 位于消费者并映射为前端 0..31，因此原候选协议正确拒绝不一致日志。改为 R 读取 counter 的实验仍失败。审阅后正式采用 D/E/X：已知前端域加每线程唯一 E、独立 attempts/emitted 计数和完整元素集即可核对完整性；不需要冗余 R。无 E 一律失败。基线和插桩保留原编译设置，不关闭自动分工，也不增加 lowered IR 解析。混合 active/inactive 及仅 inactive 的流水线均须复验。

用户提出线程选择语义后，接口进一步拆分：普通用户选择源码对象、循环条件和逻辑 region。`point.thread` 只筛选 scalar/local/元素值的前端线程；整块采集默认由工具选择 reader，高级覆盖用 `collective.reader`。whole buffer 配置非空 thread 明确拒绝，scalar/local 配置 collective 同样拒绝，不静默忽略配置。reader 的线性/XYZ 形式按前端 CTA 归一化，并检查组参与域；reader 不改变搬运和 barrier 的参与者，也不缩小必须收齐的 E 域。记录含 `thread_space=frontend`，不宣称提供自动分工后的硬件线程过滤。

固定前端的条件/嵌套/value return 控制 Python 构建而非 GPU 动态退出，当前诊断拒绝；只支持 kernel 顶层末尾 bare return。手工 Pipelined 调度数组需额外适配，当前诊断拒绝。零维 tensor 快照支持不代表零维 kernel 参数受前端支持。这些边界随真实复现保存，不以窄化测试掩盖。
