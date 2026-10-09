# 用户 kernel 通用接入：独立设计审阅 v1

**结论：FAIL，三个设计阻塞项需明确后再实施产品。方向合理；当前探测证明了部分属性保留与 layout 可读取，但还没有给出足以保证采对值、采对迭代的完整绑定规则。**

- 日期：2026-10-09。
- 受审方案：`docs/generic-kernel-plan.md`。
- SHA256：`b39893727a14d460399308638a364862f43d7ee52424df75d977f05d21564bbb`。
- 已读探测：`artifacts/generic_anchor_probe.py`、`generic_layout_probe.py`；远端 `/home/ang.gao/tldbg-generic-anchor-probe/{norm_rms_norm,rope}.py`；TileLang 0.1.12 `layout/fragment.py` 与 JIT compile/cache/launch 源码。
- 本轮没有修改产品、没有运行 GPU。LayoutInference 新增身份探测仍由主 session 补证，本报告不预判其结果。

## B1：layout → 最终寄存器身份、replica 与观察时点需要明确可验证规则

方案写了“核对数据身份”，但尚未说明跨 pass 的具体关联载体。LayoutInference 的 Buffer、最终 AttrStmt 节点、最终 local allocation 可能经过克隆、flatten、重用和改名；不能用名字、形状相同或下标范围正确替代身份链。

已保存 RMSNorm IR 中，`x_local` 是每线程 4 个 local 元素，`sumsq`/`rrms` 是每线程 1 个元素；这与逻辑形状不同。`Fragment.map_forward_thread` 的公开包装使用逻辑 forward vars，replicate 则可能仍作为自由变量存在。必须指定如何获得 replica 的合法域、替换它并形成 thread/register 映射，而不是任取 thread 0 或默认自由变量为零。

需要补充：

1. 用 AttrStmt 所携带的真实 Buffer/data identity 或明确的 pass remap 记录关联 LayoutInference 与最终 allocation；有克隆/别名时如何证明映射。缺失/多解时在 launch 前拒绝。先用探测证实链条可实现。
2. 显式枚举逻辑元素与 replica，验证 owner thread、local offset、dtype、范围、映射覆盖及冲突；明确确定性选副本策略，保存全部映射证据。映射不是源码准入白名单。
3. late 插入处需要证明寄存器在该 owner 上已有当前观察时点的值：before 点未初始化、原 guard 排除某些 owner、buffer 已被替换/重用，均不得读取或宣称完整 tile。无跨线程读取只能免除新增 barrier，不能免除 readiness/lifetime 证明。
4. 将首版异步边界写成结构规则。例如拒绝选定寄存器写入依赖 WGMMA/未知异步操作或未知参与域；可以保守拒绝相关整个函数，但应明确规则且不能按 kernel 名拒绝。原程序其他区域的已保持 TMA store 不等于寄存器写入异步。

## B2：原循环身份与完整性不能仅靠最终循环枚举

方案要求选择原 serial 第几轮，但目前只有语句属性的探测，尚无循环身份载体和重写后的关联验证。最终 IR 可能展开、合并、复制或删除循环，单靠 Var 名或遍历顺序会误选；用错误关联的最终域生成 expected keys，仍可能得到“完整”但实际采错轮次的报告。

需要补充：

1. 每个用户选择的原循环带唯一来源身份及原始 induction value/ordinal 表达式进入 IR；定义这些表达式到 late 插入点的传播和合法性检查。
2. 首版允许哪些变换（例如保留的静态 serial、可证明的常量展开），如何处理多个 marker 副本；身份不明的 pipeline/循环变换在 launch 前拒绝。
3. expected keys 由来源身份、实际有限线程/loop/guard 域及确定性 replica 选择共同导出。原访问 guard 与日志 guard 一致；不能把互斥分支、inactive mask、marker 丢失/重复误报为完整。
4. 为非零起始/step、嵌套/同名循环、展开和原条件执行明确支持或拒绝规则，并加相应正/负验收。无需首版支持所有情况，但必须避免悄悄采错。

## B3：普通 driver 的执行、reference 与失败状态需与现有样例约定解耦

目前受审 worker 依赖 driver 产生 `reference.json`，而普通用户 driver 不一定使用本仓库 helper。现有 TileOPs helper 数值不匹配会返回非零；如果通用 worker 直接把这些行为等同于采集失败，就无法完成方案要求的“错误值可采集并与 reference 比较”。反过来，也不能吞掉任意 driver 异常并宣称运行成功。

需要补充：

1. 无 reference 的普通 driver 如何成功采集，如何标记 `not_checked`；用户 reference 的明确入口（可沿用离线 analyze）及其签名/输出规范。新增 examples reference 要真正覆盖选定中间点，不能只复用目前的最终输出基线 reference。
2. 区分进程/编译/launch/完整性失败、正常完成后数值不匹配、driver 自身异常/非零退出。明确不吞异常；用于错误计算验收的 driver 如何正常完成并交给分析阶段报告 mismatch。
3. 代理如何归一化实际 out_idx（负下标、多个输出、显式输出/in-place 是否支持）、在 launch 前后保存实际参数，并检验 baseline/instrumented 输入与输出位一致。未支持形态在 launch 前给出拒绝原因。
4. 导入 hook、linecache、compile/JIT/layout 包装与缓存的作用域和 finally 恢复；只计目标编译或所有编译要明确，目标标记未出现时不能直接运行后宣称支持。现有 JIT 内存缓存和磁盘缓存都必须纳入策略。

## 已接受的方向与验收约束

- 默认通用路径不按用户 kernel/driver/CUDA/layout 哈希准入；原代码仅显式 `--engine reviewed` 使用，不自动回退。这符合用户要求。
- 原包环境中的实际 import hook、源码属性、原 pass 结果读取、late printf 和两处擦除等价性，是可继续推进的路径。源码属性 probe 的 CUDA 相同仅是阶段性证据，不能替代正式结构等价门槛。
- fragment 各 owner 读取自己的 local register 可避免 shared 中转；RoPE 至少真实记录原访问索引即可，不要求为无 fragment 的 kernel 造一个 tile。
- 访问值来自实际 lowered operands，不能用预设正确公式拒绝合法错误索引。原向量访问、guard 和同步协议必须保持；动态 gather 尚不支持时拒绝，不能重复 load。
- H200 必须取得三类的实际 debug 记录、数值 reference 和 sanitizer 证据，以及改名/空行/参数变化/合法错误计算测试。新算子只增示例配置/reference，不能再添产品算子分支。
- 这些阻塞项要求把已有设计意图落成可检验规则，不要求新增 C++ lowering 或通用自动 reference，也不要求首版支持全部异步与动态控制流。
