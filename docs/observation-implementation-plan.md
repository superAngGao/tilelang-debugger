# 嵌套源码观察：首批实施方案

2026-10-09。基于已通过审阅的 observation-strategy-design.md；本轮先闭合嵌套 scalar/local 采集，保留旧 full-tile 路径回归。后续 group/pipeline collective 继续按系统设计逐项验证，不宣称本轮已经支持。

## 接口与源码变换

配置增加 `schema: 2`，其余 source/points 沿用；point 使用 id/line/when/buffer/block/loops，可选 thread（整数或 null）。buffer 先接受已有变量名；其真实类型在 Python 宏展开获取，不解析 lowered IR。旧配置及 TLDBG1 行为不变。

AST 递归建立 observation 的有序祖先路径，支持 Kernel、serial/Serial/range/unroll、Parallel、多层 Python if/elif/else；绑定以原始行列和坐标维度区分。T.If/Then/Else 规范化为等价 if，仅保存原条件表达式一次，保持短路。每层循环入口保存唯一名称的 induction 别名，后续事件引用快照。loops 按原始 loop 行选择 1-based iteration；未选层采全部；Parallel 可选 coordinates。所有边界在宏展开检查为静态有限整数域；按事件预算限制，非固定两层。

仅在相关祖先路径拒绝 while/early exit/未知 context。函数其它位置的 gemm/ws/pipeline 不再造成 scalar 采集拒绝。首批 v2 emitter 为 PrimExpr 和 local Buffer：各执行线程直接原位输出，不加 barrier、不读取布局。fragment/shared/global 整块先保留旧路径；v2 请求这些对象明确报对象策略尚未实现，不能误用逐线程 scalar emitter。

## 记录、覆盖与预算

TLDBG2 单次 printf 输出一条完整事件：point、launch=0、选定 block、thread、event kind、scope index、可变长原循环坐标、element index、原始 bits。manifest 包含有序作用域路径和静态实际边界。scalar 是单条 D；local 每元素 D，元素集合必须为完整 0..size-1，身份含线程和循环坐标。float16/bfloat16/float32/int32/bool/int64 使用明确原位编码，64-bit 用两个 uint32 字段避免 printf ABI 模糊。

每点在 Kernel 入口输出 R 根见证（每个选定线程），在每个祖先分支的原 then/else 中输出 B 见证（arm 0/1；无 else 补只输出见证的 else），不重新计算条件。在分支进入的上下文记录已存在的循环坐标；不在该 arm 内的点不输出数据。root/branch/data 均带自足身份，不按日志顺序配对。

parser 从根的静态线程域开始递归：serial 域枚举原始坐标；分支检查该父上下文的 arm 见证，只有所选 arm 继续向下；Parallel 枚举逻辑坐标域，之后逻辑闭合不推断物理线程映射。对每个实际 D 的线程身份仍检查重复和完整元素。含 Parallel 或线程相关前置分支且无法闭合 logical domain 时覆盖标 unverified；不伪称严格完整。含 Parallel 的 execution_coverage 始终 unverified，无 Parallel 的静态线程/serial/branch 域可严格闭合。缺根、缺分支、缺 DATA 或整次实例不能报告 not_executed。编译期未构建点单独 inactive_at_build，并仍要求所需根和相反 arm 见证；无法建立证据时失败。

确定状态转换：无 Parallel 时按 thread 精确递归。含 Parallel 且指定 thread，或路径含任何分支（无论在 Parallel 前后），logical_coverage=unverified；不从已收到线程的 arm 一致性推断未收到的副本。无分支、无 thread 筛选的 Parallel 路径可以按完整逻辑坐标集闭合；execution 仍 unverified。不同线程选不同 arm 本身合法；同一 thread/branch/context 的矛盾或重复见证非法。所有真实 D 仍须具有自身线程的祖先分支支持，不能使用其它线程的相反 arm 判定未执行。只有完整遍历闭合域后才宣称 not_executed。后续可增加轻量源码依赖分类来闭合坐标确定的分支，本轮不以此为打印前提。Kernel 体内含任何 break/continue/return 先拒绝，以覆盖会跳过观察点的前置兄弟语句，不只检查祖先。

预算在宏构建时根据实际静态循环边界、线程数、元素数和见证上界计算；总事件最多 65536，坐标参数数量受 CUDA printf 32 参数上限检查（资源上限而非两层），超限编译前拒绝。解析也验证事件总数、格式、位宽、域、重复和矛盾见证。输出的 execution/capture_integrity/logical_coverage/execution_coverage/numerical_status 分开；新 schema 不交给旧 tile parser。

## Reference 与证据

沿用隔离 worker、相同输入和 baseline/instrumented 输出位比较，IR/CUDA 只原样导出。samples 的 reference 返回版本化点规格：logical coordinate + element → value，或 thread + coordinate + element → value；键必须精确覆盖实际逻辑域，重复物理副本逐个比较。离线结果保留 execution_coverage，样本比较成功不宣称所有线程实例完整。旧 tensor reference 不变。

## 小测例与验收

新增 examples/nested_scopes，使用整数值编码原始 loop/branch 坐标。kernel 包含三层 serial、交替分支、非零起点、同名循环绑定、Parallel 尾部有效条件、无 else、零次循环；不同配置选择多观察点和线程。CPU reference 独立枚举算法，不读取插桩结果生成期望值。

先测 AST 绑定快照/条件求值一次；协议乱序、重复、截断、删除分支/根/整个实例；精确 dtype（含 int64 超过 2**53）。H200 跑小测例和 TileOPs MaxPool2D（含 indices/bool/int64）、RoPE scalar，分块 Softmax 的分支/循环；local 增加独立微测。现有 source 27 项及 reviewed GELU/GQA 保留回归，必要的 racecheck/synccheck 验证无新增同步风险。错误 reference 和错误计算必须保持可分析并报告 mismatch；缺失日志必须失败或明确 unverified。

实施前独立方案审阅；实现后独立代码审阅，修订通过；H200 证据独立验收。README 以实际验证边界更新，不把设计能力写成已实现能力。
