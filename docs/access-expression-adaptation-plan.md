# 访问表达式适配与兼容性修订方案

2026-10-09。状态：设计经两轮独立审阅通过后实施；实施时发现前端延迟求值，补充设计再审阅通过后修正。表达式代码独立复审已通过，见[审阅记录](reviews/access-expression-code-review.md)。最终安装包和双版本验收另见[验证记录](access-compatibility-validation.md)。

## 本轮目标

先补 pipeline/group 元素访问及有 shared/fragment 搬运的自动 pipeline 的访问测试，覆盖全部前端线程和指定线程，核对独立生成的索引、区域、计数和输出。兼容性验证使用隔离安装的 TileLang 0.1.12 与 0.1.15；记录实际包路径，防止 editable 安装覆盖测试版本。

兼容层按实际前端表示适配 `tl.tileop.region` 与 `tl.region`，验证 Buffer/BufferLoad/BufferRegion、copy 区域归一化、既有 Var/Buffer 的身份保护和所需前端 API。能力检测失败应指出具体不兼容接口；版本号及文件指纹记录为证据，不作为源码哈希准入。接口探测通过与 GPU 验证通过分别陈述。其他版本不声称已验证。

本轮不扩展失败 kernel 的部分日志恢复、跨 stream/Graph、物理地址解析或新 lowering pass。

## 发现的问题及统一模型

1. 当前候选收集时立即拒绝，导致选择安全 occurrence 仍被未选的复杂表达式阻止。
2. 布尔路径规则分散：Python and/or 已有条件传播，T.And/T.Or 和链式比较遗漏，可能把未执行请求标为 active。
3. 重放安全只防 selected index 中的部分除法，未统一检查 guard、位移和条件表达式。
4. scalar wrapper、区域操作和未知宏混在同一遍历，操作语义应明确分类。
5. runtime 参数表达式把 Div/Mod 当作 floor 运算，并丢弃 Cast；动态形状的主机求值必须保留前端语义。

按四层处理：候选枚举与选择 → 操作及求值路径模型 → 表达式重放/编码检查 → 前端对象适配。源码解析放 source_analysis；注入放 instrumentation；版本对象差异放 frontend；协议只消费明确操作数。

## 表达式和操作矩阵

| 类别 | 本轮策略 |
| --- | --- |
| Name、整数常量、纯算术/比较/位运算、明确纯标量转换 | 可重放；前端检查 scalar 整数/布尔与编码范围 |
| IfExp、T.if_then_else、and/or/not、T.And/T.Or、链式比较 | 统一计算每个访问的路径条件，区分条件本身的访问与条件分支内访问 |
| //、%、位移 | 若访问/guard 可能不求值，禁止提前计算存在未定义域的表达式；静态安全操作可放行，动态域需用户在有效分支内先绑定标量。整个 predicate 和 region 都检查 |
| 元素读写、AugAssign | 显式 read/write 候选；先按 buffer/operation/occurrence 选择，再验证所选项及其求值依赖 |
| T.copy | 统一绑定 src/dst 位置或关键字参数，保持原调用和可选调度参数不变；按前端区域归一化 |
| T.atomic_add/min/max 的单元素目标 | 显式 read_write 访问语义，仅采集目标索引；原子数据参数仍按普通表达式分析。整块原子操作继续明确拒绝直到有区域适配器 |
| T.address_of 的单元素参数 | 显式 address 语义，仅表示取址请求，不计为内存读或写，不承诺记录物理地址 |
| 嵌套 BufferLoad、未知/有副作用调用、自定义 buffer 宏 | 不自动 hoist；需要用户已有标量绑定或明确操作适配器。未选且不影响所选项的复杂表达式不应阻止采集 |
| float、向量、无法无损编码的 uint64 操作数 | 前端明确拒绝；不得静默强制转 int64 |

不同操作应携带明确 kind/operation，报告中的 address 不能描述为已发生读写。逻辑请求、边界和采集完整性的含义保持不变。原语句不做通用单次求值重写。

“未选且不影响所选项”使用可检查的求值前缀规则：为候选保留原表达式求值顺序中的前缀。遇到前缀中的未知调用、宏、原子或其他可能改变状态的操作，拒绝在整句前观察该候选；已知纯计算和普通读取不算状态修改。未知调用自身的内部访问仍拒绝，后续未选的复杂调用不影响前面的安全候选。赋值按 RHS 再目标索引处理，AugAssign 按目标再 RHS 处理。带 local.var 修改的宏用于验证前缀拒绝；不尝试推测宏内部是否修改特定变量。

实施审阅补充：前端构造 BufferLoad 后，后续 Python helper/宏仍可能先插入 local.var 写入，导致最终表达式使用新值。因此候选还携带后续潜在副作用信息；若前端发现任何被采集操作数含 local.var 加载，则拒绝有这种后续调用的观察。不可变 Var 不受该规则限制。普通赋值最终写入不视为构造期调用；链式/解包赋值可能重复使用未绑定的 RHS，当前对该语句的访问观察明确拒绝，其他行照常支持。验收增加真实前端先构造访问、后宏修改索引的反例。这是前端求值模型补充，不涉及 lowering。

## 参数表达式求值

保留 Div/Mod（向零截断）与 FloorDiv/FloorMod（向下取整）；用整数算法处理大整数，不经过 float。Cast 保留目标 dtype，执行确定的整数位宽截断和符号解释，bool 按非零判定；不支持的非整数转换明确拒绝。前端表达式序列化携带运算结果 dtype；普通 signed 算术溢出、INT_MIN/-1、除零明确拒绝，不替编译器的未定义行为指定结果。unsigned 算术按其位宽取模。直接变量的参数绑定保留 dtype 范围校验，避免主机整数超出实际 ABI。

此处是读取前端维度/stride/参数表达式并离线求值，不解释 lowered IR。真实前端常量替换与简化结果作为对照，覆盖负数、超大整数、Cast 和复合维度。

## 验收

- 源码测试：安全 occurrence 不受后续未选访问影响；被选项在 unknown 操作内部或有副作用前缀后仍拒绝；所有条件表达式真假分支和链式比较；guard/index 中非法除法和位移；copy 位置/关键字组合与歧义；operation schema 支持 read_write/address，atomic/address 原调用次数不增加。
- 真实前端：原 Var/Buffer 身份不变；目标 BufferLoad、copy、同步数量不增加；scalar 整数检查；参数表达式求值对照。
- GPU：pipeline/group/copy 六种组合在两个版本独立验证；新增表达式 fixture 包含条件路径、位运算、keyword copy、单元素 atomic/address，逐请求匹配独立期望；baseline/instrumented 输出一致，sanitizer 通过。
- CPU：完整既有 suite，补动态参数及 view 重叠/跨 dtype/冲突字节回归。
- 文档：公布精确版本矩阵与限制，明确测试版本而非推测支持区间。
- 审阅：设计先通过；最终代码与验收证据独立审阅，修正后复审。
