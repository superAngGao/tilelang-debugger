# 第四步设计独立复审 v2

**结论：PASS（设计通过；B1、B2 均关闭）。** 可按方案进入 4A 实验及后续实施。本结论不是新插桩路径的编译或 GPU 验收结果。

- 日期：2026-10-09，Asia/Shanghai。
- 受审文件：`docs/phase4-plan.md`，以第 12 节 v2 优先规则为准。
- 文件原始字节 SHA256：`e1e6af3dc4e9dc44fbcecd771620f30f3a6503be8ee9c64136cdb51d4d944fdc`。
- TileLang checkout：`2d63708c8ad57196051c4636a1167c5453c73a48`，本轮再次核对。
- 范围：复审 v1 两项阻塞及新增规则，核对固定源码、既有 baseline IR/CUDA 和相关 TileLang runtime/codegen 源码；未修改方案或实现，未运行 GPU。

## B1 已关闭：固定映射与事件域

第 12.1–12.2 节明确八类固定源码点的访问角色、索引骨架、vector width、alias、原循环/末端循环、线程或 election 组和记录数量。源/driver 摘要、受审 baseline CUDA 摘要、本次 baseline/pre-trace 同边界等价共同限制了匹配域；零匹配、超量匹配和未知节点均拒绝。此有限方案足以支持当前四例，不需要通用跨 pass provenance。

GEMM 的 `gi_prod` 初始化/递增及 `stage` 关系、GQA 的 `eff` 单次赋值和实际 shape 绑定已有明确规则。日志保留原分支，既不重新读取这些局部变量，也不重复 election。错误索引 fixture 使用独立受审骨架，运行记录仍取原表达式，避免用期望公式覆盖实际错误。

表中的 Sum 每行 512 候选、257 active/255 masked，GELU 每点 2048 分量，以及 GQA load/store 各静态点的范围与现有产物一致。支持域足够具体，可直接进入实现和负例验收。

## B2 已关闭：TMA 构造与调用证据链

第 12.3 节限定真实 `tvm_ffi` backend，保存 host/device IR 和 descriptor metadata，沿 launch 参数绑定 device 形参；增加作用域内 FFI 构造 hook，原样转调原函数并保存实参/返回值。输入及 adapter 分配的输出均用实际 tensor 地址、shape/stride 交叉核对，缺失或歧义即失败，已经形成可验证闭环。

本轮核查 `src/cuda/runtime.cc:322` 的 `TensorMapArgs::Extract`：参数数量确为 `8+4*rank`，顺序与方案一致；`:548` 的 CUDA 调用使用 `globalStride+1`，`:559` 返回 CUDA result。方案正确区分原始 stride 数组与 Driver API 数组。hook 能否被固定 host 执行路径可靠触发属于 4A 实验；方案已规定捕获失败必须停下修订，未把未验证能力当作事实。

load/store schema 明确排除了末尾 policy/reduction 参数；GQA 两处输出明确为 `tma_store`，shared 为源、barrier 为空，保留 arrive/wait。与 `gqa/baseline/device.py:315`、`:517` 一致。descriptor mode、运行实参、区域推导及未解释 swizzle 的边界也已明确。

## 非阻塞的实施检查

1. 先验证三个包装点的正常与异常恢复、原函数恰好调用次数，尤其 FFI hook 在编译前安装并覆盖实际 launch；不能漏记后仍标成功。
2. 严格区分 pipeline 等价检查与 `_prepare_device_codegen_mod` 返回的真实 codegen 输入。结构等价之外，检查新增日志的索引来源、作用域和安全求值；原 vector 访问及惰性 `if_then_else` 保留。末端 predicate 拒绝的规则正确。
3. 22 参数协议在固定域内足够，uint32 高低位避免 int64 printf 格式歧义。实现时按 uint64 位模式分拆，再根据 manifest signedness 重组；测试负数、大偏移、unsigned 极值及未知/重复/缺失 key。
4. baseline CUDA 摘要要明确采用一致的保存/换行规范；摘要变化按方案拒绝，不能自动重建受审契约。
5. 完成最长 schema/FIFO 实验、实际 CUDA 检查、四例输出逐位与 reference 验证及 sanitizer 验收。保留旧 capture 契约和独立 access validator。

以上均是已纳入方案的实施验收事项，不构成新的设计阻塞。本轮只新增本报告。
