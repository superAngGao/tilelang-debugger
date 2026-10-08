# 阶段 1 独立设计复审 v2

结论：**PASS（设计审阅）**。v1 的 B1–B3 已在第 10 节得到具体、可实施的处理；可以进入实现。本结论不是 H200 功能验收，也不替代后续独立代码审阅。

审阅日期：2026-10-08。审阅对象为 `docs/phase1-plan.md` v2，文件 SHA256：`7D8126DC1070952348AA1CA06A60D47D29B3095199F60DFBE96CC3EE9C920488`。第 10 节明确优先于前文较宽表述，本次按其收紧后的范围判定。复审未运行 GPU，未修改方案或实现。

## B1：已关闭

首版接受范围收紧为四个完整样例的受审源码契约，包含完整文件哈希、固定构建参数、精确 AST 锚点，以及构建时和实际 IR 的复核。未知结构拒绝，源码变动须重新审阅契约。这避免在本阶段隐含承诺一个通用异步数据流证明器，范围可控，也不需要新增 lowering pass。

- GELU 只观察整 tile copy 或完整 Parallel 后的 fragment，并限定无尾块问题。
- Sum 在完整 pad/non-pad 分支之后观察 x_f32 或归约后的 acc，排除了部分行未初始化时的整 tile 读取。
- GEMM 在整个 ring-slot dispatch 后观察 c_local，只剩外层 ki 的选择；明确 gi_cons 的一致更新和完整 s 覆盖，足以形成受审源码范围内的路径证明。
- GQA 分别绑定两个 consumer 的词法 buffer，锚定 prologue wait(0) 后、kfree 发布前，并由 monitor 添加已有 operand-fence intrinsic。明确只接受固定布局，并检查实际 accumulator 每线程覆盖范围以及 fence 在 wait 后的位置。

GQA 的 `8192 / 128 = 64` 只是待实际布局核实的预期；第 10 节已经要求核查实际分配，因此这里没有把简单除法当成充分证明。实现必须检查该 fence 覆盖 monitor 将读取的实际寄存器范围，不能只检查 CUDA 文本中出现数字 64。与这条设计一致即可，无需再扩大通用布局分析范围。

## B2：已关闭

v2 明确了 baseline 先完成、manifest 交接、插桩真实编译、产物检查、最后 launch 的顺序。import 前禁用 TileLang cache，并对 `artifact.device_mod` 缺失在 launch 前失败，解决了独立进程仍可能命中持久化缓存的问题。

named barrier 采用每点唯一 ID，避免不必要的并发分配分析。基线扫描只用于候选分配；插桩实际 IR 的检查才是 launch 门禁。检查包含两处 monitor barrier 的 ID/count、线程域、观察条件和相对 staging/printf 的位置，并拒绝自动同步碰撞、动态 ID、未知同步 asm 及无法解释的原有同步改变，足以形成失败即拒绝的实现契约。

实际编译参数逐项对比，禁止改变影响同步的配置，保证 baseline/instrumented 的比较没有默认混入 ThreadSync 配置差异。实现时“恰有两处”指实际可识别的 monitor 同步调用及其作用域，不能退化成对 CUDA 字符串的全局次数统计。

## B3：已关闭

v2 给出了固定记录版本、point ID 字符约束、有限循环深度、整数 printf ABI、同宽 reinterpret 后零扩展的具体路径，以及 point 到实际 buffer 对象的绑定规则。一次 printf 包含完整身份和 bits，避免按日志顺序配对。

最小设备端保真 kernel 覆盖四种 dtype、正负零、Inf、多种 quiet NaN payload 和极端整数；快照保留字节及 dtype/shape/stride，NaN 不走普通数值相等，补足了 CPU 解码测试不能证明设备端保真的缺口。这个专用验收 kernel 应有自己的受审测试契约，不应给任意用户程序开启绕过四例契约的入口。

固定预算与 FIFO 请求不会被当作完整性的保证，结果仍严格检测缺失、重复和截断，边界清楚。四例 baseline/instrumented 要求逐位相同，独立 reference 的容差预先固定，出现差异后不得自动放宽，基线公平性要求明确。

## 后续代码及 H200 验收关注点

以下是已获通过设计的执行检查点，不是本轮残留阻塞：

1. 将源码契约、point 作用域和实际编译元数据关联起来；尤其验证 GQA 两个同名 acc_s 不会混配。
2. 证明所有实际 launch 都经过编译对象与 barrier 检查门禁，异常时没有先行 launch。
3. 检查 GQA wait→operand fence→staging 的实际代码顺序和寄存器覆盖；分别验证两个 consumer，并保留完整 kernel 数值及 sanitizer 证据。
4. 保真测试核对整个 GPU 路径的原始位模式；最终输出与 baseline 的逐位一致检查不能被 reference 容差替代。
5. 仅在全部规定验收完成后宣称四例受支持；任一测试或子进程失败必须使正式验收返回失败。

无新增设计阻塞。可按 v2 实施，后续支持范围以契约和实际通过的 H200 验收为准。
