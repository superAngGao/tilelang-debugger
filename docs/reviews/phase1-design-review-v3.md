# 阶段 1 独立设计补充审阅 v3

结论：**PASS（本文件限定的设计例外）**。允许受审普通 CTA 样例中的编译器自动全 CTA barrier 随布局变化；允许 fragment staging 的写线程集合是 monitor barrier 参与域的真子集。二者都不能解除下文的检查要求。

审阅日期：2026-10-08。本次审阅对象是主代理提出的两项设计修订及现有 v2 方案。未审实现代码，未运行 GPU，未独立验证所报告的 Sum 实际 lowering 变化。有关“原有两处自动 barrier 消失”的描述是实施方提供的发现，不作为本审阅已验证的运行事实。

## 自动全 CTA barrier 变化

v2 对原有同步调用严格保留，作为保守门禁合理，但不适合作为所有编译器自动同步的语义等价标准。当 fragment 布局改变，使 shared 数据从跨线程读取变成线程读取自己的写入时，编译器可以合法删除原来的自动同步。原版与插桩版必须分别正确；其自动 barrier 的文本或数量不必相同。

这项例外只适用于既有受审源码契约和固定参数的 GELU、Sum。接受条件明确为：

1. 源码没有手工同步和异步操作，实际 lowering 也没有出现需要另行完成等待的异步数据搬运。若出现 TMA、cp.async、mbarrier、WGMMA 等相关路径，不得凭样例名称套用例外。
2. 两个版本使用相同目标、backend 和编译配置，ThreadSync 保持开启；例外不能靠修改同步配置实现。
3. 只允许编译器自动生成的普通全 CTA shared-memory barrier 增删或移动。原有 named barrier、手工协议、其他同步类型均不在例外内；GEMM/GQA 及其他 WS 样例继续遵守原有严格门禁。CUDA 中普通 `__syncthreads` 可能最终使用保留的 barrier 0，它必须作为普通全 CTA 同步辨认，不能因此把任意 named barrier 都归入例外。
4. 在 launch 前确认该变化属于上述范围，保存两个版本的实际 IR/CUDA、同步差异及采用的例外标识。对这次 Sum 变化，应保留布局/访存映射变化的可复核说明；仅写“少了两个 barrier，但输出正确”不构成原因定位。
5. monitor 自己的独占 named ID、count、参与域、前后同步及 staging 顺序仍须在 launch 前全部检查，原门禁不能因为存在自动 barrier 例外而整体跳过。

H200 验收须对这些样例的 baseline 和 instrumented 完整 kernel 都运行 racecheck，覆盖 Sum 的 padded/unpadded 两条既定路径，并保留逐位输出、完整采集数据和独立 reference 检查。不能只对 monitor staging 片段运行 sanitizer，也不能把 baseline 的错误作为忽略 instrumented 错误的理由。未解释的报告阻塞验收；对这些有限普通 CTA 样例，可直接要求两版 racecheck 均无错误。

这是一条对固定源码、固定编译环境和已核实同步类型的例外，不是允许任意程序在 sanitizer 没报错时绕过同步门禁。编译器仍负责原程序的自动共享内存同步，工具负责限制适用范围、验证新增采集路径并保存验收证据；不需要为此新增 lowering pass 或通用依赖证明器。

## staging writer 可以是参与域子集

此修订在语义上正确。归约 fragment 可能在多线程间复制值，或仅由代表线程负责将其逻辑元素写入 shared。要求每个 barrier 参与线程都执行 staging store 过强；同步的必要条件是所有写入者包含在参与域中，并且所有参与线程按相同路径到达 barrier。

检查应区分两个集合：完整同步参与域 P，以及实际执行 staging store 的线程集合 W。允许 W 为 P 的子集，不允许用 W 的大小替换原定 barrier count。leader 必须属于 P，并且只有在 P 的第一个 barrier 完成之后才开始读取。

逐索引枚举证明必须覆盖实际 lowered store，而不仅是源码 `T.copy` 的抽象 shape：在选中的 block/迭代下，枚举固定线程域、相关循环及写入 guard，验证每个期望输出元素对应的实际 shared 地址恰写一次、没有越界或意外额外写入，且全部 store 在第一个 barrier 之前。无法求值的 guard、动态范围或无法解释的地址映射应拒绝。逻辑索引与 shared 物理偏移不能在存在布局变换时混为一谈。

同样必须检查：写入来自对应 fragment 的正确逻辑元素；读取覆盖同一组已初始化地址；writer guard 不包住 barrier；打印结束后的第二个 barrier 由整个 P 执行，并位于任何可能复用 scratch 的后续操作之前。若归约值有多份物理副本，只选择实际 lowering 指定的代表写入者，不能容许多写者同时写相同地址，即使预期值相同。

原先的完整性检查和端到端数值/位模式验收继续适用。唯一写入覆盖证明解决 staging 地址的初始化与竞争问题，不单独证明所选 fragment 值计算正确。

## 审阅边界

本补充准许作者将以上两项有限规则写入主方案并据此实施；v2 其余门禁、固定契约、未知结构拒绝及 GPU 验收要求继续有效。后续独立代码审阅需要验证例外分类没有扩大、writer 枚举没有遗漏 guard/布局，以及失败路径确实发生在 launch 前。本轮没有代码通过或 GPU 验收通过的结论。
