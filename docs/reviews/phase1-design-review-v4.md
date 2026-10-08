# 阶段 1 独立设计补充审阅 v4

结论：**PASS（设计修订）**。批准第 12 节的 GQA 样例四行 epilogue 修正，以及 GEMM monitor 对实际完整 accumulator 增加 operand fence。两者均有明确范围，不需要新增 lowering 指令/pass。后续仍须对正式修正版执行完整 baseline/instrumented 验收和独立代码审阅。

审阅日期：2026-10-08。方案 SHA256：`DE7C92DB876BD2774C9B600F1CD9CBF7C323DDF3D3D440F423CF188244532F6C`。本次只读检查方案第 12 节、隔离实验脚本、`artifacts/gqa-epilogue` 的源码/CUDA/日志/summary，以及既有 GEMM baseline device IR。未运行 GPU，未修改实施文件。

## GQA：证据与修正范围足够支持设计通过

独立核对结果：

- 原版三次进程均在 `torch.testing.assert_close` 处失败，不是编译失败或 sanitizer 退出错误；分别有 527、352、345 个输出元素不匹配，日志含 NaN 差异。
- ordered 三次均返回 0，reference 通过，记录的最大绝对误差均为 `0.00017371773719787598`。
- 六份 racecheck 日志均为 0 hazards。正确表述是数值验收发现了错误，不能声称 racecheck 检出了该问题。
- 两份保留源码的 diff 只有四行：各 consumer 在 `T.copy(acc_o, Os[group, :, :])` 后分别增加 proxy fence 和 `T.sync_threads(3/4, 128)`。
- 原 CUDA 两个 epilogue 均由各线程写 Os，之后只有选出的 TMA 发起线程执行 proxy fence 和 store；ordered CUDA 的 proxy fence 与局部 barrier 位于 leader 分支之外，随后才由 leader 发起 TMA store。

NVIDIA 的 TMA shared→global 说明要求 shared 写入线程先建立到 async proxy 的可见性，再通过线程同步把这些操作排在发起线程的 TMA 操作之前。该四行修正采用这一顺序。[NVIDIA CUDA Programming Guide：Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)

在这里将全 CTA 同步缩小为各自 128 线程 consumer 组是有明确条件的：每个 TMA store 只读取该组负责的 Os slice，所有该 slice 的 writer 和 TMA issuer 都在同组。已检查的 CUDA 中两组分别使用 Os 偏移 0 和 4096，并分别使用 barrier 3 和 4；原 producer/consumer named 1/2 协议不被这两次局部同步合并。正式产物仍须保持该映射、参与域和原有 TMA store 完成等待。

三次相同 seed 通过不能证明所有输入和所有参数都正确，但结合明确的缺失顺序及生成 CUDA，对既定固定参数契约下的设计修正已经足够。原有完整 reference、逐位 baseline/instrumented 对比和两个 consumer 的 monitor 验收继续保留。

## baseline 公平性与原件保存

正式 GQA baseline 和 instrumented 共同使用同一个四行修正版是正确选择。修复不能只出现在 instrumented 中，也不能把修正版的通过结论记在原始交付源码名下。保留 `kernel.original.py`、源码 diff、原失败证据和修正版哈希，使“交付原件失败”与“正式受审样例已修正”可以分别复核。

新 barrier 3/4 属于修正版基线协议的一部分：必须进入基线资源 manifest，并在插桩对比中保留。它们不是 monitor 新增 barrier，不能在剔除采集节点时被一并删除。修正后的源哈希和锚点需要更新为新的受审契约，原哈希不能继续代表正式样例。

实验脚本当前读取 `examples/gqa/kernel.py`。正式文件修正时，必须同步将实验的原始输入改为保留的 `kernel.original.py`，并验证原件没有已经插入的四行；否则再次运行会将修正版误标成 original 并重复加补丁。第 12 节已经要求从保留原件生成对照，这是实施必须落实的一项，不是增加新的设计范围。

## GEMM：完整 operand fence 修订正确

检查的 baseline device IR 明确包含 `c_local` 的 128 个 float32 本地元素；WGMMA 发射器生成的全范围 fence 位于用户 `wait_wgmma(0)` 之前，用户 wait 后的显式 fence 仅覆盖 offset 0 起的 64 个元素。不能把该后置 fence64 作为随后整个 fragment staging 的完整 compiler operand ordering 证明。

在受审 ring dispatch 后、staging 前增加已有 operand-fence intrinsic，覆盖同一 allocation 的 offset 0 和全部 128 个寄存器，符合原先“不新增完成 wait”的设计边界。GQA 相应覆盖 64 个。两者的数字都是固定布局下的预期，实际编译仍必须核实指针对象、offset、allocation 大小和所有 staging 源读的覆盖范围；不能仅根据 buffer 名或源码 shape 除线程数判断。

原同步协议对比只剔除已经验证归属的 monitor barrier 和新增 operand fence，保留其他 wait、fence、named/mbarrier 调用及其 loop/guard 上下文。特别不能通过“删除所有 fence 后比较”掩盖原始 ordering 变化。这一限定已由第 12 节明确，能够避免新增 fence 成为扩大忽略范围的借口。

## 审阅边界

无新增设计阻塞。批准的是这两个有限变更：GQA 的双侧共同样例修正，以及 monitor 针对实际 accumulator 的完整 operand fence。未批准对任意原程序自动修同步、改变其他异步协议，或放宽输出一致性要求。

正式实现验收仍需证明新增 GQA baseline barrier 与 monitor ID 无冲突、修正版两侧实际协议一致、GEMM/GQA fence 完整覆盖，以及整个四例和位模式保真测试通过。本轮不作代码已通过或 GPU 最终验收已完成的结论。
