# 阶段 1 独立代码审阅 v1

结论：**FAIL**。源码白名单、实际编译对象导出、两版原始字节比较和记录完整性检查已经落地；但 launch 前的 IR 门禁尚未满足已通过的 v2/v3 约束，已有证据复核入口存在明确的假阳性。以下 B1–B3 是代码阻塞，B4 是尚未关闭的 GPU 验收阻塞。

审阅日期：2026-10-08。审阅者未参与本实现或方案编写，阅读了 `docs/phase1-plan.md`、三轮设计审阅、`src/tilelang_debugger`、`tests` 及四例源码/驱动。没有运行 GPU、没有修改实现。接受范围始终按完整源码和驱动哈希限定的四例处理；下述修复不要求支持任意 kernel。

本轮检查的文件 SHA256：

- `src/tilelang_debugger/ir.py`：`44C988FDC99628CAC2436836B4818CB92E203AC719CB40F20312B517B37437CB`
- `src/tilelang_debugger/capture.py`：`761E611FB3C01615557699469DA73A239F8FB862549657F61470585CCCBCC520`
- `tests/validate_gpu.py`：`B59B899500516AA8E0AEE1E1AF20FD431D262908FABA3D89EE5AA53A74560FEF`

## B1：staging 门禁没有验证实际读取和源元素映射

位置：`src/tilelang_debugger/ir.py:163`、`:176`、`:227`。

`check_staging_coverage` 只枚举 shared **目标写地址**，证明它们覆盖 `0..prod(shape)-1`。它不读取 staging 的源索引；`check_instrumented` 对源表达式只要求“含一个 local BufferLoad，dtype 相符”，因此 `local[错误索引] + 1` 也满足所谓直接复制检查。打印侧只检查最后一个参数里有一个 shared BufferLoad、与 store 的 allocation 指针相同，没有枚举该 load 的索引、打印循环、记录 index 与物理地址的对应关系。写入覆盖正确并不证明打印读到了同一组已初始化地址，更不证明输出元素来自所选 fragment 的正确元素。

这直接缺少 v3 明确要求的读取覆盖和正确逻辑元素证明。全部记录依然可以有合法身份和完整 index，故 `records.parse` 不能弥补这种误采。四例正常数值验收是必要证据，但不能代替承诺的 launch 前检查。

CPU 负例检查：以轻量 TIR 接口替身调用**原函数**，保留 128 个 barrier 参与者、leader=0、一个代表 writer，以及完整无重叠的 shared[0:2] 写覆盖；把 store value 改为 `local[999] + 1`，printf load 改为 `scratch[999]`。`check_instrumented(...)["passed"]` 仍返回 `True`。这是检查器控制流的负例，不是实际 TVM lowering 或 GPU 运行证据；本机没有 TVM。

最小修复：仅为受审四例的实际 lowered 形式建立可检查的映射。要求 store value 是允许的原值复制形式，绑定实际源 allocation 和索引；枚举 leader 的打印循环，证明记录 index 与 shared 物理地址一一对应，读地址正好等于已证明写入地址。遇到无法解释的转换、算术、索引或布局立即拒绝。实际 GQA 同名 accumulator 继续用对象/指针身份绑定。增加真实 TIR 的错误源索引、额外算术、错误读索引及正确归约代表 writer 的 CPU 测试。

## B2：原同步协议的检查与 fail-closed 范围不完整

位置：`src/tilelang_debugger/ir.py:47`、`:91`、`:133`。

`manifest` 只保存带 `id` 的 named/CTA 同步事件及 guard 文本；不保存循环上下文，也不保存 WGMMA wait、mbarrier wait/arrive、TMA 相关同步和 operand fence。因此代码注释“Preserve the full original synchronization call sequence and guards”及产物中的 `original_barriers` 不能代表已通过设计要求的完整原协议检查。

CPU 直接调用 `manifest` 的负例：一个版本包含 `named_sync(id=1,count=256)` 于三轮循环、`wait_wgmma(0)` 和 mbarrier wait；另一个版本把循环改成两轮、wait 改为 `wait_wgmma(1)` 并删除 mbarrier wait。两者 manifest 完全相同。当前 `check_instrumented` 只对 GQA 新增 fence 之前最近的 wait 作局部检查；GEMM 合同 `fence_regs=0` 不进入该检查。

此外，未知 extern/intrinsic 默认加入 events 后放行，只有名字含 `asm` 的调用被拒绝。`source` CUDA 参数完全未被使用。故“未识别同步拒绝”和 IR/CUDA 资源核查尚未落实。v3 自动 CTA 例外只检查插桩 events 的名字关键字，没有基线同步 inventory 可证明两版均无异步路径。

最小修复：为四例当前实际 IR 的同步调用建立有限识别表，记录类型、参数、实际指针身份、guard 和循环范围；比较移除明确 monitor 操作后的原协议，保留 GQA 新增 fence 和 GELU/Sum 自动全 CTA barrier 两项明确例外。未知调用若可能影响同步必须拒绝；可采用四例受审调用集合，不需要通用同步证明器。检查两版例外适用条件，并保存原协议差异及实际 CUDA 资源证据。添加 wait/count/loop/未知调用改变必定在 launch 前拒绝的负例。

另须核实 GEMM：源 `c_local` 是 128×128、consumer 为128线程，但源中的显式 `warpgroup_fence_operand(c_local,num_regs=64)` 只声明64寄存器。当前门禁完全未验证它的覆盖。本轮尚无实际 GEMM allocation/CUDA 可独立核对，不能据逻辑形状直接断言发生错误；复审应提供实际寄存器读取范围与 wait 后 fence 覆盖证据，若仅覆盖一半则修复受审样例或 monitor，并重新验收。已有 normal/racecheck 数值通过不能独立证明 compiler operand ordering。

## B3：`--verify-existing` 能把失败或错误类型的证据重新标为通过

位置：`tests/validate_gpu.py:28`、`:65`、`:77`、`:90`。

`verify()` 不检查 `run.json.status`、两版 execution/launch-gate、输入输出字节一致性、原始 stdout 的记录完整性或 sanitizer 模式；只校验已经整理出的 tile，并把 `instrumented/reference.json` 原样装入一个顶层 `passed=True` 的对象。`--verify-existing --sanitizer racecheck` 对已有 normal 目录也没有类型核对。一个运行已失败、输出比较已失败或 sanitizer 证据缺失的目录，只要保留可用 tile 文件，就可被该入口认证。

CPU 状态传播负例：隔离 torch/tile 数值操作（替身返回空点列表），准备 `run.json={"status":"failed","error":"sanitizer failed","sanitizer":null}` 及 `reference.json={"passed":false}`，调用原 `verify()`，结果为 `{"passed":true,"tile_reference":{},"output_reference":{"passed":false}}`。替身只隔离本机缺少 torch 的数值部分；该结果不代表真实 tile 已验过，而直接证明状态字段没有参与判定。

`summary.json` 又只在每个成功项后覆盖写入，失败项没有失败摘要；复用已有输出目录时还可能保留先前整轮全通过的 summary。进程非零退出是正确的，但持久化验收证据仍可能给出相反结论。

最小修复：复核已有结果前要求完整、成功且类型匹配的采集证据；重新检查实际两版 input/output 字节与 metadata、执行计数、门禁状态、原始日志的完整记录，以及两版 reference 成功。明确核对所请求 sanitizer 的两版命令、退出状态和日志。启动验收时写入本轮 running 状态，任何失败原子写入 failed/partial 结果；禁止残留全通过 summary 被当成本轮结果。无需重新运行 GPU 才能补齐这些 CPU 负例测试。

## B4：GQA baseline NaN 与剩余 GPU 验收仍阻塞阶段通过

实施方在本轮审阅期间报告：GELU、Sum padded/unpadded、GEMM 的 baseline/instrumented racecheck 和 tile reference 已通过；四 dtype 特殊 bits 的 racecheck 已通过。GQA 原始 baseline 在 racecheck 环境连续两次 reference 失败并出现 NaN，sanitizer 报0 hazards；一次 normal 成功不能覆盖该失败。

这些是实施方同步的运行信息，本轮尚未取得并独立检查正式本地 artifacts，不能写成 reviewer 实测。NaN 是实际数值失败，不能把0 sanitizer hazards当作整体通过。实施方正在隔离 GQA epilogue shared→TMA store 的同步问题；本审阅不预判根因或修复已经成立。

最小关闭条件：保留失败原件、定位及原版对照；若改动原样例同步协议，明确记录其修订并更新受审契约，然后重新运行 baseline/instrumented 全部相关数值、两个 consumer tile、racecheck及适用的 synccheck。还需归档四 dtype原值保真和原打印实验回归的证据。复审只需覆盖实际修复和既定验收范围。

## 已确认的正确部分与测试限制

- `prepare` 同时约束完整 kernel 和 driver 哈希；公共 run 路径不因 AST 可解析就放行未知 kernel。无需把可信用户脚本或可编辑内部 request 当成安全沙箱问题。
- 实际 `tilelang.compile` 返回对象用于导出 frontend/device IR 和 CUDA；缺少 device artifact 会先失败。插桩 wrapper 只有 gate 返回后才返回 launch 闭包，未发现四个受审 driver 绕开该闭包的调用。
- 输入在实际 launch 前按 dtype/shape/stride/原始字节保存并对比，输出也按字节摘要严格对比；reference失败会让worker非零退出。NaN不会被普通浮点相等吞掉。
- GQA 当前 fence 检查使用 staging源 allocation 与 fence pointer 的 `same_as`，没有仅用 `acc_s` 字符串配对；这是正确实现方向。
- 每 point 从基线未用 named ID 分配独占ID、保留0，插桩检查恰有两处相应count，并验证选中block/iteration中的参与域。monitor原值编码先同宽 reinterpret，再转uint32，协议一次printf输出完整身份；记录解析拒绝缺失、重复和错误身份。
- `PYTHONPATH=src python -m unittest discover -s tests -p test_cpu.py -v`：**7项通过**。本机没有 torch/TVM，没有执行真实TIR负例、CUDA编译或GPU测试。现有7项CPU测试不覆盖IR门禁、失败时禁止launch、barrier冲突或已有证据复核，不能将其通过扩大为上述门禁已验证。

修复 B1–B3、提供 B4 的闭合证据并复审后才能给代码/阶段验收 PASS。本文件保留为 v1 原始审阅，不应在修复后覆盖其失败结论。
