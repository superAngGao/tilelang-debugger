# 第三步独立设计审阅 v1

结论：**PASS（设计；允许按本方案实施）**。未发现必须先修改方案才能实施的阻塞。通过范围是现有四例、五条受审路径的 reference 接入与离线逐元素分析，不是实现审阅或 H200 验收通过。

审阅日期：2026-10-08。被审方案 `docs/phase3-plan.md` 的 SHA256 为 `CE897CA5B40AB263233A865656C1D62DAFABAB055419969B237F14114ECF0677`。本审阅独立读取方案和当前文件，不依赖先前对话中的实现解释；没有修改实现，也没有运行 GPU。以下实施核查项是方案已经要求的行为落点，不要求另建架构或扩展 kernel 安全范围。

## 阅读证据和范围

核对了 `capture.py`、`evidence.py`、`records.py`、`cli.py`，四例 `run.py` 及 Sum 的 `run_unpadded.py`，`tests/validate_gpu.py`、`tests/test_evidence.py`、`tests/build_contracts.py`。同时查看了现有 release capture 的 run 与 point 元数据，以及前阶段验收文档，确认旧产物字段和 reference 的切片依据可供复用。

当前捕获链路已保存实际输入/输出字节、元数据、两版源码、观察点、原始设备日志和完整记录。`records.parse` 已依据 bits、index、block、原循环值严格拒绝缺失、重复、未知及截断记录，因此本轮可以在 CPU 上重建完整 tensor，不需要更改 monitor 或 IR 门禁。现有包初始化和 CPU 证据模块不加载 TileLang；分析命令通过独立分支调用 CPU 模块即可满足无 GPU 的离线入口。

## 捕获成功、数值失配和执行错误

方案正确识别了当前阻塞点：各驱动在写 `reference.json` 前调用 `torch.testing.assert_close`，baseline 退出非零后 `subprocess_worker` 抛异常，instrumented 不会启动。只替换驱动末尾的主机 reference 判定与保存即可解除这个阻塞，不必改变 launch 或异常处理框架。

新 helper 必须仅把**已经成功算出且结构合法的 reference 的数值失配**记录为 `passed=false`。reference 计算异常、shape/dtype/容差错误仍向上传播；不能捕获任意 `AssertionError` 后继续。正常独立运行驱动仍需在数值失配时失败。capture 场景下数值失配后保存 execution、继续第二个 worker、生成完整 records，再以 `numerical_status=failed` 和 CLI 退出 2 表达结果，这一分工可实施。

`verify_capture` 从当前 `verify_success` 提取证据校验，后者额外保留两版 reference 成功要求，能够保持现有 `tests/validate_gpu.py` 的严格验收。需要明确落实的细节：

- 两份 `reference.json` 必须存在，`passed` 必须是实际布尔值；缺失、null、字符串或整数不能按 false 放行。新增汇总必须由两份 reference 推导，不能只相信 `run.json` 的摘要。
- analyze 需按 capture 记录的实际 sanitizer 模式调用校验。当前 `verify_success` 默认参数为 None；直接不传参数会错误拒绝已有 racecheck/synccheck 产物。不得通过忽略模式或跳过日志检查解决。
- gate/process/timeout、快照摘要、两版输入输出一致性、原始日志完整性和 sanitizer 完成日志均保留。不得让 numerical failure 分支提前绕过其中任何一项。
- 退出 2 只表示捕获或分析已经完整完成且存在数值差异。证据或执行失败必须退出 1，不能产生 completed 的成功报告。

这些要求已经包含在方案第 2、4、7 节；实施时应直接覆盖相应正负测试。

## Provider 接口与数值规则

`reference(inputs, points)` 足以支持限定的四例，不需要暴露 CUDA tensor、TileLang 对象或调用捕获模块。全部点 key 和全部输出的严格覆盖、禁止广播、CPU/实数/strided/非空约束以及容差验证，使接口错误和数值错误有清晰边界。复制 inputs 和元数据、只调用一次、立即 detach/clone 返回值，可以避免 provider 原地改写影响后续分析实际值。

浮点转 float64 比较、int32 不经过低精度浮点转换、以 reference 为相对容差基准，定义充分。NaN 总不匹配、同号 Inf 匹配、异号及其他 Inf 组合失败、正负零数值相等而 bits 保留，也符合调试需求。实施必须先分类特殊值，再计算有限值误差；不能把 `Inf-Inf` 产生的 NaN 当作正常统计结果，或依赖单次 `max()` 隐去错误。特殊值误差可以按方案使用 null/字符串，但 matched、特殊值计数和失败坐标必须明确。

实际输出中的 dtype 和 reference dtype 应分别记录。浮点 actual 对应支持的浮点 reference；int32 actual 仅接受 int32 reference 且容差为零。非法容差包含 bool、负数和非有限值。provider 导入失败、无 callable、返回结构错误以及 tensor 约束错误均属于 analysis failed，而不是 matched=false。源文件保存与 SHA256 应对应实际执行的那份代码；期望 tensor 的 bits/值需保留，使报告可追溯。方案已明确可信本地代码而非沙箱，无需在本轮新增 Python 安全执行系统。

## 四例映射和旧契约

四例 reference 的数学表达式均能从现有真实输入构造。GELU 的 y_reg 从原先“与最终输出逐位相同”的采集保真验收，新增为独立 GELU 数学结果比较；两种检查必须继续区分，不能用放宽容差替代旧保真验收。

Sum 可以从实际输入列数区分 N=256/257，补零长度由观察点 shape 决定。GEMM 必须使用 `points[*].loops[*].iteration` 的选定迭代次数计算 K 前缀；已核对旧产物 iteration=2 而 `loop_values=[1]`，不能误用原始零基循环值直接乘 64。GQA 的 consumer 行偏移可由 leader/128 推出，既有真实元数据包含两个 consumer；Q/K 切片和完整 attention 仍限定已验证的 noncausal/softcap=0 参数。

驱动 helper 接入会改变五条驱动哈希。`tests/build_contracts.py` 同时生成 monitor 配置、无 padding 驱动及 contracts，故运行生成器后必须逐项比较差异，确认只更新驱动摘要，kernel/source、point、layout 和门禁授权没有连带变化。离线读取旧 capture 不应调用当前驱动白名单重新授权旧驱动；使用旧产物及其证据校验即可。报告的源码身份应来自保存的源码/元数据，不能错误地引用当前工作区已经变化的文件。

## 证据真实性和验收充分性

本方案复用的是持久化证据的一致性与完整性校验，不是对任意人为伪造的整个目录提供密码学真实性保证。由日志重新解析 records、检查原始快照摘要和两版字节一致，优于只接受旧 summary；保存实际分析所读取文件的 SHA256 清单则提供本次分析所依据版本的追踪。不得把测试中手写的 passed 状态、修改过的真实日志或人为拼接的 records 当作 H200 运行证据。

计划中的 H200 正例覆盖五条 racecheck 路径及 GEMM/GQA synccheck，且继续要求输入输出逐位一致和完整采集；旧真实产物离线分析另行验证兼容性。CPU/GPU reference 的累计误差如超过预定容差，应定位原因，不自动扩大容差。这些检查足以验证新薄层是否改变原采集链路。

数值失败继续捕获的 H200 负例是必要的，计划已正确要求真实执行两版 worker。验收应同时见到 baseline/instrumented 的 reference=false、两版成功进程与一次 launch、完整 records、capture 成功但 numerical failed、CLI 退出 2、`verify_capture` 通过和 `verify_success` 拒绝。临时测试契约只允许在测试进程内替换驱动摘要，且独立审阅需证实仅偏移 reference；不能用改写运行后 JSON 替代此测试，也不能给产品增加跳过契约的开关。

最终实现审阅仍需检查 provider 异常、输入改写隔离、缺失/多余 point、shape/dtype/容差错误、损坏证据和失败 gate/process/sanitizer 的拒绝行为，以及 running/completed/failed 原子状态与标准 JSON。当前本机没有 PyTorch 不构成设计阻塞，可在已有 PyTorch 的 H200 主机运行 CPU 分析和测试；这不表示离线分析需要启动 GPU。

## 审阅结论

无设计阻塞，方案 PASS。允许按现有模块边界实施；不得据此修改 kernel、monitor、IR 门禁、布局授权或扩大支持范围。代码审阅和真实 H200 正负验收尚未进行，必须完成后才能宣称第三步交付完成。
