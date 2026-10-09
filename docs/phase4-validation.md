# 第四步验收记录

日期：2026-10-09。环境沿用第三步的 TileLang 0.1.12 固定 checkout、H200、PyTorch 2.10.0+cu129、NVCC 13.2.78 和 Compute Sanitizer 2026.1.1.0。

设计与代码独立审阅均已通过；以下分别记录实际验收和实验失败。

[最终独立验收复核：PASS](reviews/phase4-acceptance-review.md)。审阅者另行重跑42项CPU/TIR测试、复验三种sanitizer的30个worker原始证据与IR、旧数值tile及wheel内容，没有运行额外GPU任务。

## 设计与实现

设计审阅 v1 FAIL，补充源码→site/事件域及 descriptor 证据链后 v2 PASS。4A 实验发现 FFI hook 退出时 segfault，修订为 host IR 与真实 tensor 绑定后 v3 PASS。代码审阅 v1 的 host 门禁、racecheck 解析和 TMA 区域报告问题均已修正；v2 又修正负起始 TMA store 的分类后，独立代码复审 PASS。

实现保留原访问节点、原 vector 宽度、原 elect/guard/barrier；不额外读取目标数据，不新增同步和 shared 采集缓冲。原 pipeline 只运行一次，scoped wrapper 在异常和成功后均恢复。baseline 与 pretrace、擦除日志后的实际 codegen 与 baseline 均结构比较；固定 baseline CUDA 摘要未自动更新。

TVM 的 Var-keyed `tma_descriptor_args` 在 JSON roundtrip 后也会结构比较失败，比较副本将唯一 descriptor Var 键规范化为名称；参数列表、原 Var 身份与真实 constructor/launch 仍完整核对，执行 IR 未修改。该处理经独立审阅原则确认。

## 已完成实验

- GELU：4096 条 runtime 访问记录，两版输出逐位一致。
- Sum N=257：512 候选、257 active、255 masked；N=256：1280 条，覆盖 global read / shared write / shared read。
- GEMM：两个 TMA 事件，选 K 第二轮和 ring slot 第二次；GQA：六个 TMA 事件，含两处输出写回。
- `artifacts/access-protocol-03`：外层进程退出 0，65536 条、22 个 numeric printf 参数无缺失；INT64_MIN、-1、超过 2^31/2^32 的整数及 UINT64_MAX 正确重组。独立 fixture 将加载索引错移一位，日志保留原错误表达式的实际值。
- 10 项新增 CPU/真实 TIR 测试已通过：配置约束、丢失/重复/截断/错 mask、错 offset 保留、host 零次构造/重复 launch、错误 packed discriminator/Cast、错误 descriptor/elect、pipeline 异常恢复及 TMA 区域解释。

协议实验的早期失败也保留：`access-protocol-01` 是 TVM Python `IntImm` 无法接收 UINT64_MAX 的构造问题，改用 `T.const`；`access-protocol-02` 使用 INT32_MIN 作为 `%d` 裸 C 常量，CUDA codegen 未加 cast，C 将其提升导致 varargs 错位。产品的 10 个前缀字段均为已穷举验证的非负小整数；极值通过原位宽→64 位→uint32 对编码。最终容量 fixture 使用同等最长字符宽度的 -2147483647 前缀，记录的 64 位极值没有缩减。

FFI callback 的旧两次失败目录 `pipeline-probe-gqa`、`pipeline-probe-gqa-gc` 中内部 validation 不代表进程成功；工具调用观察到退出 1。使用父进程重跑保留为 `ffi-failure-preserved`：子进程 returncode=-11（SIGSEGV）、timeout=false，外层 validation 明确 passed=false。产品完全不采用 FFI hook。

## 最终验收

| 路径 | 记录数 | racecheck | synccheck | memcheck |
| --- | ---: | --- | --- | --- |
| GELU | 4096 | PASS | PASS | PASS |
| Sum N257 | 512 | PASS | PASS | PASS |
| Sum N256 | 1280 | PASS | PASS | PASS |
| GEMM | 2 | PASS | PASS | PASS |
| GQA | 6 | PASS | PASS | PASS |

目录分别为 `access-racecheck-final`、`access-synccheck-final`、`access-memcheck-final`。共 30 个 baseline/instrumented worker 均正常退出、恢复 hook、sanitizer 零错误（racecheck 同时零 hazards/warnings）；每对真实输入/输出逐位一致，原始访问记录及最终输出 reference 均通过。独立 reviewer 另从原 stdout、快照字节和 IR 重算全部 racecheck 证据。

CPU 完整回归 `artifacts/cpu-final`：7 个独立进程 suite、42 项全部通过，无 skip。必须按 suite 隔离，因为离线 analysis 测试明确要求进程从未导入 TileLang，IR 测试则需要导入它；最初把所有文件合并 discover 到同一进程，42 项中此隔离断言失败，原日志保留为 `phase4-cpu-tests.log`。`tests/run_cpu.py` 显式隔离并保存各进程状态，没有删除原断言。

旧数值采集 `artifacts/value-regression-final`：GELU/GQA 两对 worker 的 racecheck、完整 tile/reference 和最终输出校验均通过。GELU tile 逐位一致，GQA QK tile 最大绝对误差约 9.54e-6。保留原 `monitor.py`、`ir.py`、数值布局契约和 kernel/driver 未修改。wheel 构建成功，已核对包内包含新 access 模块与 `access-pins.json`。

所有失败原始产物保留，不用内部 passed 或 sanitizer 单项结果替代完整成功。

产物位于本地 `artifacts/` 与 H200 工作区 `/home/ang.gao/tilelang-debugger-phase4-20261009/artifacts/`，不提交大体积日志或 tensor 快照到 public repo。

## 边界

TMA store 的负起始坐标单独标记 `invalid_store_origin`，非负起点、tile 尾部越界才解释为丢弃越界部分。规则依据 [NVIDIA PTX ISA tensor addressing](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-addressing)；CPU 负例覆盖该区分，未故意执行非法 GPU store。

这是固定样例、固定源码/driver/CUDA 的索引观察；没有任意 kernel 支持。TMA coords/shared offset/barrier index 为 GPU runtime 实参，descriptor 和 tensor 区域为 host IR/真实 tensor 绑定推导。记录不代表事务完成或物理 shared swizzle 布局；sanitizer 不能替代数值 reference，也不保证覆盖所有竞态。
