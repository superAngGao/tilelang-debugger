# 第三步独立代码复审 v2

结论：**PASS（代码及本轮已列明的 CPU/H200 验收证据）**。v1 唯一 P2 provider 加载问题已修复并独立复测；没有剩余阻塞性代码发现。五路径最终 racecheck、GEMM/GQA synccheck、历史/新 capture 离线分析以及真实数值失配继续采集的证据均已只读复核。v1 原文保留，不覆盖初次 FAIL 结论。

日期：2026-10-08。本次仍由独立审阅 session 完成，未改实现、未启动 GPU。审阅仅运行 CPU 检查、读取实施 session 产生的真实产物和写审阅文档。详细初审范围及未变化的安全边界见 `phase3-code-review-v1.md`。

## 修复复审

`analysis.load_provider` 现在以 uuid 生成唯一模块名，在执行保存源码之前注册 `sys.modules`。上下文覆盖模块执行、reference 调用及所有返回 tensor 的冻结；`finally` 清理模块。执行内容仍为保存的 provider 原字节，输入 clone、元数据 deepcopy 和严格接口检查保持原状。

独立远程复跑 `test_analysis.py` 共 8 项全部通过，包含新增的 future annotations/dataclass 合法 provider 和导入异常后的模块清理。本机额外独立验证 v1 最小复现已恢复正常、嵌套两个 provider 的模块名/命名空间相互独立、上下文内部异常也清理模块。v1 发现关闭。

初审独立运行的 `test_cpu.py` 7 项、`test_numerics.py` 5 项、`test_evidence.py` 5 项全部通过；修复没有修改这些模块的行为。初审额外验证的实际 CLI 执行错误退出 1、数值差异退出 2，以及损坏/错误 sanitizer 日志拒绝行为仍适用。

受审主要实现文件 SHA256（本地原始字节）：

| 文件 | SHA256 |
| --- | --- |
| analysis.py | 7a9b2d38bd84b96dc40a5cb6d8953595b29850118f35f0ddf4466a1fecadbb01 |
| numerics.py | e77723d40b0b98254060802bc0b3fe15171ab40c683f7ab1f75d137034ff31c1 |
| capture.py | d3550671e58168ecda507c9f9070c73ad967f7806746ddad83666cfebcfb298a |
| cli.py | 8895460c08d2c57551191c0c2926b798443f6e120cf0987575623695d785cf4b |
| evidence.py | dfcdf4ebd402a961ef69ad08db7b544ea7451c09df8864439c347cc59f601bb1 |

## 真实验收产物复核

产物根目录为 `tileops-dev:/home/ang.gao/tilelang-debugger-phase3-20261008/artifacts/`。不是仅阅读汇总 passed：对 capture 重跑严格证据校验，校验进程、一次 compile/launch、launch gate、对应 sanitizer 完成日志、两版快照摘要一致、原始 stdout 完整解析与保存 records 一致。对分析产物重新计算原 capture manifest，核对保存的 evidence.json、provider SHA256、完整 JSONL 行数和错误计数。

| 产物 | 独立复核结果 |
| --- | --- |
| phase3-racecheck-final | 五路径全部严格通过；GELU/Sum/Sum 无 padding/GEMM/GQA 记录数依次为 4096/1026/514/16384/16384；两版 reference 均 true、numerical_status=passed |
| phase3-synccheck | GEMM/GQA 全部严格通过，各 16384 条记录 |
| phase3-historical | 五条历史 capture 全部严格证据有效，五例 reference 全部匹配；错误 reference 分析完成且 matched=false |
| phase3-analysis-final | 最终 racecheck 五例全部 completed + matched=true；错误 reference 为 completed + matched=false；完整 JSONL 行数依次为 8192/1030/518/49152/49152，错误 reference 为 8192 行 |
| phase3-analysis-synccheck | GEMM/GQA 全部匹配，错误 reference 正确报告差异；各分析完整 JSONL 为 49152 行 |
| phase3-numeric-mismatch | 真实两 worker reference=false 但执行/采集完整、4096 条记录，CLI 退出 2；verify_capture 通过，verify_success 独立确认拒绝；正确 reference 离线重分析 matched=true |

数值失配夹具保存的原/修改驱动逐字比较，仅改变 `check_output(out, ref,` 为 `check_output(out, ref + 100,`，未改变输入、kernel、launch、门禁或比较规则。没有通过改写运行后证据制造该负例。

再次检查 `monitor.py`、`ir.py`、`instrument.py`、`records.py`、kernel、monitor 配置和 layout 授权没有内容 diff；contracts 仍仅驱动哈希变化。额外对最终五条 racecheck 的实际 `instrumented/kernel.cu` 逐个计算 SHA256，全部与旧 `tests/layout-contracts.json` 的 `verified_cuda_sha256` 完全一致。

## 保留的失败及结论边界

首次 `phase3-racecheck/gemm` 仍为 failed，插桩 worker 在 `ir.py` 的既有 `original barrier uses changed` 检查处拒绝，未把该运行计入成功。本轮后续独立运行的最终产物通过原门禁，且生成 CUDA 哈希回到既有已验证值。没有放宽 IR/monitor/layout 授权，没有覆盖初次失败目录。

这能证明本轮数值层改动在上述受审配置下完成验收，并保留生成代码变化时的拒绝行为；不能证明编译器偶发生成差异的内部机制已被查明或修复。范围继续限定四例五路径、固定受审构建参数和已有观察位置，不新增任意 kernel 的安全承诺。

代码复审及本轮计划要求的列明验收证据 PASS，可完成本阶段交付；提交时应保留 v1/v2 审阅、首次拒绝说明和精简验收记录。
