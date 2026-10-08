# 阶段 1 独立验收复核

结论：**PASS（阶段 1 本轮限定范围）**。代码 v3 已授权的布局与前置门禁在最终 H200 回归中实际通过；完整采集、两版输入/输出逐位一致、独立数值参考及相应 sanitizer 证据均齐备。没有残留本轮阻塞。

审阅日期：2026-10-08。本轮只读取最终 release 产物并执行 CPU 证据复核，没有启动 GPU，没有重新进行整轮代码审阅或修改实现。范围限定为固定源码/驱动/布局契约下的 GELU、Sum 两条路径、GEMM、修正版 GQA，以及专用四dtype保真测试。原 GQA 交付原件不在通过范围内。

## 最终运行证据

直接对 `artifacts/release-racecheck` 的五个case和 `artifacts/release-synccheck` 的两个case调用当前 `verify_success`，七项全部通过。逐项核对了 run成功状态及模式、非空run_id、两版各一次compile/launch、成功launch-gate、进程returncode=0且未timeout、两版reference成功、实际输入/输出文件字节及metadata、原始stdout严格解析后的完整记录，以及sanitizer命令与完成日志。没有仅凭summary判断成功。

| 最终racecheck case | 完整记录数 | 布局授权 | baseline/instrumented |
| --- | ---: | --- | --- |
| GELU | 4096 | 两point均匹配 | 逐位一致；两份零hazards |
| Sum N=257 | 1026 | 两point均匹配 | 逐位一致；两份零hazards |
| Sum N=256 | 514 | 两point均匹配 | 逐位一致；两份零hazards |
| GEMM | 16384 | 单point匹配 | 逐位一致；两份零hazards |
| GQA | 16384 | 两consumer分别匹配 | 逐位一致；两份零hazards |

`release-synccheck` 的 GEMM/GQA 两版共四份日志均为0 errors，运行、快照和记录同样通过复核。七个case的 `validation.json` 与各自最终passed summary一致。

全部最终输入/输出二进制、metadata、`records.jsonl` 及数值误差结果，又与前轮已经独立复算的对应产物逐项比较，完全一致。五例的前轮CPU独立reference复算详见代码复审v2；因此本轮新证据没有引入新的数值差异，也没有更改预定容差。

## 新门禁与布局授权

最终每个插桩worker的成功门禁都保存 `layout_sha256`，九个point逐项匹配代码复审v3授权的 `tests/layout-contracts.json`。points中的授权列表也与相应摘要一致。最终五case及两项synccheck的CUDA均与授权表的 `verified_cuda_sha256` 匹配。

本轮重新计算了v3列明四个受审文件的SHA256：`ir.py`、`contracts.json`、`tests/layout-contracts.json`、`tests/test_ir.py`，全部保持不变。故本次成功记录对应已经审阅的完整元素配对和collective循环门禁，没有在回归前改回较弱检查。实际frontend/device IR、CUDA、两版源码、compile及environment产物均存在；环境记录为TileLang 0.1.12/H200，缓存禁用，printf FIFO至少64MiB。

## 保真及既有对照

`artifacts/release-fidelity` 四dtype均重新解析原始日志，恰有128条记录且每条bits等于expected；输入/输出文件经metadata摘要核验后逐字节相等。四份racecheck均为0 hazards，完成summary包含四项passed。每项门禁保存并匹配专用测试的 `(i, i, 0)` 配对摘要，CUDA与此前已核验保真产物逐字相同。覆盖float16、bfloat16、float32、int32，包括负零、Inf、不同quiet NaN payload、极端值及原方案规定的模式。

原始打印实验八个正负对照、GQA原件三次失败及修正版三次成功的隔离证据，已在代码复审v2独立核实并保留。本轮没有将负对照的竞争报告或GQA原件的NaN失败改写为成功，也没有声称sanitizer检出了原GQA数值缺陷。

## 验收边界

本轮方案规定的有限采集链路、独立代码审阅、失败前置拒绝、完整样例和原位保真验收均已完成。README与 `docs/phase1-validation.md` 对固定参数、Linux/H200/已核查TileLang环境、原GQA修正和未支持能力的说明与证据一致。

PASS不扩展至任意用户kernel、未知布局或同步协议、直接shared采集、多launch、性能无扰动、reference产品接口或后续诊断功能。修改受审源码、布局或相关协议后应按契约重新审阅验证。
