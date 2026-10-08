# 阶段 1 独立代码复审 v3

结论：**PASS（代码及本文件列明的候选布局授权）**。v2 剩余的完整元素配对和 monitor collective 循环上下文两项阻塞已关闭。允许使用本轮核实的布局摘要执行最终 GPU 回归；本结论不把尚未执行的新版门禁回归写成已完成阶段验收。

审阅日期：2026-10-08。本轮只读复核实现、契约生成和候选提取脚本、真实 TIR 测试，以及 `artifacts/layout-candidates`；运行 CPU 检查，没有启动 GPU或修改实现。沿用 v2 已独立核实的正式数值、原始字节和 sanitizer 证据，没有重复宣称那批产物来自本轮新版门禁。

受审文件 SHA256：

- `src/tilelang_debugger/ir.py`：`E9CCA8AC5D54DD81FCEB53F5BCD3223CCD4C2589239B409E4EE4B40C4C08013C`
- `src/tilelang_debugger/contracts.json`：`4BBBACD7777ACCB6A479D2EB820F1A540D575092D6A8B504B8EE49B6AA191706`
- `tests/layout-contracts.json`：`092BFD05BF166AE11837B7939947A16071D619E782D80DB37534B1CAC0A33FA6`
- `tests/test_ir.py`：`3AF0B04C39EAEDD45DEBA64C689A6E3F69F42886D744E7A21CB2543C8929415E`

## v2 B1：已关闭

`check_staging_coverage` 在保留完整写覆盖、源索引范围、唯一 writer 和原值复制检查的同时，逐向量 lane 保存 `(dst index, actual writer tx, source local index)`。排序后以固定 JSON 编码计算 SHA256，`check_instrumented` 必须将结果匹配到当前 point 的受审 `layout_sha256` 列表才能返回成功。完整置换改变配对摘要，不再能借两个独立 Counter 通过门禁。

摘要不是从本次运行自动学习并放行。`observed-layouts.json` 明确标记 `passed:false`，只是诊断输出；若摘要缺失或不匹配，函数抛错，compile wrapper不会返回launch闭包。`tests/extract_layouts.py` 不修改契约，已有布局授权时拒绝再次提取，也没有临时绕过检查的开关。维护者生成的契约仍需独立审阅，符合固定四例的边界。

本轮独立验证了候选与已验收布局的关联，不只读取作者写出的摘要：

1. `tests/layout-contracts.json` 与 `artifacts/layout-candidates/candidates.json` 内容完全一致。
2. 五个 case 的候选 `instrumented/kernel.cu` 与 v2 已独立核验的 `acceptance-final` 对应CUDA逐字节相同，且匹配表中 `verified_cuda_sha256`。
3. 用独立 Python AST 遍历序列化 `device.py` 的实际 staging store，枚举线程、enclosing loop、分支 guard 和向量 slice；不调用 `check_staging_coverage`，重新生成全部九个点的配对关系。九个结果均与候选表、诊断观察值和正式 contracts 中的授权摘要一致。
4. 候选编译请求当时没有授权摘要；五个插桩worker均非零退出，stderr明确报 `no reviewed layout contract`。均无 `inputs.json`、`execution.json` 或成功 `launch-gate.json`；结合检查器和wrapper时序，证实诊断没有先launch插桩kernel。baseline正常执行不能误称为整个诊断完全不执行GPU。

授权条目如下；完整摘要由上面的受审文件哈希固定：

| Case | Point | 配对数 | 布局摘要前16位 |
| --- | --- | ---: | --- |
| GELU | gelu_0 | 2048 | 0acdc600ac35a9be |
| GELU | gelu_1 | 2048 | 0acdc600ac35a9be |
| Sum padded | sum_0 | 1024 | 241a9be4963e23d9 |
| Sum padded | sum_1 | 2 | 89a7fa6064f7232a |
| Sum unpadded | sum_0 | 512 | 76cb8aa6a1167184 |
| Sum unpadded | sum_1 | 2 | 9a6ad6b6f51313d6 |
| GEMM | gemm_0 | 16384 | 37fc275dd19b0972 |
| GQA | gqa_0 | 8192 | 9eefdfcdb2e7bb5e |
| GQA | gqa_1 | 8192 | caa2dbda875236e2 |

这项授权依赖受审源码/驱动、观察点和实际配对同时匹配；不授权未来布局变化或自动新增摘要。v2 已独立重算的各tile数值与这里的CUDA字节一致性共同提供候选映射的正确性依据。

## v2 B2：已关闭

`check_monitor_loops` 要求两个monitor barrier的完整loop签名等于point的受审enclosing serial loops，包括变量名、常量min/extent和loop kind。printf的外层上下文必须相同，且同层loop变量对象必须相同；`check_print_read`再要求恰好多一个完整元素循环，并要求记录index和shared读取index都是该循环变量。

固定四例只有GEMM的 `ki: min=0, extent=4, SERIAL`，其余观察点无enclosing loop；当前检查明确符合这一有限范围。线程相关extent不能转换为固定整数，额外或不一致循环不能匹配签名，因此v2的非一致collective负例会在launch前拒绝。没有把该实现描述为已支持任意起点、步长或动态循环。

新增真实TIR测试包含完整置换和线程相关collective循环两个负例，保留完整/代表writer的正例，以及原有索引、算术、身份、barrier碰撞、参与域、未知调用和协议差异检查。未发现修复放宽v1/v2已关闭门禁。

## 测试与剩余执行边界

本轮亲自运行 `test_cpu.py` 7项、`test_evidence.py` 3项，全部通过。独立序列化IR枚举和五份CUDA字节对照也全部通过。本机没有TVM；真实 `test_ir.py` 的7项测试已阅读，实施方报告通过，本审阅没有将该报告写为本机执行结果。

代码及上述候选布局没有残留阻塞，可进入最终GPU回归。最终阶段记录仍需补齐：带布局授权与collective循环门禁的新版本，应能通过既定五case、相关synccheck及保真回归；保存新门禁返回的布局摘要及原始运行证据，再更新阶段验收状态。若实际CUDA、布局或已批准的同步差异发生新变化，需重新核实对应变化，不能沿用本文件自动授权。
