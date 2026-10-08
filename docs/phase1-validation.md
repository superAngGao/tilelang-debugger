# 阶段 1 验收记录

日期：2026-10-08。设计 v2/v3/v4 已通过独立审阅；代码 v1/v2 曾 FAIL，v3 对代码及九个布局契约给出 PASS。其后最终门禁版本已重新运行以下 GPU 验收，并获得 [阶段1最终独立复核 PASS](reviews/phase1-acceptance-review.md)；各轮原始结论另存 `docs/reviews`。

## 环境与边界

Linux NVIDIA H200；TileLang 0.1.12，核查 checkout `2d63708c8ad57196051c4636a1167c5453c73a48`；PyTorch 2.10.0+cu129；CUDA 编译器 13.2.78；Compute Sanitizer 2026.1.1.0。工具固定打印/sync/reduce helper 摘要，并在运行目录保存环境与实际编译配置。

此次结论只覆盖固定构建参数、受审源码与配置支持的观察位置。完整参数见方案和 contracts.json。GQA 使用已记录的四行 epilogue 同步修正版；未修改安装的 TileLang，也未添加 lowering pass。

## H200 完整样例

`artifacts/release-racecheck/summary.json`：五个 case 均通过；每个 case 的 baseline 和 instrumented 分别运行 racecheck。全部输入及最终输出逐位一致，所有原始设备日志通过完整性校验，十次 racecheck 均为 **0 errors / 0 warnings**。

| Case | 采集元素 | tile 最大绝对误差 | 最终输出与 fp32 reference 最大绝对误差 |
| --- | ---: | ---: | ---: |
| GELU | 4096 | 0（输入/输出逐位核对） | 0.0077686309814453125 |
| Sum padding N=257 | 1026 | 0 | 0.00731658935546875 |
| Sum non-padding N=256 | 514 | 0 | 0.00292205810546875 |
| GEMM | 16384 | 0.00002288818359375 | 0.0229644775390625 |
| GQA，两 consumer | 16384 | 每组 0.0000095367431640625 | 0.00017371773719787598 |

误差容差在方案中预先定义；没有因运行失败放宽。GEMM 观察第二轮 K 后的 accumulator，reference 只计算前128列K；GQA分别核对两组64×128 QK tile，再核对完整 attention 输出。

`artifacts/release-synccheck`：GEMM、GQA 两版，共四次 synccheck，均 **0 errors**，同样通过数值及原始输出一致性检查。

## 位模式与同步对照

`artifacts/release-fidelity`：float16、bfloat16、float32、int32，每种128条采集记录逐位符合已知输入；包含正负零、正负Inf、多个quiet NaN payload、浮点极值/次正规数与极端整数。输出字节也逐位相同，四次 racecheck 均零报告。

`artifacts/print-regression`：原始 T.print 实验八个 case，每个打印768条且所有数值正确。默认自动同步与内部局部 patch 的四个 case 为零 hazards；关闭自动同步、只在整个 print 外围加同步的四个负对照各报告3个 hazards、退出86。断言入口验证了全部正负对照。竞争报告不等于此次实际观测到了错值。

## GQA 原件与必要修正

`artifacts/gqa-epilogue` 保存六次隔离对照。原件三次 reference 失败（含NaN；分别527/352/345个错元素）；修正版三次通过，最大误差均为0.00017371773719787598。六次 racecheck 均零 hazards。

实际生成代码确认：原来的 shared stores 后缺少到 leader TMA store 的组内发布顺序；修正版是所有writer proxy fence → 各自128线程barrier → leader TMA store。原件保留在 `examples/gqa/kernel.original.py`，详细来源及协议链接见 examples/README.md。失败结果保留，不纳入“原版通过”统计。

## 静态与失败路径验证

- `test_cpu.py`：7项，源码/驱动哈希限制、原始选点、循环序号、歧义/不支持结构拒绝、记录乱序与重复/缺失/身份/位宽/截断。
- `test_evidence.py`：3项，拒绝失败状态、错误sanitizer类型、失败reference/gate/process、被修改的字节和截断原日志。
- `test_ir.py`：7项，使用真实 TVM TIR、不launch GPU；验证完整与代表writer、拒绝错误源/读/记录索引和额外算术、barrier冲突或部分参与、未知extern、缺失wait及循环范围变化，另拒绝完整元素置换和线程相关的collective循环。

设备IR和CUDA门禁在launch之前执行。有效产物只来自实际编译对象，两个worker禁用缓存。复核已有结果会重查原始记录、tensor字节、执行状态及两版sanitizer证据；不能仅凭旧summary重新标记通过。

布局授权证据在 `artifacts/layout-candidates`：没有授权摘要的插桩worker在launch前拒绝；生成CUDA与早一轮完整数值验收的 `artifacts/acceptance-final` 逐字相同。Reviewer用独立AST分析重新枚举九个点的 `(逻辑index, writer线程, local index)` 配对，全部匹配候选表与契约。最终release目录的门禁继续逐项匹配这些摘要，不自动接受新布局。

无依赖 wheel 构建通过：`tilelang_debugger-0.1.0-py3-none-any.whl`，仅用于本次打包验证，未发布到包索引。

大体积原始数据留在忽略提交的 artifacts；仓库提交方案、代码、受审契约、原始独立审阅和本记录。
