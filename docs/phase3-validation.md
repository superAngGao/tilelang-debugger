# 第三步验证记录：reference 与数值分析

日期：2026-10-08。独立设计审阅 v1 PASS 后实施；代码审阅 v1 发现 provider 使用 dataclass 的加载兼容问题，已修复并新增回归。[最终独立代码/产物复核 v2 PASS](reviews/phase3-code-review-v2.md)，无剩余本轮阻塞项。

## 本轮实现

- `analyze CAPTURE --reference FILE --output DIR`：CPU 离线分析全部选中 tile 与最终输出，不加载 TileLang、不 launch GPU。
- 自包含 `reference(inputs, points)` 返回全部点/输出的 tensor 及逐项容差；四例提供可直接使用的 reference。
- `analysis.json`、`elements.jsonl`、`report.md`、provider 源码副本与 capture 文件摘要清单。每元素保留 actual/expected 值、双方 dtype/bits、逻辑坐标、误差及匹配状态。
- capture 完成状态与驱动数值正确性分开：数值失配保留完整采集、run CLI 退出2；执行/门禁/证据失败退出1；原严格验收仍拒绝 reference 失配。
- 原 monitor、IR 门禁、kernel、观察点和布局授权均未改变；五个驱动哈希仅因主机 reference helper 接入更新。生成器显式写 LF，避免跨平台换行漂移。

## CPU 验证

25 项通过：既有源码/记录7项、证据5项、数值比较5项、真实 PyTorch 分析8项。前17项在Windows本机运行；PyTorch分析在Linux主机 `CUDA_VISIBLE_DEVICES=''` 下运行，确认未加载TileLang且未初始化CUDA。

覆盖容差边界、以reference为基准的相对误差、零值、NaN/Inf/负零、int32极值、多维坐标、错误数及前20项预览；非法容差/shape/dtype/非稠密tensor；provider异常/缺少点/多余点/输入原地改写隔离；dataclass+延迟注解加载与finally清理；CLI失配退出2；原始bits保留；错误gate/process/log拒绝与旧strict接口仍拒绝数值失败。

dataclass修复使用唯一临时模块名注册sys.modules，覆盖provider执行、调用及返回tensor冻结，finally清理。独立审阅保留首轮FAIL原文，不将修复后的结论覆盖原问题。

## H200 正例

环境沿用阶段1：Linux H200、已核查TileLang0.1.12源码、PyTorch2.10.0+cu129、nvcc13.2.78、Compute Sanitizer2026.1.1.0。

`artifacts/phase3-racecheck-final` 五条路径全部通过：十个baseline/instrumented worker racecheck零错误/警告；原始输入和最终输出逐位一致；所有采集记录完整；原阶段1中间tile验收继续通过。各例插桩CUDA SHA256逐一等于 `tests/layout-contracts.json` 的既有 `verified_cuda_sha256`。

`artifacts/phase3-synccheck`：GEMM/GQA的两版，共四个worker零错误，同样通过数值和完整性验收。

## 离线分析

最终CLI分析 `artifacts/phase3-analysis-final`：

| Case | tile 元素数 | tile 最大绝对误差（CPU reference） | 最终输出最大绝对误差（CPU reference） | 不匹配元素 |
| --- | ---: | ---: | ---: | ---: |
| GELU | 4096 | x=0；y=0.0077686309814453125 | 0.0077686309814453125 | 0 |
| Sum N=257 | 1026 | 0 | 0.00731658935546875 | 0 |
| Sum N=256 | 514 | 0 | 0.00292205810546875 | 0 |
| GEMM | 16384 | 0.00002288818359375 | 0.022979736328125 | 0 |
| GQA | 16384 | 两组均0.0000095367431640625 | 0.00017371773719787598 | 0 |

保留方案预定容差。GELU y现在直接与独立GELU reference比较；原阶段1的y与输出逐位一致检查仍通过。CPU GEMM最终reference与GPU reference有少量累计误差差异（GPU对应最大误差0.0229644775390625），均在既定容差内，没有放宽容差。

`artifacts/phase3-historical` 对阶段1旧release-racecheck五条真实路径直接分析，全部通过，证明无需用新驱动重采集。`artifacts/phase3-analysis-synccheck` 对本轮两条synccheck路径分析也通过。

以上三组CLI验证各附一个故意错误的reference：给选中tile期望值加100，原始capture未改。结果为completed、matched=false、退出2；全部tile元素报告失配、最终输出仍匹配。完整元素记录和错误坐标保留在各组wrong-reference目录。

## 真实数值失配继续采集

`artifacts/phase3-numeric-mismatch` 在H200运行GELU测试夹具，只将驱动末尾的reference改为 `ref + 100`；输入、编译、kernel、monitor和门禁不变。临时驱动摘要只在测试进程内替换，不进入产品契约或提供绕过开关。

结果：两版worker均完成一次compile/launch，reference均false，racecheck干净；完整4096条tile记录、输入输出逐位一致；run.status=passed、numerical_status=failed，真实run CLI退出2。verify_capture接受完整证据，verify_success明确拒绝。再用正确GELU reference离线分析，所有数值匹配，同时保留原驱动reference失败状态。这是实际运行证据，不是事后改写状态文件模拟。

## 保留的拒绝记录与边界

首次 `artifacts/phase3-racecheck` 的GEMM编译在monitor staging之前额外生成 `tl::__sync_thread_partial(3,128)`，与同时执行成功的synccheck产物相比，CUDA仅多这一条同步。原有IR检查在插桩launch之前以 `original barrier uses changed` 拒绝，整组记录仍为failed；GQA当时未执行。

未修改IR门禁或增加同步例外。随后完整五例重新运行的最终产物均匹配既有CUDA授权并通过。此次仅确认生成产物出现过上述差异，未定位编译器偶发差异的内部原因，不能宣称已修复该编译器行为。若再次生成未受审同步结构，工具仍会拒绝；没有自动重试绕过门禁。本次第三步的通过结论适用于通过原有检查的完整capture。

原始大产物保存在忽略提交的artifacts，本文件与代码、参考实现、方案、审阅记录提交仓库。第四步runtime访问索引采集未实施。
