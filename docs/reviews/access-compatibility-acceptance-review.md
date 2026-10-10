# 源码访问表达式与版本兼容性独立验收

日期：2026-10-09。独立 session：`access_compatibility_acceptance`。

**结论：本轮验收通过。** 最终安装包在 TileLang 0.1.12 和 0.1.15 上的测试与声明范围一致；未发现尚未解决的验收阻塞项。该结论限定于下列版本、测试组合及明确支持的表达式，不推断任意 TileLang 版本或任意 kernel 都兼容。

## 安装包与运行身份

工作目录：`/home/ang.gao/tilelang-debugger-access-final-20261009`。

最终 wheel：`artifacts/tilelang_debugger-0.2.0-py3-none-any.whl`。

```text
SHA256 4b920e012ca6ada9f2897012eb382d640dfb38205bb07fa339a32b9930eca700
```

独立下载 wheel，核对其中全部 56 个 Python/JSON 包成员与本地当前源码逐字节一致；远端 `src/`、`installed/` 与同一 wheel 也一致。另核对远端 52 个测试及统一样例 Python 文件与本地一致。

逐一检查全部 capture 的 baseline/instrumented 环境记录，debugger 均来自上述目录的 `installed/`，不存在源码路径覆盖。两套环境的实际 TileLang 导入分别为：

- 0.1.12：`/home/ang.gao/workspace/tilesight-gemm-h200-goal-20260825/tilelang-v0.1.12-src/tilelang/__init__.py`。
- 0.1.15：`/home/ang.gao/tilelang-debugger-compat-015-20261009/lib/python3.12/site-packages/tilelang/__init__.py`。

## 独立执行与结果

本 session 未修改实现或测试，也未启动重复 GPU 工作负载。GPU 执行使用主矩阵留下的最终安装包产物；本 session 独立重跑完整 CPU/TIR 测试，并离线复算所有 GPU 产物、独立索引期望和数值 reference。

| 检查 | 0.1.12 | 0.1.15 |
| --- | --- | --- |
| 完整 CPU/TIR 独立重跑 | 14 套、117 项通过，无跳过 | 14 套、117 项通过，无跳过 |
| capture 证据复算 | 21 次通过 | 21 次通过 |
| baseline/instrumented launch 对 | 33 对通过 | 33 对通过 |
| 访问索引独立期望 | 16 组、3,103 条逻辑请求通过 | 16 组、3,103 条逻辑请求通过 |
| 数值 reference 独立重算 | 10 份全部完整且匹配 | 10 份全部完整且匹配 |

访问验收复用并独立执行 `validate_access_contexts.verify`、`validate_access_expressions.verify`、`validate_source_access.verify` 和 `validate_source_access_tileops.verify`；先阅读其期望集合构造，确认预期索引来自 fixture 规则而非实际日志。动态视图另按 n=17/39、stride=1/2/0 独立枚举并核对。

覆盖内容包括：

- pipeline/group/copy 的全部前端线程与指定线程，循环坐标及迭代序号，global/shared/fragment 起点、区域和操作角色。
- 条件表达式、布尔路径、链式比较、位运算、keyword copy、单元素 atomic add/min/max 和 address 请求；核对 baseline/instrumented 前端目标读取、copy、同步、原子及取址次数不增加。
- 原索引与偏移一位、masked 访问、分支内到达、while 的 break/continue、copy 尾部部分相交；人为修改派生字节偏移后，证据重验按预期拒绝，并恢复原文件。
- MaxPool、RoPE、Softmax 的上游源码索引集合，以及非连续/零 stride 输入、动态大小、kwargs 和调用方 stream。
- 既有 pipeline/group/fragment/shared/dynamic 数值 reference，运行参数视图 reference，以及访问与数值混合的 reference。

证据复算核对原始记录、计数完整性、编译和源码身份、参数及返回快照、访问报告派生结果、sanitizer 结果与运行状态。所有 capture 均为完整通过；没有把预算截断、拒绝或失败计入成功。

## 文档及边界复核

已核对更新后的 README、[适配方案](../access-expression-adaptation-plan.md)和[验证记录](../access-compatibility-validation.md)：仅声明实测 0.1.12 与 0.1.15，不把前端接口探测等同于 GPU 验收，也不推断 0.1.13/0.1.14 已通过。

访问报告表示源码逻辑请求；address 不冒充读写或物理地址，atomic 目标标为 read_write。未知宏、嵌套内存读取、整块原子、危险重放、链式/解包赋值及 mutable local.var 的副作用依赖仍明确拒绝。没有自动生成算法 reference，也没有声称范围内索引必然符合算法预期。

失败 kernel 的部分证据恢复、手写 pipeline 调度数组和跨 stream/Graph 等明确不属于本轮。本验收不批准这些尚未实现的能力。代码层问题及修订依据分别见[表达式审阅](access-expression-code-review.md)和[运行参数/兼容性审阅](access-runtime-compatibility-review.md)。

## 复核产物

以上工作目录下保留：

- `artifacts/v012/`、`artifacts/v015/`：最终 GPU/CPU 主矩阵和原始证据。
- `artifacts/independent-cpu-v012/`、`artifacts/independent-cpu-v015/`：本 session 完整 CPU/TIR 重跑日志。
- `artifacts/independent_access_acceptance.py`：本 session 的离线验收脚本。
- `artifacts/independent-v012.json`、`artifacts/independent-v015.json`：逐 capture、索引期望及数值分析结果。
- `artifacts/independent-v012/`、`artifacts/independent-v015/`：独立数值分析产物。
