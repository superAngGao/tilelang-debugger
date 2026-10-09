# 统一源码采集验证

2026-10-09，起点 `11cbc13`。最终安装包已通过独立设计、代码、runtime 与完整验收审阅，结论 **PASS**，适用范围与边界见下文。

## 实现与审阅

数值采集按 source_analysis → instrumentation → emitters → protocols → runtime 分层。源码递归模型覆盖循环、分支、组及控制转移；前端实际对象决定 scalar/local 或协作 buffer 策略。没有新增 lowered IR/layout 分析门禁，没有修改 TileLang 安装。原 schema 1/2 与 reviewed/trace 兼容路径保持。

先完成[方案](unified-capture-plan.md)及[独立设计审阅](reviews/unified-capture-design-review.md)，再实施、复审。[核心代码](reviews/unified-core-review.md)、[runtime 与证据](reviews/unified-runtime-review.md)、[独立验收](reviews/unified-acceptance-review.md)分别保存结论及适用产物。

实验发现真实 TMA Pipelined 的前端 32 线程经自动 warp specialization 变成 160 硬件线程。候选 R 常量开始标记被放在分工外，D/E 在消费者内并使用前端 0..31，因此严格 parser 拒绝。让 R 读取 counter 的尝试仍失败。经[独立修订审阅](reviews/automatic-pipeline-root-design-review.md)改为 D/E/X，依靠独立已知前端域、每线程 E、attempts/emitted 和完整元素集合校验，不从日志反推线程域。active+inactive 多桩及仅 inactive 均通过；整线程结束记录删除必须失败。

用户进一步提出“是否应该要求选线程”，接口随之拆分：普通整块采集只选逻辑对象/region，自动安排 reader；`point.thread` 只筛 scalar/local/元素执行，`collective.reader` 是高级可选输出者。两者都使用源码前端坐标，不宣称硬件线程过滤。whole+thread、direct+collective、reader 越 CTA/组必须明确拒绝。

## 固定安装产物

最终验收 wheel：`tilelang_debugger-0.2.0-py3-none-any.whl`。

SHA256：`c0d703825a43e158022c790e4f2c3db04acfe69764e24a8b28185071ab588456`。

独立安装及证据根目录：

```text
/home/ang.gao/tilelang-debugger-unified-reader-final-20261009/
  installed/
  artifacts/
```

完整证据归档为 `tilelang-debugger-unified-reader-final-20261009-evidence.tar.gz`，SHA256 `a0ddb8433fa2f0ce956cabcb1b99c697cebb23d75e8e31a0bba72b98ca03e71c`；本地副本为 `artifacts/unified-final-evidence.tar.gz`，不提交大型运行产物到 Git。归档含实际 wheel、测例/脚本、原始 capture、独立复算与扰动检查结果。

验证不从开发 `src` 导入。每个 schema 3 GPU worker 的 `environment.json` 记录实际 debugger 路径；独立 reviewer 已核对 wheel 的 48 个 Python 文件与本地源码、installed 文件逐字节一致。旧候选 `ac3038...` 和 `f42e16...` 保留原始证据，不当作本轮最终安装包验收。

环境：H200、TileLang 0.1.12 / `2d63708c8ad57196051c4636a1167c5453c73a48`、PyTorch 2.10.0+cu129、CUDA 13.2.78 / driver 595.71.05、Compute Sanitizer 2026.1.1.0。TileOPs 固定 `95ba6cae` checkout。

## 验收矩阵

下表的最终安装包测试脚本均已完成，全部符合预期。共 116 次 schema 3 采集尝试：102 次完整采集、4 次预期 partial、10 次预期拒绝（8 次配置错误、2 次越界观察）；另有 7 次旧 reviewed 回归。预期 partial/拒绝没有计为完整采集通过。独立复核结果见验收报告。

独立 session 从原始证据重新验证并重新执行 reference：106 份有效 schema 3 capture，共 130 次 launch、9,976 条 DATA；旧 7 例另有 71,172 条 DATA。99 项 CPU/TIR、25 项协议/证据扰动检查、6 个 reader 语义组合、9 项快照/alias roundtrip 均通过。配置拒绝核对了精确错误原因及零 instrumented launch，没有仅凭退出码判定。

| 组 | 测例 | 检查 |
| --- | --- | --- |
| core | 11 条路径，各 2 次 launch | Parallel 尾部、Pipelined、Group、二维 CTA/local、动态边界、while+transfer、负步长、fragment/shared/global 选区、in-place 重建 |
| advanced | 17 条路径，含预期拒绝/partial | 组内 fragment、pipeline fragment、嵌套动态循环/分支、重复值、末尾 return、helper/alias、多种 reader、选线程/迭代、未到达、scalar/collective 截断、越界观察及配置拒绝 |
| types | 13 dtype × 4 scope = 52 | scalar/local/fragment/shared 原始位模式和独立数值 reference；int64/uint64 超过 2**53、FP64 可区分精度 |
| pipeline-copy | mixed、only-inactive、mixed synccheck | 真实 global→shared→fragment 三 stage TMA 流水线、自动扩大 CTA、active/inactive 多桩 |
| TileOPs | 5 | MaxPool、带索引 MaxPool、RoPE、Softmax、RMSNorm 的真实上游源码 |
| 旧 reviewed | GELU、Sum padded/unpadded、GEMM、GQA | racecheck + 中间 tile/最终输出 reference；GEMM/GQA 另跑 synccheck |
| CPU/TIR | 13 suites，99 tests | 已全部通过，无 skip；旧 80 项加统一路径 19 项 |

core/advanced 分别运行 racecheck 与 synccheck；TileOPs 运行 racecheck；dtype 矩阵核对编码和数值，不声称 52 个组合都各自跑过 sanitizer。每份普通成功 capture 都重新运行独立 reference，输入/调用后参数/返回值同时核对。

`advanced` 的 scalar 截断预算为 96（32 E + 每线程两条 D），collective 截断预算为 64（32 E + 一次 32 元素完整 tile）。原程序仍完成所有迭代与同步，`run` 必须退出 3，reference 分析退出 2；不能把完整前缀提升为全采集通过。OOB 观察跳过额外非法读取，X 缺失时 E 的持久错误位仍使采集失败。

## 重现命令

先在准备好 GPU 依赖的环境中将 wheel 安装到独立目录；设 `PYTHONPATH` 为该目录。以下命令从包含 examples/tests 的仓库根执行，输出目录须不存在：

```bash
python tests/run_cpu.py --output artifacts/cpu --access-fixture /path/to/verified/gqa-trace
python tests/validate_unified.py --output artifacts/core --sanitizer racecheck
python tests/validate_unified.py --output artifacts/core-sync --sanitizer synccheck
python tests/validate_unified_advanced.py --output artifacts/advanced --sanitizer racecheck
python tests/validate_unified_advanced.py --output artifacts/advanced-sync --sanitizer synccheck
python tests/validate_unified_advanced.py --types --scope scalar --scope local --output artifacts/types-direct
python tests/validate_unified_advanced.py --types --scope fragment --scope shared --output artifacts/types-collective
python tests/validate_unified_pipeline_copy.py --output artifacts/pipeline-mixed --mode mixed --sanitizer racecheck
python tests/validate_unified_pipeline_copy.py --output artifacts/pipeline-inactive --mode inactive --sanitizer racecheck
python tests/validate_unified_pipeline_copy.py --output artifacts/pipeline-sync --mode mixed --sanitizer synccheck
python tests/validate_unified_tileops.py --tileops /path/to/TileOPs --output artifacts/tileops --sanitizer racecheck
python tests/validate_gpu.py --output artifacts/reviewed --sanitizer racecheck
python tests/validate_gpu.py --output artifacts/reviewed-sync --sanitizer synccheck --cases gemm gqa
```

独立审计另外直接读取实际 stdout 和元数据，打乱日志应通过；删数据、删整次访问、删整线程/结束记录、重复、截断、越域、篡改计数/host boundary/快照/持久记录/CUDA/scalar/compile identity 必须失败。完整性负例与 numerical mismatch 区分，不以 sanitizer 成功代替数值 reference。

## 验证边界

- 自动 num_stages 通过；手写 Pipelined order/stage/sync/group 数组明确拒绝，未宣称支持手工 schedule 改写。
- 末尾 bare return 通过；条件/嵌套/value return 在固定 eager 前端控制构建，不是 GPU 动态 return，因此明确诊断。
- 独立 snapshot roundtrip 覆盖零维 CUDA tensor、重叠非零 offset views、alias-preserving deepcopy、负零、NaN payload、64 位整数及 Python scalar。它验证快照 API；固定 TileLang 基线无法编译零维参数，不能将其称为零维 kernel 集成通过。
- collective.ready/uniform 和 named barrier 保留是显式用户契约；工具不会从一次成功运行证明任意异步就绪、任意线程布局或跨 block 一致性。
- Parallel 的具体尾部/线程对应已用独立 fixture reference 验证；协议保证已声明采集事件完整，不等于证明任意程序的算法执行覆盖。
- 原样例 GQA 本身使用已记录的输出同步修正版，见[样例说明](../examples/README.md)；本轮没有悄悄改变该计算。
