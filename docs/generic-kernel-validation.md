# 源码打印引擎验证记录

> 历史记录：旧 `trace` 已从当前产品移除；文中 trace 命令、专用测试和兼容性描述仅适用于[移除前提交](https://github.com/superAngGao/tilelang-debugger/tree/e838fc88d96a562ee6230ee154d6593bac6632f8)。保留当时结果，不代表当前产品能力。

日期：2026-10-09。设计复审 v3、代码复审 v2 通过后执行本轮验收。数值采集直接在源码插入已有 Python 打印/同步宏；没有新增 lowering pass、layout 查询、寄存器映射或 device IR 插桩。device IR/CUDA 仅导出。

## 实际采集矩阵

H200 / TileLang 0.1.12，环境与既有验证一致。TileOPs checkout `95ba6cae78856ec2f610535fde3d44df198d39e8`。原包直接导入，用户源码未改动；算子配置及 reference 仅属于 examples。

`release-source/summary.json`：**27/27 采集及独立数值分析通过**，包括 21 组 shape/path/dtype 基础组合和 6 组 sanitizer 组合。每组分别运行 baseline/instrumented，共 54 worker；12 个指定 sanitizer 的 worker 都有完成且零错误的原始日志。

| 样例 | 实际采集 | 每次记录数 | 基础组合 |
| --- | --- | ---: | ---: |
| Softmax N=256、257 | row_max、row_sum | 2 | 6 |
| Softmax N=513/tile_n=256 | 原 serial 第 3 次的 tile_sum | 1 | 3 |
| RMSNorm N=256 | sumsq、rrms、normalized fragment | 258 | 3 |
| RMSNorm N=257 | sumsq、rrms、normalized fragment（含 padding） | 514 | 3 |
| RoPE 32×64、16×128 | 只读输入 x | 2048 | 6 |

全部选择 block `(1,0,0)`，覆盖 FP16/BF16/FP32。每次检查两版输入和最终输出逐位一致、打印元素无缺失/重复/截断，再用 CPU reference 比较选点和输出。Softmax 多 tile 观察值按前 3 tile 的 running max 计算，不用最终结果替代中间 reference。

sanitizer 组合为 Softmax tiled FP32、RMSNorm N257 FP16、RoPE 32×64 BF16，各跑 racecheck 和 synccheck。RoPE 输入打印不代表中间 scalar 或 runtime 访问索引已实现；外部 trace 不在本轮完成范围内。

## 通用接入和失败路径

`release-source-edges` 九项通过：

- 普通包内相对导入；改 buffer 名并添加空行；source 与 driver 同文件。
- 无 reference 的 driver 完成采集，状态为 not_checked；后续用户 reference 分析匹配。
- 合法错误计算仍完整采集，独立 reference 报 mismatch 并退出 2。
- driver 异常、未加载选定源码、线程相关循环上界、越界迭代、global 输入共享 storage 均失败。线程相关上界和越界迭代在插桩 launch 前拒绝。
- 失败时编译/导入环境恢复，原源码内容不变。源码别名/不透明 global 使用的拒绝另有 CPU 测试。

`release-source-mismatch` 在独立 TileOPs 副本中将 RMSNorm 平方和每元素加 1：run 仍采集完整 514 条记录，numerical_status=failed；示例继续生成 completed/matched=false 的离线分析，最终退出 2。不是更改白名单后放行，也不是把执行异常当成数值不匹配。

`release-source-reviewed`：原 GELU/GQA 受审数值路径、CPU tile/reference 复算、两版输入输出位一致均通过；四个 worker 的 racecheck 零错误。旧异步/分组同步功能未被新路径替换。

## CPU 与打包

`release-source-cpu`：10 个隔离 suite、64 项全部通过、无跳过，包括实际 TVM IR 测试及 GQA access fixture。analysis 必须在未导入 TileLang 的独立进程中运行；一次合并 discovery 因其他 suite 先导入 TileLang 而触发其断言，原日志保留为 combined-discovery.log，之后按仓库既有隔离 runner 完整执行通过。Windows 无 torch/TileLang，不作为完整 CPU/TIR 验收环境。

wheel 构建通过，包内 21 个产品文件逐字节匹配源码。wheel SHA256：`eeb0c460e99e31eadfa21effa830f2aab540ff8f1c8e61bdc73a0e620aa4e540`。未发布包索引。

## 重现命令及证据

```bash
export PYTHONPATH="$PWD/src"
python tests/validate_source_engine.py --tileops /path/to/TileOPs \
  --output artifacts/source-matrix --sanitizers
python tests/validate_source_edges.py --output artifacts/source-edges
python tests/validate_tileops_mismatch.py --tileops /path/to/TileOPs \
  --output artifacts/source-mismatch
python tests/validate_gpu.py --output artifacts/reviewed-regression \
  --cases gelu gqa --sanitizer racecheck
python tests/run_cpu.py --output artifacts/source-cpu \
  --access-fixture /path/to/verified/gqa-access
```

原始日志、配置、source snapshots、IR/CUDA、tensor bytes、记录、分析和负例保存在忽略提交的 `artifacts/source-engine/`。归档 `artifacts/source-engine-evidence.tar.gz` SHA256：`8dd3f1126113ae5658b68d8185fe035f596a510b4dd28a01efb48220bed1aa91`。

审阅记录：[设计复审 PASS](reviews/generic-kernel-design-review-v3.md)、[代码复审 PASS](reviews/generic-kernel-code-review-v2.md)、[最终独立验收 PASS](reviews/generic-kernel-acceptance-review.md)。验收 session 另写 CPU 公式复算了全部 49 个观察点、19,745 条矩阵记录及最终输出，并复核错误计算、失败路径和旧样例。
