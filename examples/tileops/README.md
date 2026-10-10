# 真实 TileOPs kernel 测试

这里直接加载指定 TileOPs checkout 的源码，不保存改写后的 `kernel.py`。首批包括 Softmax、RMSNorm、RoPE，共 21 个 shape/path/dtype 组合。测试版本与参数见 [manifest.json](manifest.json)，版本记录不用于用户 kernel 准入。

**数值采集使用默认 source 引擎，不受固定源码/driver 契约限制。** `capture.py` 直接导入上游源码，按当前行号插入 Python 打印宏，再运行独立 CPU reference 分析。Softmax/RMSNorm 采集中间 fragment；RoPE 仅采输入 x。外部 kernel 的 runtime 访问索引和 RoPE 中间 scalar 尚未实现。

此前 21 组基线、9 次 sanitizer、6 次入口拒绝属于历史记录，见[原验证记录](../../docs/tileops-integration-validation.md)。新源码引擎单独验收，不能用历史 baseline 代替。

新源码引擎 21 组基础采集及 6 组 sanitizer 采集均通过选点/输出 reference 比较，见[本轮验证记录](../../docs/generic-kernel-validation.md)。

```bash
python examples/tileops/capture.py --tileops /path/to/TileOPs \
  --case softmax-tiled-float32 --output artifacts/softmax-capture --sanitizer racecheck
python tests/validate_source_engine.py --tileops /path/to/TileOPs \
  --output artifacts/source-matrix --sanitizers
```

新单例产物含实际 CLI 配置 `monitor.json`、`capture/` 的两版编译与完整记录、`analysis/report.md` 的中间点/输出比较。无数值匹配时仍生成分析报告并退出 2；执行异常或记录不完整失败。fixture `capture_reference.py` 按真实输入提供 reference，不属于产品自动推导。

## 运行

在具有 torch、TileLang 及 TileOPs 所需依赖的 Linux/H200 环境，从本仓库根目录执行：

```bash
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=2
python examples/tileops/softmax/run.py \
  --tileops /path/to/TileOPs --case softmax-tail-float16 \
  --output artifacts/tileops/softmax-tail-001

python tests/validate_tileops.py --tileops /path/to/TileOPs \
  --output artifacts/tileops/matrix-001 --sanitizers
```

输出目录必须不存在，避免覆盖证据。`--case` 可从 manifest 的 case id 加上 `-float16`、`-bfloat16` 或 `-float32` 得到；验证脚本支持重复 `--case` 缩小矩阵。

测试使用普通 Python 包导入，核对所有已导入 TileOPs 模块的真实文件路径。如果环境加载了另一个已安装版本，则失败。驱动直接调用上游 kernel factory，覆盖 kernel 数学行为；不测试完整 Op 的 dispatch、autotune 或多 kernel 调度。RMSNorm 的驱动按上游 forward 约定给 x/weight 补零。

`reference.py` 是这些基线驱动使用的 CPU 数学 reference，签名为 `reference(inputs, case)`，不是现有 `analyze` 命令的 provider。使用实际输入值，比较完整输出；Softmax/RMSNorm 的 padding 另以零容差检查。未采集的中间状态不会伪造为实际结果。

## 目录及观察点

| 目录 | 运行路径 | 计划观察位置 |
| --- | --- | --- |
| `softmax/` | N=256、257；N=513/tile_n=256；FP16/BF16/FP32 | row_max、row_sum、第三轮 tile_sum、输入索引/掩码 |
| `rms_norm/` | N=256、257；FP16/BF16/FP32 | sumsq、rrms、输出 fragment、weight 广播访问 |
| `rope/` | 32×64、16×128；FP16/BF16/FP32 | x/cos/sin/配对元素读取、y 写入；原 kernel 无 fragment |

例子中的 `monitor.json` / `access.json` 是**测试观察点意图**，不是直接传给 CLI 的配置。`capture.py` 按 factory、Python 分支及 AST 语句生成实际 `source/points` 配置；工具本身只使用用户行号，不按例子匹配。歧义或源码语句变化需更新 fixture 锚点；空行不影响定位。循环次数从 1 起。`access.json` 仅保存上游源码位置意图，也供基线与数值观察点定位复用，不是已移除的旧 trace 配置或可执行访问采集入口。

## 历史基线与 reviewed 入口探测

单例目录包含：

- `provenance.json`：实际 checkout、Git commit/状态、所有导入模块摘要、driver/reference 摘要。
- `environment.json`、`compile.json`：环境及实际编译配置。
- `frontend.py/json`、`device.py/json`、`kernel.cu`：本次**未插桩**程序的真实编译产物。
- `tensors.pt`：CPU 输入副本、实际完整输出、reference；仅加载自己信任的运行产物。
- `comparison.json`、`elements.jsonl`：数值汇总及逐元素比较；`result.json` 保留 padding 检查。
- `observations.json`：计划源码观察点，标记 `planned_not_captured`。

旧脚本 `validate_tileops.py` 保留基线和历史契约探测，数值探测显式指定 `--engine reviewed`；其 unsupported 不代表新 source 引擎不支持。新引擎验收请使用上方 `validate_source_engine.py`。旧 RoPE y 请求仅是契约准入负例，不代表支持全局输出快照。

`--probes-only` 可只复测入口，不重跑已通过的基线；因没有基线证据，该模式不会通过完整验收，当前返回非零。`--sanitizers` 为每类一例追加 memcheck、racecheck、synccheck，缺少工具或错误摘要不算通过。

默认退出 0 表示所选数值基线通过且接入探测没有未解释的故障；`summary.json` 仍会明确保留 `debugger_status=unsupported`、`delivery_passed=false`。完整调试验收必须加 `--require-debugger`，当前预期退出非零，不能将该缺口包装成 PASS。

下一步通用入口的改动与验收路线见 [方案](../../docs/tileops-integration-plan.md)。源码/driver 哈希、固定 shape、固定索引公式不应成为新增算子的产品分支。
