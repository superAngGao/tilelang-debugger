# 真实 TileOPs kernel 测试

这里直接加载指定 TileOPs checkout 的源码，不保存改写后的 `kernel.py`。首批包括 Softmax、RMSNorm、RoPE，共 21 个 shape/path/dtype 组合。测试版本与参数见 [manifest.json](manifest.json)，版本记录不用于用户 kernel 准入。

**当前这些是数值基线和调试接入测试。现有 debugger 的 `run` / `trace` 尚不能加载这些外部模块；这里的基线通过不表示已取得中间 tile 或 runtime 访问记录。**

2026-10-09 H200 实测：21 组基线、9 次 sanitizer 检查通过；6 次调试入口探测为 unsupported，严格验收非零退出。详见[验证记录](../../docs/tileops-integration-validation.md)。

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

`monitor.json` / `access.json` 的 `tileops-observation-intent-v1` 是**测试观察点意图**，不是现有 CLI 可执行配置。运行时按 factory、Python 分支及完整 AST 语句唯一定位，生成 `observations.json`，保留当前真实行号与原循环上下文。歧义或源码语句变化会明确失败；空行不会影响定位。循环次数从 1 起。源码锚点需要修改时，修改测试配置即可。

## 产物与验收含义

单例目录包含：

- `provenance.json`：实际 checkout、Git commit/状态、所有导入模块摘要、driver/reference 摘要。
- `environment.json`、`compile.json`：环境及实际编译配置。
- `frontend.py/json`、`device.py/json`、`kernel.cu`：本次**未插桩**程序的真实编译产物。
- `tensors.pt`：CPU 输入副本、实际完整输出、reference；仅加载自己信任的运行产物。
- `comparison.json`、`elements.jsonl`：数值汇总及逐元素比较；`result.json` 保留 padding 检查。
- `observations.json`：计划源码观察点，标记 `planned_not_captured`。

矩阵还保存每个独立进程的命令、stdout/stderr、退出码、超时，以及每类 `run` 和 `trace` 的实际 CLI 接入探测。仅已确认的缺失 sibling `kernel.py` 入口限制分类为 `unsupported`；其他错误分类为 `failed`。`--sanitizers` 为每类一例追加 memcheck、racecheck、synccheck，缺少工具或错误摘要不算通过。

默认退出 0 表示所选数值基线通过且接入探测没有未解释的故障；`summary.json` 仍会明确保留 `debugger_status=unsupported`、`delivery_passed=false`。完整调试验收必须加 `--require-debugger`，当前预期退出非零，不能将该缺口包装成 PASS。

下一步通用入口的改动与验收路线见 [方案](../../docs/tileops-integration-plan.md)。源码/driver 哈希、固定 shape、固定索引公式不应成为新增算子的产品分支。
