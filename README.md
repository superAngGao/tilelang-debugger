# TileLang Debugger

面向 NVIDIA H200 的 TileLang 定点采集原型。用户在**原始源码**中选择行号、buffer、block 和循环的第几次迭代；工具在临时副本插入 monitor，保存完整 tile 的原始位模式及两份实际编译产物。

当前接受范围是经过审阅的 GELU、Sum、GEMM、GQA 四个样例及其固定构建参数。源码和运行脚本用哈希限定；改动后的 kernel 需要新契约，不会因为 AST 能解析就自动运行。此版本尚不支持任意用户 kernel、直接 shared 采集、`T.Pipelined`/`T.unroll` 内选点、多 launch、reference 产品接口或访问诊断。

## 使用

GPU worker 需要 Linux、NVIDIA H200 和已验证的 TileLang 0.1.12 环境。核查的源码 checkout 为 `2d63708c8ad57196051c4636a1167c5453c73a48`；还会检查打印及同步 helper 摘要。普通 `pip install tilelang==0.1.12` 不保证与该 checkout 相同。CPU 源码/记录测试也可在 Windows 运行。

```bash
pip install -e .
CUDA_VISIBLE_DEVICES=0 python -m tilelang_debugger run examples/gelu/run.py \
  --monitor examples/gelu/monitor.json --output artifacts/gelu-001
```

输出目录必须不存在。每次运行先编译并执行 baseline，再编译插桩版本，通过同步/索引门禁后执行一次；实际输入和最终输出必须逐位一致。原样例不被覆盖。`run.py` 是可信用户代码，worker 不是安全沙箱。

`monitor.json` 示例（行号以仓库中的原文件为准）：

```json
{
  "source": "kernel.py",
  "points": [{
    "id": "gelu_input",
    "line": 87,
    "when": "after",
    "buffer": "x_reg",
    "block": [1, 0, 0],
    "loops": []
  }]
}
```

多行语句按开始行选点。`when` 是 `before` 或 `after`；循环配置为 `{"line": 原循环开始行, "iteration": 1起的序号}`，从外到内逐层选择。可在已审阅观察位置选择合法 block/迭代；GEMM 示例观察整个 ring dispatch 之后的第二次 K 迭代。各样例附带可直接运行的配置。

## 已实现的采集链路

- 源 AST 定位、词法作用域内唯一 buffer 身份、临时源码插桩和原始位置映射。
- 完整 fragment；float16、bfloat16、float32、int32 的原始位模式。每条记录一次 printf 输出身份、index 和 bits，浮点不经 `%f` 转换。
- 独占 named barrier、显式参与人数、staging 前后顺序检查；异步 accumulator 使用已有 operand-fence intrinsic，不新增 lowering pass。
- launch 前检查实际 device IR/CUDA、源/目标索引覆盖、打印读索引、原同步协议及 helper 资源；无法解释时拒绝。
- 严格识别重复、缺失、截断和错误身份；最多 65,536 个元素，设备 printf FIFO 至少 64 MiB，仍以实际完整性检查为准。
- 从实际编译对象导出 baseline 和 instrumented 的前端 IR、最终 device IR、CUDA 与有效编译配置；禁用编译缓存，避免用另一次 lowering 代替运行产物。

每次输出包含 `run.json`、`points.json`、`records.jsonl`，以及两版源码、IR/CUDA、编译配置、门禁结果、stdout/stderr、进程状态和输入/输出原始字节。`records.jsonl` 的 `bits` 是规范值；`dtype`、`shape` 和源码身份在 `points.json` 中，不能按日志出现顺序拼 tile。

## 四个样例

| 样例 | 观察位置 | 验证方式 |
| --- | --- | --- |
| GELU | 输入 x_reg、计算完的 y_reg | 与输入/输出 tile 逐位核对 |
| Sum | 完整 x_f32、归约 acc | padding/non-padding 两条路径及独立求和 |
| GEMM | consumer 第二轮 K 的 c_local | 前两轮 K 范围 matmul 与完整 GEMM reference |
| GQA | 两个 consumer 的 QK prologue acc_s | 两份 QK reference 与完整 attention reference |

来源、许可证及 GQA 必要修正见 [examples/README.md](examples/README.md)。GQA 原样例关闭自动同步，输出阶段缺少 shared→TMA 的组内发布顺序；原件保留为 `kernel.original.py`。正式示例仅增加每组 proxy fence 与 128 线程 barrier，基线和插桩都使用修正版。不能将原样例标为已通过。

monitor 会改变 shared memory、寄存器与同步开销，采集结果不能用作未插桩程序的性能数据。

## 验证与审阅

```bash
# 无 GPU 依赖
PYTHONPATH=src python -m unittest discover -s tests -p test_cpu.py -v
PYTHONPATH=src python -m unittest discover -s tests -p test_evidence.py -v
# 需要真实 TileLang/TVM，测试本身不 launch GPU
PYTHONPATH=src python -m unittest discover -s tests -p test_ir.py -v
# H200：完整 kernel 与中间 tile，同时检查两版 racecheck
python tests/validate_gpu.py --output artifacts/acceptance --sanitizer racecheck
python tests/validate_gpu.py --output artifacts/sync --sanitizer synccheck --cases gemm gqa
python tests/fidelity.py --output artifacts/bits --sanitizer racecheck
python tests/validate_print_probe.py --output artifacts/print-regression
```

验收记录见 [docs/phase1-validation.md](docs/phase1-validation.md)，方案及原始独立审阅在 [docs/phase1-plan.md](docs/phase1-plan.md) 与 [docs/reviews](docs/reviews)。设计通过与代码/GPU验收分别记录。

后续阶段：用户 reference 接入与数值分析 → 访问索引/边界与布局观察 → 分层采集和报告。reference 不自动为任意 kernel 生成。其他芯片由合作项目组负责。
