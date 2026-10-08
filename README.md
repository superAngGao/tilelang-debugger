# TileLang Debugger

面向 **NVIDIA H200** 的 TileLang 源码定点调试原型：选择源码行、buffer、block 和循环迭代，采集该位置的完整 tile，并保留插桩前后的 IR 与 CUDA，帮助核对中间计算结果。

**当前状态：阶段 1 已实现并通过独立代码审阅及 H200 验收。** 支持范围是本仓库 GELU、Sum、GEMM、GQA 四个样例的固定构建参数和已审阅观察位置。当前以命令行和文件产物为主，尚未支持任意用户 kernel 或交互式调试界面。

## 现在能看到什么

以下是仓库自带 `monitor.json` 已经实际验证的采集内容，均选择 block `(1, 0, 0)`：

| 样例 | 观察位置与数据 | 用途 | 记录数 |
| --- | --- | --- | ---: |
| [GELU](examples/gelu) | 加载后的 `x_reg` 和计算后的 `y_reg`，各 2048 个 BF16 元素 | 核对输入及逐元素 GELU 结果 | 4096 |
| [Sum](examples/sum)，N=257 | 归约前 `x_f32[2,512]`，含补零部分；归约后 `acc[2]`，均为 FP32 | 检查加载、padding 和每行求和 | 1026 |
| Sum，N=256 | 无 padding 路径的 `x_f32[2,256]` 和 `acc[2]` | 对照无 padding 时的加载与归约 | 514 |
| [GEMM](examples/gemm) | 第 2 次 K 循环完成后的 `c_local[128,128]`，FP32 | 检查前 128 个 K 元素贡献的部分累加结果 | 16384 |
| [GQA](examples/gqa) | 两个 consumer 各自首次 QK 计算完成后的 `acc_s[64,128]`，FP32 | 分别核对两组 QK 分数；当前未采集 softmax/PV 中间值 | 16384 |

每个元素都有观察点、launch、block、循环变量值、逻辑索引和原始位模式；观察点另附源码行、buffer 名、shape 和 dtype。按这些身份重建 tile，不依赖设备日志的出现顺序。重复、缺失、截断和身份不匹配都会判为失败。

采集支持 `float16`、`bfloat16`、`float32`、`int32`，以原始位模式保存，避免 `%f` 文本转换损失精度。数值展示时可以解码；需要核对 NaN payload 等细节时，应直接比较 `bits`。

## 环境与安装

GPU 运行需要 **Linux + NVIDIA H200 + 已核查的 TileLang 环境**。当前验证环境为：

| 组件 | 已验证版本 |
| --- | --- |
| TileLang | `0.1.12`，源码 checkout `2d63708c8ad57196051c4636a1167c5453c73a48` |
| PyTorch | `2.10.0+cu129` |
| CUDA 编译器 / 驱动 | `13.2.78` / `595.71.05` |
| Compute Sanitizer | `2026.1.1.0`，运行同步检查时使用 |

工具会核对 TileLang 打印和同步 helper 的摘要。仅执行 `pip install tilelang==0.1.12` 不保证得到相同实现；本包也不会自动安装或替换 TileLang、PyTorch、CUDA。请先准备上述已验证环境，再安装本项目：

```bash
git clone https://github.com/superAngGao/tilelang-debugger.git
cd tilelang-debugger
python -m pip install -e .
```

以下命令均从仓库根目录执行。无 GPU 依赖的源码和记录测试也可在 Windows 运行。

## 运行一次采集

```bash
CUDA_VISIBLE_DEVICES=0 python -m tilelang_debugger run examples/gelu/run.py \
  --monitor examples/gelu/monitor.json \
  --output artifacts/gelu-001
```

`--output` 必须指向尚不存在的目录。每次运行依次在独立进程中编译并执行 baseline 和 instrumented 两版，分别只 launch 一次。工具在插桩版本 launch 前检查实际生成代码；运行后要求两版输入、最终输出逐位一致，并验证采集记录完整性。样例驱动还会检查最终输出的 reference。

替换驱动及配置即可运行其他样例：

| 样例 | 驱动 | 采集配置 |
| --- | --- | --- |
| GELU | `examples/gelu/run.py` | `examples/gelu/monitor.json` |
| Sum，N=257 | `examples/sum/run.py` | `examples/sum/monitor.json` |
| Sum，N=256 | `examples/sum/run_unpadded.py` | `examples/sum/monitor.json` |
| GEMM | `examples/gemm/run.py` | `examples/gemm/monitor.json` |
| GQA | `examples/gqa/run.py` | `examples/gqa/monitor.json` |

需要 Compute Sanitizer 时附加 `--sanitizer racecheck` 或 `--sanitizer synccheck`。可用 `--timeout 600` 调整单个 worker 的超时秒数，默认 240 秒。驱动是本地可信 Python 代码，worker 不是安全沙箱。

## 如何选择源码位置

配置中的行号指向**原始 `kernel.py`**，多行语句使用开始行。例如当前 GEMM 配置：

```json
{
  "source": "kernel.py",
  "points": [{
    "id": "gemm_0",
    "line": 116,
    "when": "after",
    "buffer": "c_local",
    "block": [1, 0, 0],
    "loops": [{"line": 113, "iteration": 2}]
  }]
}
```

含义是：在原始第 113 行循环的**第 2 次执行**中，于第 116 行开始的语句执行完后，采集 block `(1,0,0)` 的 `c_local`。此处观察的是整个 ring dispatch 完成后的累加器。

- `when`：语句的 `before` 或 `after`，具体取值必须符合该位置的受审配置。
- `iteration`：从 **1** 开始的执行序号；输出记录中的 `loops` 保存实际循环变量值，因此这里是 `[1]`，即 `ki=1`。
- `block`：从 **0** 开始的三维 block 坐标。
- `buffer`：原始源码中的变量名，工具在临时副本中处理作用域和重命名。

当前可在已审阅位置选择合法 block 和受支持的循环迭代。**不能仅修改行号就采集任意位置**：源码、驱动、观察位置、参与线程与布局受契约约束，无法匹配时会拒绝执行。修改 kernel 或扩展观察位置需要补充验证和审阅。

## 输出文件怎么读

成功运行的主要产物如下；`baseline/` 和 `instrumented/` 各自保存对应版本的编译与执行信息：

```text
artifacts/gelu-001/
├── run.json                  # 本次运行身份、状态、记录数和两版一致性
├── monitor.json              # 用户选择的观察点配置
├── points.json               # 观察点的源码身份、shape、dtype、循环值等
├── records.jsonl             # 每个元素一条记录，bits 为规范值
├── baseline/
└── instrumented/
    ├── source/               # 本次实际使用的源码副本
    ├── frontend.py           # 前端 IR
    ├── device.py             # 最终 device IR
    ├── kernel.cu             # 实际生成的 CUDA
    ├── compile.json          # 编译配置及提取的同步/分配信息
    ├── environment.json      # 软件与设备环境
    ├── launch-gate.json      # launch 前检查结果
    ├── inputs.json / outputs.json
    ├── inputs-*.bin / outputs-*.bin
    ├── reference.json        # 样例最终输出的 reference 检查
    ├── execution.json / process.json
    └── stdout.log / stderr.log
```

指定 sanitizer 后，还会保存对应检查日志。失败时保留已产生的文件和失败状态，不应将部分数据当作完整 tile。

例如 GEMM 的一条真实记录：

```json
{"schema":1,"point":"gemm_0","launch":0,"block":[1,0,0],"loops":[1],"index":0,"bits":3229736808,"dtype":"float32"}
```

结合 `points.json`，它表示第 2 次 K 循环后的 `c_local[0,0]`。`index` 是 tile 的行优先展平索引，不是线程号或物理地址。run 身份保存在同目录的 `run.json` 中。

下面的 CPU 示例读取一次 GELU 采集的前 8 个元素，无需 GPU：

```python
import json
from pathlib import Path
from tilelang_debugger.records import value

folder = Path("artifacts/gelu-001")
run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
assert run["status"] == "passed", run
points = {
    p["id"]: p
    for p in json.loads((folder / "points.json").read_text(encoding="utf-8"))
}
with (folder / "records.jsonl").open(encoding="utf-8") as stream:
    for _, line in zip(range(8), stream):
        record = json.loads(line)
        point = points[record["point"]]
        print(point["buffer"], record["index"],
              value(record["bits"], record["dtype"]))
```

当前尚无通用数值分析 UI、逐元素误差报告或用户 reference 接入接口。四个样例的中间值 reference 由下面的验收脚本提供，普通 `run` 不会额外生成 `validation.json`。

## 数值与同步验证

完整验收会同时检查中间 tile、最终输出、两版输入/输出逐位一致性和 sanitizer 结果：

```bash
# 四类样例、五条路径；包含中间 tile reference
CUDA_VISIBLE_DEVICES=0 python tests/validate_gpu.py \
  --output artifacts/acceptance --sanitizer racecheck

# GEMM / GQA 的同步检查
CUDA_VISIBLE_DEVICES=0 python tests/validate_gpu.py \
  --output artifacts/sync --sanitizer synccheck --cases gemm gqa

# 四种 dtype 的原始位模式保真
CUDA_VISIBLE_DEVICES=0 python tests/fidelity.py \
  --output artifacts/bits --sanitizer racecheck
```

`validate_gpu.py` 在每个 case 下额外写入 `validation.json`，记录中间 tile 和最终输出的 reference 误差与是否通过，并在根目录生成 `summary.json`。reference 针对这些已知样例编写，不自动为任意程序生成。

2026-10-08 的 H200 验收结果：五条路径全部通过，baseline/instrumented 共十次 racecheck 均为零错误/警告；GEMM/GQA 共四次 synccheck 为零错误；四种 dtype 的位模式测试通过。详细误差、容差和实验边界见 [验收记录](docs/phase1-validation.md)。

GQA 使用了局部同步修正版：原样例的输出阶段缺少 shared 写入到 leader TMA store 之间的组内发布顺序，正式样例为两个 consumer 各增加 proxy fence 与组内 barrier。原件保留在 [kernel.original.py](examples/gqa/kernel.original.py)。六次隔离实验中，原件三次数值失败、修正版三次通过，但六次 racecheck 都是零 hazards，因此不能只凭 sanitizer 判断数值正确。来源、许可证和修改细节见 [样例说明](examples/README.md)。

## 实现方式与当前边界

```text
原始源码 + 选点配置
  → AST 定位，在临时副本插入采集宏
  → 分别编译 baseline / instrumented，导出实际 IR 和 CUDA
  → 检查插桩版本的同步、参与线程、元素映射与打印读取
  → 运行并收集 printf 原始位模式
  → 校验记录完整性、两版输入/输出一致性，保存产物
```

采集宏沿用 TileLang 已有的 fragment→shared 搬运和 printf 机制，在内部搬运后及打印后安排组内同步；异步 accumulator 使用已有 operand-fence intrinsic。**没有新增 lowering pass 或指令，也没有修改安装的 TileLang。** 原始源码不被覆盖，编译缓存被禁用，导出的是实际编译对象的产物。

当前限制：

- 仅支持经过审阅的四个样例、固定构建参数、观察位置与布局；未知同步协议或布局会在 launch 前拒绝。
- 采集对象为完整 fragment，暂不支持直接 shared buffer 采集、`T.Pipelined` / `T.unroll` 内选点或多 launch。
- 一次采集最多 65,536 个元素，设备 printf FIFO 至少 64 MiB；仍以实际记录完整性检查为准。
- monitor 会增加 shared memory、寄存器和同步开销，插桩程序不能代表原程序性能。
- 内存访问索引/边界诊断、通用 reference 接入、自动根因诊断和交互式报告尚未实现。

## 开发与审阅记录

```text
src/tilelang_debugger/   源码定位、采集宏、IR 检查、运行与记录解析
examples/               GELU、Sum、GEMM、GQA 及选点配置
tests/                  CPU/TIR 测试、GPU 验收与保真测试
experiments/            T.print 同步、GQA 输出同步的隔离实验
docs/                   实施方案、验证结果与独立审阅记录
artifacts/              本地运行产物，不提交到 Git
```

安装本项目后可运行：

```bash
# 不需要 GPU 或 TileLang
python -m unittest discover -s tests -p test_cpu.py -v
python -m unittest discover -s tests -p test_evidence.py -v

# 需要真实 TileLang/TVM，但测试本身不 launch GPU
python -m unittest discover -s tests -p test_ir.py -v

# H200 上重现 T.print 同步的八个正负对照
CUDA_VISIBLE_DEVICES=0 python tests/validate_print_probe.py \
  --output artifacts/print-regression
```

- [阶段 1 方案](docs/phase1-plan.md)
- [第三轮独立代码复审：PASS](docs/reviews/phase1-code-review-v3.md)
- [最终独立验收复核：PASS](docs/reviews/phase1-acceptance-review.md)
- [全部审阅记录](docs/reviews)，保留前两轮问题及修复过程

下一阶段计划：用户 reference 接入与数值分析 → 访问索引/边界与布局观察 → 分层采集和报告。硬件范围继续聚焦 NVIDIA H200。
