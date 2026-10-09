# TileLang Debugger

面向 **NVIDIA H200** 的 TileLang 源码定点调试原型：选择源码行、buffer、block 和循环迭代，采集该位置的完整 tile，并保留插桩前后的 IR 与 CUDA，帮助核对中间计算结果。

**当前能力：源码定点数值采集、用户 reference 离线分析，以及运行时访问索引与边界观察。** 支持范围是本仓库 GELU、Sum、GEMM、GQA 四个样例的固定构建参数和已审阅观察位置。当前以命令行和文件产物为主，尚未支持任意用户 kernel 或交互式调试界面。各阶段验证和独立审阅见文末链接。

新增的 [TileOPs 外部示例](examples/tileops/README.md) 直接调用指定 checkout 的 Softmax、RMSNorm、RoPE，用于数值基线与真实调试接入测试。它们不增加产品白名单；当前外部模块接入仍是未完成项。示例基线成功与 debugger 采集成功分别记录，完整验收使用 `--require-debugger`。

这批外部示例已在 H200 上通过 **21 组数值基线和 9 次 sanitizer 检查**；6 次公开 CLI 接入探测均为 `unsupported`，严格调试验收返回失败。具体范围、命令及证据见[验证记录](docs/tileops-integration-validation.md)。

## 指定源码文件

`run` 和 `trace` 都接受用户指定的 `--source FILE`，不再要求输入文件名为 `kernel.py`，也不再寻找 driver 同目录的这个文件。

```bash
python -m tilelang_debugger run examples/gelu/run.py \
  --source /path/to/gelu_impl.py --monitor examples/gelu/monitor.json \
  --output artifacts/gelu-custom-source
```

`--source` 的相对路径基于当前工作目录；未指定时使用配置中的 `source`，其相对路径基于**配置文件所在目录**。显式参数优先。实际路径保存于 `source.json`、运行/观察点元数据及报告，原文件不会被修改。worker 中 `source/kernel.py` 是内部快照名称。

这项修改解决文件输入，**没有解除当前源码内容/driver 契约限制或实现外部包/JIT 的通用接入**。以上例子中 `gelu_impl.py` 可是已有 GELU 源码的改名文件；受审 driver 仍通过内部 `kernel` 模块加载快照。任意用户 driver 的 import 方式尚未通用化。TileOPs 接入探测现显式传入上游文件，在源码/driver 契约处报告 unsupported，缺失文件则判为错误。

已在 H200 上验证改名、移目录和含空格路径，数值/访问采集各得到 4096 条记录，报告关联实际输入文件；详见[路径输入验证](docs/source-path-validation.md)。

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

## 看一次访问的索引与边界

`trace` 在源码中选择访问语句、block 和串行循环次数，采集 lowering 后**原访问表达式在 GPU 上的值**。它与数值 `run` 分开执行，使用独立的 `access.json`；不会在原 kernel 文件里写入代码。

```bash
CUDA_VISIBLE_DEVICES=0 python -m tilelang_debugger trace examples/sum/run.py \
  --access examples/sum/access.json \
  --output artifacts/sum-access-001 --sanitizer memcheck
```

| 样例 / 配置 | 原始源码位置 | 默认选择得到什么 | 记录数 |
| --- | --- | --- | ---: |
| GELU / `examples/gelu/access.json` | 87 行 read x、90 行 write y | global 元素偏移、字节偏移、线程及向量分量 | 4096 |
| Sum N=257 / `examples/sum/access.json` | 35 行 read x，33 行循环第 1 次 | 512 个候选访问：257 active、255 masked | 512 |
| Sum N=256 / `examples/sum/access_unpadded.json` | 42 行 read x / write shared_buf；47 行 read shared_buf | global→shared 两端分别记录，以及第一行 shared 读取 | 1280 |
| GEMM / `examples/gemm/access.json` | 93/98 行 transfer a/b，70/77 行循环均第 2 次 | 两次实际 TMA 调用的坐标、shared 偏移、barrier 下标及 issuer | 2 |
| GQA / `examples/gqa/access.json` | 114/115/120/128/293/455 行 | Q 两次加载、第二轮 K/V 加载、两个 consumer 输出写回 | 6 |

Sum N=256 使用驱动 `examples/sum/run_unpadded.py`。其余驱动均为样例目录下的 `run.py`。所有默认配置选择 block `(1,0,0)`；`loops[].iteration` 从 1 开始，记录保留从 0 开始的实际循环变量。可在这些受审位置选择其他合法 block / 串行迭代；GEMM 的 ring slot 必须与所选 K 迭代匹配。

普通读写逐线程、逐向量分量记录索引，保持原向量访存不变，不额外读取目标数据。掩码为 false 的候选访问单独展示，不能把它判为非法读取。TMA 则**每次实际搬运调用一条记录**，不伪造逐线程 global 元素访问；线程来自原 election 的结果。

主要产物：

- `access-report.md`：源码行、记录数、active/masked 统计和 TMA 逻辑区域。
- `access-records.jsonl`：runtime block、线程、循环值、索引或 TMA 实参；主键包含 run / launch / site，不依赖日志顺序。
- `access-points.json`：源码身份、原索引/guard 表达式、向量宽度、预期事件域及 shared 别名信息。
- `access-analysis.json`：边界解释和 TMA 区域。descriptor 参数标为 `derived_from_host_ir_and_launch_bindings`，不声称读取过 opaque descriptor。
- 两个 worker 目录中的 `pretrace.py/json`、`codegen.py/json`、`kernel.cu`、`descriptors.json` 和进程/门禁/sanitizer 记录；插桩目录另存 `traced.py/json`。

实现是在原 CUDA pipeline 结束后追加 Python IR 日志转换，继续使用既有 `printf`；不新增 TileLang intrinsic、C++ lowering 指令或安装文件修改，也不添加采集 barrier/shared 中转。插桩前 IR 必须符合已授权 baseline；移除日志后，pipeline 边界和真实 codegen 输入均须与基线结构等价。TMA host 构造与实际 launch 参数另外核对。实验中的 FFI hook 会在进程退出时崩溃，产品路径没有采用它。

`trace` 支持 `--sanitizer racecheck|synccheck|memcheck`，记录预算为 65,536 条、FIFO 至少 64 MiB。缺失、重复、截断、未知事件、原程序故障或两版输出不一致均失败。最终 reference 不匹配时保留采集，CLI 退出 2。访问记录描述插桩程序的 IR 实参，不证明硬件事务完成；TMA shared swizzle 的物理逐元素地址尚未展开，也不自动判断并发竞态或根因。

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

**数值不匹配也可以完成采集。** 此时 `run.json` 为 `status: passed`、`numerical_status: failed`，CLI 退出 2，保留完整 tile 供后续诊断。执行异常、同步检查失败、数据缺失或插桩改变输出仍是采集失败，退出 1；不能作为成功采集交给正常分析。单独运行样例驱动时，reference 不匹配仍会抛出断言。

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

## 用 reference 分析数值

四个样例均提供 `examples/<case>/reference.py`，包括中间 tile 和最终输出的独立计算。Sum 的两个尺寸共用一个 reference。分析只需 PyTorch 和 CPU，不加载 TileLang，也不需要重跑 kernel：

```bash
python -m tilelang_debugger analyze artifacts/gelu-001 \
  --reference examples/gelu/reference.py \
  --output artifacts/gelu-analysis-001
```

分析目录必须尚不存在，且位于 capture 目录之外。工具先核对原始设备日志、记录完整性、输入/输出快照及执行证据，再比较数值。上一阶段保存的完整 capture 也可以直接分析；原始 capture 不会被改写。

| 分析产物 | 内容 |
| --- | --- |
| `report.md` | 人可读摘要、每个 tile/输出的误差与前 20 个错误坐标、源码行和 block/迭代 |
| `analysis.json` | 完成状态、是否全部匹配、容差、错误数、最大误差、NaN/Inf 数量及来源 |
| `elements.jsonl` | **全部元素**的坐标、actual/expected、双方原始 bits 与 dtype、绝对/相对误差及是否匹配 |
| `reference.py` | 实际执行的用户 reference 源文件副本 |
| `evidence.json` | 本次分析所依据的 capture 文件 SHA256 清单 |

可从样例 reference 改写自己的函数；这是显式执行的可信 Python 文件，不是沙箱。第一版使用自包含文件，可 import 标准库和 PyTorch，不支持相对导入其他本地模块：

```python
def reference(inputs, points):
    # inputs: 本次 baseline 实际输入的 CPU tensor tuple
    # points: 观察点元数据，含源码行、buffer、shape、block、loops 等
    expected_tile = ...    # 根据实际输入及观察点计算
    expected_output = ...
    return {
        "points": {
            points[0]["id"]: {"tensor": expected_tile, "atol": 0.001, "rtol": 0.0001}
        },
        "outputs": [{"tensor": expected_output, "atol": 0.0625, "rtol": 0.002}]
    }
```

上例为单观察点接口示意，完整实现见 [GEMM reference](examples/gemm/reference.py)。返回值必须覆盖所有选中点和全部输出，shape 严格一致，不做广播。reference 必须是非空 CPU tensor，浮点支持 FP16/BF16/FP32/FP64；int32 actual 只接受 int32 reference 和零容差。输入及观察点在调用前复制，provider 原地操作不会改变采集的 actual 数据。

比较采用 `abs(actual - expected) <= atol + rtol * abs(expected)`，浮点比较在主机使用 float64。NaN 总是不匹配，同号 Inf 匹配，正负零数值相等且保留原始 bits。期望值为零而实际值非零时，相对误差为 `+Inf`。JSON 中特殊值使用字符串或 null，不使用非标准 NaN/Infinity token。最大误差不纳入 NaN 项，需结合特殊值计数查看。

分析退出码：全部匹配为 **0**；分析完成、发现数值差异为 **2**；证据不完整、reference 接口/计算异常为 **1**。`analysis.json` 相应区分 `completed + matched=true/false` 与 `failed`。新 reference 的结论与驱动原有 reference 结论分别保存：可以对原 reference 不通过的 capture 换一个 reference 重新分析。

普通 `run` 不会额外生成 `validation.json`，该文件由下面的严格验收脚本生成。离线分析的结论保存在独立目录的 `analysis.json`，不会把错误样例改写为验收通过。

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

数值 `run` 的采集宏沿用 TileLang 已有的 fragment→shared 搬运和 printf 机制，在内部搬运后及打印后安排组内同步；异步 accumulator 使用已有 operand-fence intrinsic。数值路径没有新增 lowering pass 或指令，也没有修改安装的 TileLang。访问 `trace` 的 Python IR 转换见前述说明。原始源码不被覆盖，编译缓存被禁用，导出的是实际编译对象的产物。

当前限制：

- 仅支持经过审阅的四个样例、固定构建参数、观察位置与布局；未知同步协议或布局会在 launch 前拒绝。
- 数值采集对象为完整 fragment，暂不支持直接 shared buffer 数值采集、`T.Pipelined` / `T.unroll` 内选点或多 launch；访问 trace 可查看上述固定 shared 读写。
- 一次采集最多 65,536 个元素，设备 printf FIFO 至少 64 MiB；仍以实际记录完整性检查为准。
- monitor 会增加 shared memory、寄存器和同步开销，插桩程序不能代表原程序性能。
- 访问观察限于表中已审阅位置；任意数据相关 gather 索引、自动根因诊断和交互式报告尚未实现。reference 由用户提供，不自动生成。

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
python -m unittest discover -s tests -p test_numerics.py -v
python -m unittest discover -s tests -p test_access.py -v

# 需要 PyTorch，CPU 即可；不加载 TileLang
python -m unittest discover -s tests -p test_analysis.py -v

# 需要真实 TileLang/TVM，但测试本身不 launch GPU
python -m unittest discover -s tests -p test_ir.py -v

# 先在 H200 生成 GQA trace，再只用 CPU 检查真实 host/device IR 及拒绝负例
TLACC_FIXTURE=artifacts/gqa-access-001 python -m unittest discover -s tests -p test_access_ir.py -v

# 访问观察五条路径；依次把 tool 换成 synccheck、memcheck（输出目录须不同）
CUDA_VISIBLE_DEVICES=0 python tests/validate_access.py \
  --output artifacts/access-racecheck --sanitizer racecheck
CUDA_VISIBLE_DEVICES=0 python experiments/access_protocol_probe.py \
  --output artifacts/access-protocol

# 完整 CPU 回归按 suite 隔离；analysis 测试要求从未导入 TileLang
python tests/run_cpu.py --output artifacts/cpu-regression \
  --access-fixture artifacts/gqa-access-001

# H200 上重现 T.print 同步的八个正负对照
CUDA_VISIBLE_DEVICES=0 python tests/validate_print_probe.py \
  --output artifacts/print-regression
```

- [阶段 1 方案](docs/phase1-plan.md)
- [第三轮独立代码复审：PASS](docs/reviews/phase1-code-review-v3.md)
- [最终独立验收复核：PASS](docs/reviews/phase1-acceptance-review.md)
- [全部审阅记录](docs/reviews)，保留前两轮问题及修复过程
- [第三步实施方案](docs/phase3-plan.md)与[独立设计审阅](docs/reviews/phase3-design-review-v1.md)
- [第三步独立代码复审：PASS](docs/reviews/phase3-code-review-v2.md)与[CPU/H200验证记录](docs/phase3-validation.md)

- [第四步方案](docs/phase4-plan.md)与[第三轮独立设计审阅：PASS](docs/reviews/phase4-design-review-v3.md)
- [第四步第二轮独立代码复审：PASS](docs/reviews/phase4-code-review-v2.md)
- [第四步最终独立验收复核：PASS](docs/reviews/phase4-acceptance-review.md)
- [第四步验证记录](docs/phase4-validation.md)；各轮代码审阅保存在 [docs/reviews](docs/reviews)。

后续方向：扩展受审观察位置、分层采集和报告。硬件范围继续聚焦 NVIDIA H200。
