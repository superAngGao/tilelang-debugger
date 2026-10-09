# TileLang Debugger

面向 **NVIDIA H200** 的 TileLang 源码定点调试工具。选择源码行、对象、block、循环迭代和逻辑选区，采集数值，用用户提供的 reference 离线核对，同时保存插桩前后的 IR 与 CUDA。

**0.2 的 `schema: 3` 统一了嵌套控制流、打印策略和运行记录。** 数值采集在 Python 源码层插入宏，使用已有 printf、搬运和同步机制；不解析 lowered IR/layout 来选择打印方法，不增加 TileLang intrinsic 或 C++ pass，不修改安装的 TileLang。源码和驱动无文件名限制，也没有算子白名单。

## 支持范围

| 维度 | schema 3 的行为 |
| --- | --- |
| 源码位置 | 原始语句前后，包括循环/分支整体前后；最多 8 个观察点 |
| 控制流 | 多重 serial/unroll/Parallel/Pipelined、分支、显式 Group，动态循环边界、while、合法 break/continue |
| 数据 | scalar、线程 local、fragment、shared/shared.dyn、global；整块或静态选区，也可观察单个 Buffer 元素 |
| 协作采集 | fragment 经 shared 中转；shared/global 直接读取；同步独立于 reader、迭代筛选及输出预算 |
| 类型 | bool、int8/16/32/64、uint8/16/32/64、FP16/BF16/FP32/FP64，保存原始位模式 |
| 运行 | 多次 build/compile/launch、标量参数、in-place、参数别名、多维 CTA 元数据 |
| 完整性 | 独立线程域与结束计数，检查整线程、整次访问、元素缺失、重复、截断和溢出 |
| 数值分析 | 按 launch 对齐中间样本、调用后的参数和返回值；整数精确比较，不转浮点 |

以上是功能范围；具体已测组合见[验证记录](docs/unified-capture-validation.md)。测试不能证明任意程序的同步或线程分工正确。

## 环境与安装

GPU 运行使用 Linux + H200。验证环境为 TileLang `0.1.12`（checkout `2d63708c8ad57196051c4636a1167c5453c73a48`）、PyTorch `2.10.0+cu129`、CUDA 编译器 `13.2.78`、驱动 `595.71.05`、Compute Sanitizer `2026.1.1.0`。工具核对所依赖 helper 的摘要；同版本号不保证实现相同。本包不自动安装或替换 GPU 依赖。

```bash
git clone https://github.com/superAngGao/tilelang-debugger.git
cd tilelang-debugger
python -m pip install -e .
```

## 从用户源码采集

```bash
python -m tilelang_debugger run /path/to/driver.py \
  --source /path/to/package/my_kernel.py --monitor /path/to/monitor.json \
  --output artifacts/my-capture --sanitizer racecheck \
  -- --driver-option value
```

原文件不修改；临时副本按原模块名导入，保留包内相对导入。source 可以与 driver 是同一文件。`--source` 相对当前目录；配置 `source` 相对配置文件目录，显式参数优先。`--` 后的参数原样传给 driver。输出目录必须尚不存在。

例如在源码第 66 行赋值后观察 scalar：

```json
{
  "schema": 3,
  "source": "/path/to/my_kernel.py",
  "budget": 65536,
  "points": [{
    "id": "running_max", "line": 66, "when": "after", "buffer": "max_val",
    "block": [0, 0, 0], "loops": []
  }]
}
```

`line` 指原始语句的开始行；`when` 为 `before` 或 `after`。`loops: []` 采集所有祖先迭代；`{"line": 59, "iteration": 2}` 选择该层第 2 次执行，从 1 开始，每次重新进入该循环重新计数。Parallel 用 `{"line": 45, "coordinates": [3]}` 选择原循环坐标。

整块 fragment/shared/global 需增加显式协作契约，例如：

```json
{
  "id": "acc", "line": 82, "when": "after", "buffer": "accumulator",
  "block": [0, 0, 0], "loops": [],
  "region": [[0, 32, 1], [0, 64, 2]],
  "collective": {"ready": true, "uniform": true}
}
```

`region` 每维为 `[start, stop, step]`，省略时观察整块。`ready` 声明数据已就绪；位于循环或分支内时，`uniform` 声明所有协作线程一致到达该位置。工具不会替用户证明异步操作已经完成。显式 `T.ws`/`T.WarpSpecialize` 组内采集还须用 `barrier: 1..15` 指定用户保留的 named barrier。已有 WGMMA 模式可指定 `fence_regs`，但 fence 不替代异步完成等待。

### 是否需要选线程

**普通整块/选区采集不需要指定线程。** 用户选逻辑元素，工具安排 reader；reader 只负责输出，不代表这些值由它计算。高级配置 `collective.reader` 可更换 reader，通常省略。

scalar/local 的值与线程有关，才使用可选 `thread` 筛选；省略时采集所有到达该点的线程。它可以是线性编号或 `[x,y,z]`，线性编号为 `x + extent_x * (y + extent_y * z)`。这是**源码前端线程域**，自动 warp specialization 后不一定等于硬件 CUDA 线程编号。记录明确保存 `thread_space: frontend`。整块采集不能用 `thread` 代替逻辑选区。

`buffer` 也可为纯索引元素，例如 `tile[i,j]`；这是新增读取，不是原访存追踪。索引越界会记录错误并跳过新增读取。循环和线程筛选控制输出；参与协作的线程仍执行必要同步。预算耗尽时也继续同步和原计算，不会留下半个 tile。

## 完整性与运行产物

每个 `run / compile / launch / point / frontend thread / visit / element` 有独立身份。`visit` 是该点该线程在一次 launch 内累计的选中到达次数，可区分重复坐标；记录同时保存原循环坐标及各层执行序号。

TLDBG3 使用 D 数据、E 结束计数及 X 非法读取事件。每个所选 block 的前端线程都必须输出 E，包括零次到达和编译期 inactive 的点。预期线程域来自前端 launch，不能从收到的日志反推。允许线程间日志乱序；缺失任意预期结束记录或已声明的数据均失败。

`budget` 是每次 launch 的事件上限，最大 65,536，先预留各点全部 E，再分配完整访问容量。完整前缀耗尽预算标为 `truncated`/`partial`；日志丢失不会被当作正常预算截断。零次到达表示“没有满足选择条件的观察到达”，不等于原语句绝对未执行。

```text
capture/
  run.json / monitor.json / source.json
  points.json                  # 每个 launch 的观察点契约
  records.jsonl                # 数据身份与原始 bits
  baseline/                    # 无 monitor 的编译/执行
  instrumented/                # 插入 monitor 后的编译/执行
    source/                    # 原源码、驱动及实际构建副本
    compiles/<id>/             # frontend/device IR、CUDA、编译信息、观察点
    launches/<id>/             # before/after 参数及返回值元数据、二进制
    execution.json             # build/compile/launch 清单
    environment.json / process.json
    stdout.log / stderr.log    # 原始设备日志和 sanitizer 信息
```

两版分别编译运行，逐 launch 比较输入、调用后参数和返回值。保留 dtype、storage 别名、offset 和 stride；不通过 clone 传参破坏原 kernel 的 alias。IR/CUDA 原样导出供查看，不用于数值采集准入。没有 reference 时数值状态为 `not_checked`。

`run` 退出码：完整采集 **0**，完整采集但 driver 数值检查失败 **2**，预算截断或配置点未被 launch **3**，执行/证据/同步失败 **1**。失败产物保留，partial 不会标成完整通过。

## 用 reference 分析

```bash
python -m tilelang_debugger analyze artifacts/my-capture \
  --reference /path/to/reference.py --output artifacts/my-analysis
```

分析只需 CPU 与 PyTorch，不加载 TileLang、不重跑 kernel。先复核原日志、源码及编译产物、launch 身份、快照和哈希，再调用用户提供的 `reference(inputs, points)`。每个 launch 单独调用，返回 `points`、`arguments_after`、`outputs` 三部分，覆盖全部观察点、传入参数和返回值。

点规格为 `{"schema": 3, "key": "execution", "dtype": "float32", "samples": [...], "atol": 1e-5, "rtol": 1e-5}`。每个样本含 `thread/visit/coordinates/ordinals/index/value`；使用 `key: logical` 时省略 `thread/visit`，其余坐标须足以唯一标识期望值。期望键集合由 reference 独立生成，不从已收到样本缩小范围。

参数和返回 tensor 规格为 `{"tensor": expected_cpu_tensor, "atol": ..., "rtol": ...}`，scalar 参数对应 Python scalar。整数要求同 dtype、零容差，精确到 64 位；浮点按 `abs(a-e) <= atol + rtol*abs(e)` 比较，NaN 不匹配。完整例子见 [reference.py](examples/unified/reference.py) 和 [advanced_reference.py](examples/unified/advanced_reference.py)。

产物包括 `analysis.json`、`report.md`、逐元素 `elements.jsonl`、reference 副本和 `evidence.json`。全部匹配退出 0，数值不同或 partial 分析退出 2，证据/interface 异常退出 1。reference 是用户授权执行的 Python 代码，不自动生成，也不是沙箱。

## 示例与验证

[统一小测例](examples/unified/README.md) 覆盖嵌套循环/分支、线程、内存、dtype、参数和运行身份。[TileOPs 示例](examples/tileops/README.md) 直接使用固定上游 checkout 中的 pool、indices、RoPE、Softmax、RMS kernel，产品不按这些名字分支。

```bash
python tests/validate_unified.py --output artifacts/core --sanitizer racecheck
python tests/validate_unified_advanced.py --output artifacts/advanced --sanitizer synccheck
python tests/validate_unified_advanced.py --types --output artifacts/types
python tests/validate_unified_pipeline_copy.py --output artifacts/pipeline --mode mixed
python tests/validate_unified_tileops.py --tileops /path/to/TileOPs \
  --output artifacts/tileops --sanitizer racecheck
python -m unittest discover -s tests -p test_unified.py -v
```

最终安装包已通过独立验收：**99 项 CPU/TIR 测试、102 次完整 schema 3 采集、4 次预期预算截断、10 次预期拒绝，以及 7 次旧例回归**。独立复算包括原始日志、reference、证据扰动和 reader 语义；截断与拒绝不计为完整采集通过。方案、命令、sanitizer 范围及安装包 SHA 见[本轮记录](docs/unified-capture-validation.md)。

## 兼容入口与访问观察

旧 schema 1/2 数值配置保持兼容；旧 GELU、Sum、GEMM、GQA 的受审入口使用 `--engine reviewed`，该入口和 `trace` 继续保留原样例契约。旧格式及数值说明见[源码引擎记录](docs/generic-kernel-validation.md)、[嵌套采集记录](docs/observation-validation.md)、[阶段 1](docs/phase1-validation.md)及[reference 方案](docs/phase3-plan.md)。

```bash
python -m tilelang_debugger run examples/gqa/run.py --engine reviewed \
  --monitor examples/gqa/monitor.json --output artifacts/gqa
python -m tilelang_debugger trace examples/sum/run.py \
  --access examples/sum/access.json --output artifacts/sum-access --sanitizer memcheck
```

`trace` 观察原访存的 runtime 索引、边界掩码和 TMA 实参，使用已有 Python IR 日志转换；它与 schema 3 的源码数值打印是独立路径。本轮没有将访问 trace 泛化到任意 kernel。支持位置及物理含义见[第四步验证](docs/phase4-validation.md)。原 GQA 输出同步修正及实验边界见[样例说明](examples/README.md)。

## 当前边界

- 自动 `Pipelined(num_stages=...)` 支持；手写 `order/stage/sync/group` 调度数组需要专用适配，当前明确拒绝。
- while、break、continue 保留原语义；固定前端的条件/嵌套/value return 属于 Python 构建控制，不能冒充 GPU early return。当前只接受 kernel 末尾 bare return。loop-else 和未知作用域也明确诊断。
- 协作 ready/uniform、组内 barrier 保留是用户契约；不自动证明异步完成、任意分支一致性或跨 block 快照。
- 运行要求静态 Buffer 形状、contiguous CUDA tensor、顺序 default-stream 调用；不支持 CUDA Graph、跨 stream 并发采集。零维 tensor 的快照可用，但当前 TileLang 基线无法编译零维 tensor 参数，未宣称 kernel 支持。
- 打印、shared 中转和同步会影响调度与性能；采集结果不代表原程序性能。FP8、自动根因诊断和交互式报告尚未提供。

## 文件组织

```text
src/tilelang_debugger/
  source_analysis/       源码位置、绑定、统一控制流与适配器
  instrumentation/       按模型原位插入观察、计数器及循环别名
  emitters/              按真实对象选择打印与协作策略
  protocols/             CPU 解码、身份和完整性检查
  runtime/               临时 hook、build/compile/launch、快照与证据
  unified_analysis.py    多 launch 的用户 reference 对齐
examples/                原四例、TileOPs 接入、统一机制小测例
tests/                   CPU/TIR 测试和 H200 验收脚本
experiments/             隔离机制实验
docs/                    方案、验证结果和独立审阅
artifacts/               本地产物，不提交 Git
```

本轮[方案](docs/unified-capture-plan.md)、[设计审阅](docs/reviews/unified-capture-design-review.md)、[流水线协议修订审阅](docs/reviews/automatic-pipeline-root-design-review.md)、[核心代码审阅](docs/reviews/unified-core-review.md)、[运行与证据审阅](docs/reviews/unified-runtime-review.md)、[独立验收](docs/reviews/unified-acceptance-review.md)。历史记录保留在 [docs/reviews](docs/reviews)。
