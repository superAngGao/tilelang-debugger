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
| 访问观察 | 元素读写、`T.copy`、单元素 atomic add/min/max 和取址的运行时逻辑参数；与数值点共用采集协议 |

以上是功能范围；具体已测组合见[验证记录](docs/unified-capture-validation.md)。测试不能证明任意程序的同步或线程分工正确。

## 环境与安装

GPU 运行使用 Linux + H200。schema 3 的版本验证覆盖 TileLang `0.1.12`（checkout `2d63708c8ad57196051c4636a1167c5453c73a48`）及独立安装的 PyPI `0.1.15`，具体矩阵见[兼容性与表达式验收](docs/access-compatibility-validation.md)。共同环境为 PyTorch `2.10.0+cu129`、CUDA 编译器 `13.2.78`、驱动 `595.71.05`、Compute Sanitizer `2026.1.1.0`。

运行前检查所需前端接口和 copy 区域行为，将结果写入 `compatibility.json`；`environment.json` 记录真实包路径和 helper 摘要。接口检查通过不等于任意版本都已验证；未测版本不声明支持。旧 reviewed 入口仍限定原 0.1.12 契约。本包不自动安装或替换 GPU 依赖。

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

### 看访问参数

在同一份 schema 3 配置中增加 `mode: "access"`。例如原第 42 行包含 `v = T.if_then_else(i < n, x[i], 0)`：

```json
{
  "id": "input_access", "line": 42, "when": "before",
  "mode": "access", "buffer": "x", "operation": "read",
  "block": [0, 0, 0], "loops": []
}
```

仍使用 `run --monitor ...`，可和数值点混用。工具找到该语句中的 `x[i]`，记录当时的 `i`、条件值、shape、逻辑 stride 和内存级别，不为观察而额外读取 x。索引中已绑定的 scalar 可直接使用；嵌套内存读取或有副作用的函数调用不能复制求值，需要先显式绑定为 scalar。条件内可能未定义的除法也需显式绑定，避免打印改变原程序的求值条件。

- `operation` 可省略；支持 `read`／`write`／`read_write`／`address`。存在多个匹配访问时，用操作类型、明确的 `buffer: "x[i & 31]"` 或从 0 开始的 `occurrence` 消除歧义。先选择候选，再检查它的求值依赖。
- 访问点只使用 `when: "before"`。循环、分支、block、thread 选择与数值点一致；if 内的点只记录实际到达的分支。语句内的条件表达式、and/or、`T.And/T.Or` 和链式比较统一记录候选索引和路径条件。
- 对 `T.copy` 选择源或目标 buffer，支持位置参数和 `src`／`dst` 关键字参数，记录逻辑起点、区域大小和边界。无须 `region` 或 `collective`，也不增加同步。它是源语句的区域请求；可选择所属前端线程减少重复请求。
- `T.atomic_add/min/max` 的单元素目标记录为 `read_write`，`T.address_of` 的单元素参数记录为 `address`。取址仅表示指针形成请求，不表示发生了读写，也不输出物理地址。原语调用次数不增加。
- 嵌套读取、未知 buffer 宏和整块原子操作仍需明确适配。所选访问之前有副作用调用时拒绝；之后的前端调用可能修改 local.var 索引时也拒绝。链式／解包赋值中的访问暂不支持。需要改写为分开的标量绑定语句，避免采到不同求值时刻的索引。

运行结果新增 `accesses.jsonl`、`access-summary.json` 和 `access-report.md`。每条请求保留源码行、launch、观察点、线程、循环身份，以及 `origin/extent/shape/stride/active`。报告区分范围内、部分相交、范围外、掩码关闭和空区域。copy 的尾部部分相交不等于非法硬件访问；采集完整性另由同一套 D／E 计数验证。

偏移由索引与前端逻辑 stride 推导，相对 buffer 的逻辑基址；不是 GPU 指针或 shared 物理布局。记录不证明访存已经完成，也不证明范围内的索引符合算法预期。IR／CUDA 继续导出供查阅。运行例子见 [access.py](examples/unified/access.py) 和 [access_expressions.py](examples/unified/access_expressions.py)，设计见[表达式适配方案](docs/access-expression-adaptation-plan.md)。

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

分析只需 CPU 与 PyTorch，不加载 TileLang、不重跑 kernel。先复核原日志、源码及编译产物、launch 身份、快照和哈希，再调用用户提供的 `reference(inputs, points)`。每个 launch 单独调用，返回 `points`、`arguments_after`、`outputs` 三部分，覆盖全部数值观察点、传入参数和返回值。access 点不传给数值 reference，其参数和逻辑范围报告单独复核；数值匹配不代表索引算法正确。

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

移除旧 trace 前的 0.2 安装包已通过独立验收：**99 项 CPU/TIR 测试、102 次完整 schema 3 采集、4 次预期预算截断、10 次预期拒绝，以及 7 次旧例回归**。独立复算包括原始日志、reference、证据扰动和 reader 语义；截断与拒绝不计为完整采集通过。方案、命令、sanitizer 范围及安装包 SHA 见[本轮记录](docs/unified-capture-validation.md)。

## 兼容入口与访问观察

旧 schema 1/2 数值配置保持兼容；旧 GELU、Sum、GEMM、GQA 的受审入口使用 `--engine reviewed`，该入口继续保留原样例契约。旧格式及数值说明见[源码引擎记录](docs/generic-kernel-validation.md)、[嵌套采集记录](docs/observation-validation.md)、[阶段 1](docs/phase1-validation.md)及[reference 方案](docs/phase3-plan.md)。

```bash
python -m tilelang_debugger run examples/gqa/run.py --engine reviewed \
  --monitor examples/gqa/monitor.json --output artifacts/gqa
```

旧 `trace` 命令及其专用实现已移除。它依赖固定样例的源码、行号、形状和 CUDA 契约，未形成通用编译结果分析能力；不再作为产品入口或后续泛化目标。历史实现与实验可在 [移除前提交](https://github.com/superAngGao/tilelang-debugger/tree/e838fc88d96a562ee6230ee154d6593bac6632f8) 查阅。

移除后通过 89 项 CPU／TIR 回归及安装包检查，详见[清理验证记录](docs/trace-removal-validation.md)。

IR／CUDA 导出继续保留，供用户查阅；它不代表自动分析编译变换、布局或同步正确性。“看访问”已通过上面的 `mode: "access"` 接入统一源码观察点，范围是逻辑访问参数。默认数值模式的元素观察会新增读取，两者的含义不同。原 GQA 输出同步修正及实验边界见[样例说明](examples/README.md)。

## 当前边界

- 自动 `Pipelined(num_stages=...)` 支持；手写 `order/stage/sync/group` 调度数组需要专用适配，当前明确拒绝。
- while、break、continue 保留原语义；固定前端的条件/嵌套/value return 属于 Python 构建控制，不能冒充 GPU early return。当前只接受 kernel 末尾 bare return。loop-else 和未知作用域也明确诊断。
- 协作 ready/uniform、组内 barrier 保留是用户契约；不自动证明异步完成、任意分支一致性或跨 block 快照。
- schema 3 可绑定输入 tensor 的符号形状／stride，支持非连续与零 stride 视图、按前端参数名传入的关键字参数，以及调用方当前 stream。launch 之间仍同步采集，不支持 CUDA Graph 或跨 stream 并发时序分析。零维 tensor 的快照可用，但当前 TileLang 基线无法编译零维 tensor 参数，未宣称 kernel 支持。
- 动态参数表达式保留整数除法、余数、Cast 和位宽语义；不解任意符号方程，除零及 signed 溢出明确报错。baseline 必须成功执行，失败 kernel 的部分日志恢复仍待实现。
- 打印、shared 中转和同步会影响调度与性能；采集结果不代表原程序性能。FP8、自动根因诊断和交互式报告尚未提供。

## 文件组织

```text
src/tilelang_debugger/
  source_analysis/       源码位置、绑定、统一控制流与适配器
  instrumentation/       按模型原位插入观察、计数器及循环别名
  frontend/              前端对象适配、保留原符号身份的引用
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

统一采集基础版本的[方案](docs/unified-capture-plan.md)、[设计审阅](docs/reviews/unified-capture-design-review.md)、[流水线协议修订审阅](docs/reviews/automatic-pipeline-root-design-review.md)、[核心代码审阅](docs/reviews/unified-core-review.md)、[运行与证据审阅](docs/reviews/unified-runtime-review.md)、[独立验收](docs/reviews/unified-acceptance-review.md)。

源码访问接入有[统一修正方案](docs/source-access-stabilization-plan.md)和[首次验证](docs/source-access-validation.md)。后续表达式与兼容性改动有[已审阅方案](docs/access-expression-adaptation-plan.md)、[表达式代码审阅](docs/reviews/access-expression-code-review.md)、[运行参数与兼容性审阅](docs/reviews/access-runtime-compatibility-review.md)和[最终验证](docs/access-compatibility-validation.md)。历史审阅不代替新改动审阅。
