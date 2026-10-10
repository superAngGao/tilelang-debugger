# TileOPs 外部 kernel 测试方案

> 历史记录：旧 `trace` 已从当前产品移除；文中 trace 命令、专用测试和兼容性描述仅适用于[移除前提交](https://github.com/superAngGao/tilelang-debugger/tree/e838fc88d96a562ee6230ee154d6593bac6632f8)。保留当时结果，不代表当前产品能力。

本次落实用户确认的首批 Softmax、RMSNorm、RoPE 示例和目录结构。直接加载用户指定 TileOPs checkout 中的 kernel，建立可复现的 H200 数值基线与调试接入测试。已有四例的 PASS 不能外推到这些 kernel；本次也不能用增加源码哈希白名单让新例“通过”。

## 目录与来源

- `examples/tileops/{softmax,rms_norm,rope}/run.py`：可独立运行的上游 kernel 驱动。
- 各目录的 `reference.py`：CPU torch 数学 reference；不调用被测 TileOPs kernel。
- 各目录的 `monitor.json`、`access.json`：上游源码观察点意图。保留原文件路径、函数及语句锚点，运行时解析成当前文件行号；不是声称现有 CLI 已接受的新配置格式。
- `examples/tileops/common.py`：checkout 路径解析、来源记录、观察点定位、运行证据保存。
- `examples/tileops/manifest.json`：测试版本、参数矩阵、来源。版本用于复现，不能作为用户 kernel 的准入条件。
- `tests/test_tileops_integration.py`：定位/来源/验收状态的 CPU 回归。
- `tests/validate_tileops.py`：独立进程执行矩阵、记录 stdout/stderr/退出码/超时，汇总数值基线与调试接入状态。
- `artifacts/tileops/`：运行证据，忽略于 git。

不复制、改写上游 kernel 成本仓库的 `kernel.py`，不修改上游工作区。使用完整包导入并核对实际模块 `__file__` 位于指定 checkout，防止错误地测到已安装版本。记录 Git HEAD、工作区状态、实际源码摘要、参数、环境、输入/输出与实际编译产物。存在未跟踪文件不自动拒绝；来源如实记录。

## 首批矩阵

1. Softmax：单 tile 的 N=256、257；多 tile 的 N=513、tile_n=256；M=4、block_m=1、threads=128。FP16/BF16/FP32。输入原始 M×N，输出 padding 只比较数学有效区，同时验证 padding 为零。观察 FP32 转换、reduce_max、reduce_sum 与输入访问；多 tile 选择原始 serial loop 的迭代。
2. RMSNorm：N=256、257，M=4、block_m=1、threads=128，eps=1e-5。FP16/BF16/FP32。按上游 forward 约定将输入/weight 补零到 256 对齐，归一化分母仍用原 N。观察 sumsq、rrms、输出 fragment 与 weight 访问。
3. RoPE：原始 `_make_rope_neox_1d`，(seq_len,head_dim)=(32,64)、(16,128)，threads=128、num_per_thread=8，FP16/BF16/FP32。均为完整 block，避免在首批混入未知尾部行为。观察原始并行循环的 x/cos/sin 访问和 y 写入；无 fragment 的事实保留，不能为了迎合采集器重写 kernel。中间标量采集当前不支持则明确记录。

共 21 个数值基线组合。CPU reference 使用实际输入副本；保存 CPU 输入、实际输出、reference 和误差统计。记录 dtype 的 atol/rtol；任何不匹配导致基线验收非零退出，不得静默放宽阈值。最少各类一例运行 memcheck/racecheck/synccheck，超时/非零退出均为失败，不能把缺少 sanitizer 当跳过成功。

## 当前入口缺口必须显式暴露

当前 `run`/`trace` 要求 driver 同目录存在 `kernel.py`，随后按 source/driver 摘要、固定观察点、layout/CUDA 摘要准入。这三个真实上游示例不能直接进入现有采集流程。

每类分别调用现有公开 `run` 和 `trace` CLI，保存命令、失败日志及退出码。只有 traceback 明确来自当前入口读取 driver 同级 `kernel.py` 的 FileNotFoundError，且缺失路径精确匹配，才能分类为 `unsupported`（入口尚未解析到配置，后续兼容性未验证）。配置错误、依赖错误、超时、崩溃及其他异常分类为 `failed`，使默认验收也非零退出。意外返回成功不能当完整采集成功：须验证实际产物；在尚无适配验证器时分类为 `unverified` 并失败。不能用“预期拒绝测试 PASS”覆盖 debugger 状态。基线与 debugger 的结果分别汇总。完整验收模式 `--require-debugger` 在任一示例不支持时必须非零退出；默认基线模式的成功只表示测试样例和 reference 可运行。

本次示例接入不宣称修复通用调试入口；它提供后续通用化的外部验收集。这也避免把三个新 kernel 再写成三套产品特例。根 README 必须明确这个边界。

## 去除白名单的后续实现路径

1. 从指定 module/factory 捕获实际 PrimFunc、JIT 编译与 launch；支持真实包导入，不再要求 sibling kernel.py。source 位置来自原 AST，编译参数、输出位置、buffer shape/dtype、block/loop 域来自实际 PrimFunc/launch。
2. fragment 采集按实际布局证明 staging 覆盖、参与线程与同步位置；源码/driver/layout 摘要保留为来源证据，不作核准白名单。不能只删掉检查，否则现有 shape、线程数、循环边界仍然写死。异步/warp-specialized readiness 的语义检查单独保留。
3. runtime access 从实际 lowering 操作与来源锚点关联，记录实际计算出的地址操作数，保留 guard/vector lane；删除算子名驱动的公式与 SHAPES 分支。RoPE 的无 fragment 路径用于验证这一步。
4. 继续使用原数值、IR、截断/重复/覆盖检查。新增 kernel 只能增加示例配置/reference，不能改 debugger 中的算子分支。改名、空行、形状变化与合法但算错的索引应可采集，数值不匹配是报告结果。

这部分需要自己的实现与独立审阅，不能把本次基线验收标成通用化完成。

## 审阅和交付

方案独立审阅通过后实施；代码独立审阅、修正至无阻塞意见，再运行/核验 H200 证据并更新 README、验证文档。保留全部未通过项。此次不改变既有产品采集模块，运行相关原 CPU 回归确认示例接入没有影响已有接口。
