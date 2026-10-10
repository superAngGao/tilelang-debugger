# 源码访问表达式与版本兼容性验收

2026-10-09。对应[已审阅的方案](access-expression-adaptation-plan.md)。本轮补齐上下文测试、版本适配和表达式分类，不新增 lowering pass，不恢复旧 trace。

## 包与环境

- 最终工作目录：`/home/ang.gao/tilelang-debugger-access-final-20261009`。
- wheel：`artifacts/tilelang_debugger-0.2.0-py3-none-any.whl`，SHA256 `4b920e012ca6ada9f2897012eb382d640dfb38205bb07fa339a32b9930eca700`。
- 所有最终 worker 从该目录的 `installed/` 导入 debugger，使用同一个 wheel。
- 0.1.12：原 checkout `2d63708c8ad57196051c4636a1167c5453c73a48`。
- 0.1.15：PyPI wheel，独立环境 `/home/ang.gao/tilelang-debugger-compat-015-20261009`；共享 Torch 等依赖路径，但不执行基线环境的 editable `.pth`，防止测试实际仍导入 0.1.12。
- NVIDIA H200；PyTorch 2.10.0+cu129；CUDA 13.2.78；Compute Sanitizer 2026.1.1.0。

官方版本信息见 [TileLang releases](https://github.com/tile-ai/tilelang/releases)。0.1.13/0.1.14 未实测；不由两个端点推断整个区间都已验证。

## 最终矩阵

最终安装包在两个版本均全部通过：每版本 117 项 CPU/TIR、21 次 capture、33 对 baseline/instrumented launch；合计 42 次 capture、66 对 launch。独立验收另行重跑两版本全部 CPU suite，并重算每版本 21 次采集证据、16 组访问索引 oracle 和 10 份数值 reference，全部通过。

主验收摘要位于远端 `artifacts/v012/summary.json`、`artifacts/v015/summary.json`；汇总为 `artifacts/final-summary.json`，本地副本 `artifacts/access-compatibility-final-summary.json`。两个版本的通过数不包含前期失败尝试。

| 测试组 | 内容 | 每版本 capture / launch 对 |
| --- | --- | --- |
| CPU/TIR | 14 个独立 suite，参数算术对照、cast、前端契约、源码路径、证据和重叠快照 | 117 项 |
| contexts | pipeline/group/copy × 全线程/指定线程；shared/fragment 区域 | 6 / 10 |
| expressions | 条件路径 224 条、操作适配 160 条逻辑请求 | 2 / 2 |
| runtime | 动态 n=17/39，stride=1/2/0，kwargs，调用方 stream，数值 reference | 3 / 6 |
| access | 原索引/偏移一位、while、branch、copy、混合数值点、扰动验证 | 2 / 2 |
| values | pipeline/group/fragment/shared/dynamic 旧数值回归，每例两次 launch | 5 / 10 |
| TileOPs | MaxPool/RoPE/Softmax 原源码，独立枚举实际索引 | 3 / 3 |

contexts、expressions、runtime、access 使用 racecheck；values、TileOPs 使用 synccheck。每个成功 capture 核对 baseline/instrumented 输入、参数变化及输出逐位一致。访问测试逐请求核对独立生成的索引/条件；前端节点计数检查目标读取、copy、atomic、address、同步不增加，只用于验收。

## 发现与修正

1. 0.1.15 的 region 操作从 `tl.tileop.region` 改为 `tl.region`，公共 print 函数移到后端模块；适配实际对象和公共函数位置，不检查固定 helper 哈希准入。
2. 候选选择与校验分离；统一布尔和条件路径，copy 两侧均检查纯度与定义域。
3. 所有调用参数按原求值顺序遍历，保留副作用前缀和后缀；前端 local.var 依赖会触发延迟求值保护。链式/解包赋值保持明确边界。
4. 参数表达式区分 trunc/floor 除法与余数，保留 Cast、位宽及定义域；narrowing Cast 用独立字节解释核对，因为 0.1.12 Analyzer 对部分越界常量转换本身报错。
5. 实验失败也保留：最初 0.1.15 editable 路径串用、print 模块迁移、region 名称；fixture 对只读输入取址产生 const 指针编译问题，改为对可写输出取址验证。失败记录在 `tilelang-debugger-access-followup-20261009/artifacts`，不计为成功测试。

## 独立审阅与边界

表达式设计和代码见[独立审阅](reviews/access-expression-code-review.md)，运行参数/兼容性见[独立代码审阅](reviews/access-runtime-compatibility-review.md)，安装包与双版本结果见[独立验收](reviews/access-compatibility-acceptance-review.md)。三项均通过；各自列出审阅范围，不以旧审阅代替当前工作。

未知宏、嵌套读取、整块原子、危险的推测求值仍需显式标量绑定或新操作适配器；动态符号不做通用方程求解。失败 kernel 部分证据恢复不属于本轮。接口探测通过只说明已检查的接口行为匹配，不代替 GPU 验收。
