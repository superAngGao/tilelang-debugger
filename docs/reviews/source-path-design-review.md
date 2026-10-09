# 用户指定源码路径修正：独立设计审阅

**结论：PASS，无设计阻塞项。** 该方案可修复用户不能选择文件路径的问题；不代表解除源码内容、driver 或布局契约，也不代表支持外部包/JIT。

- 日期：2026-10-09。
- 受审文件：`docs/source-path-plan.md`。
- SHA256：`4e9b547199735a623c813a6990f7fca38b491213d0d3f81b9d37b9a2cb9d744a`。
- 独立核对：当前 CLI、capture/access 入口、两套 prepare 契约、数值报告与 GELU 配置。没有修改实现或运行 GPU。

## 接受理由

1. 方案给出可执行的解析顺序：显式 `--source` 按调用 cwd，缺省时配置 `source` 按配置文件目录；不再从 driver 路径推断。输入确实决定读取的内容，且不存在失败后回退到 sibling 文件的路径。
2. 内部 worker 的 `source/kernel.py` 保留为执行快照名，兼容当前受审 driver import；原始路径另行记录和显示。这样改名和移动源码可以生效，同时没有假装已支持修改 driver import 或真实包/JIT。
3. H200 要求改名/移动后的 GELU 在真实 `run` 与 `trace` 上成功，而不是只验证解析器或观察拒绝。这是证明用户输入真正生效的恰当验收。
4. 旧默认配置仍可在配置目录找到 `kernel.py`，兼容现有命令；旧产物缺少路径字段时采用兼容显示，不重写原证据。
5. TileOPs 探测显式提供上游真实源码及旧格式配置，推进到真实 source/driver 契约拒绝，保留 unsupported；不再把已修复的 FileNotFoundError 当作预期结果。

## 实施与验收要点

- 路径字段必须传至两个公开入口及相应 run/request/point/report；成功、失败的 run 元数据均尽量保留同一份路径选择信息。不要在 prepare 前无条件覆盖配置原值而丢掉用户请求。
- prepare 的输入限制改为非空字符串路径；不得仅让 CLI 接收参数，却继续在 prepare 限制 `source == 'kernel.py'`。错误配置和空参数继续失败，显式路径缺失不尝试配置 fallback。
- 报告显示的路径与行号应来自实际读取的原始文件，不能把内部快照路径当作原始来源。离线显示使用保存的元数据，不要求原始文件仍在原位置；路径包含空格或 Markdown 符号时保持可读。
- H200 验收核对原文件摘要与 baseline 快照摘要一致、run/request/point/report 的路径一致，并验证 capture 数值/reference、access 数据及 baseline/instrumented 输出位一致。构造不同内容的显式 source 加上有效 sibling 文件，必须拒绝显式文件，证明没有隐式回退。
- TileOPs 的 RoPE 没有 fragment monitor 点。若为了测试 `run` 契约门槛而使用标量语句构造字段合法的探测配置，应明确它仅用于准入探测，不能声称该点已符合 fragment/同步要求。配置合法性和真实内容/driver 契约拒绝是两个不同层次。
- 新探测分类只接受相应 prepare 位置、准确 Unsupported 消息及实际 `--source`/配置的来源证据。路径错误、错误配置、依赖错误、超时/崩溃仍失败；历史 FileNotFoundError 证据保留原义，新的验证文档必须说明拒绝位置已经改变。

实施后仍需独立代码审阅与 H200 证据审阅。本次不增加任何新的源码/算子/CUDA 白名单项。
