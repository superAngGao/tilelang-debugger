# 源码控制流与组件组织：独立设计审阅

日期：2026-10-09。结论：**PASS**。

审阅对象：`docs/source-control-flow-design.md`，SHA256 `02b7f2ee2acdf75d34677cb3a8e3b2a76f819fc21bcc5081bf9a29aeba6e0233`。产品基线：`5172d3c`。本次审阅独立检查了设计及当前 `scopes.py`、`source_instrument.py`、`instrument.py`、两个 monitor、两个 records 和打包配置；未修改产品代码，也未运行 GPU 测试。

## 通过理由

- 控制流设计同时包含结构化嵌套、顺序和控制转移边，能表达前置兄弟语句的 continue/break 对观察点的影响。循环种类由适配器解释，避免为每种嵌套组合重新实现解析。
- 明确区分宿主构建、macro 和设备阶段，不把 Python AST 接受某种语法等同于固定 TileLang 前端可执行。Pipelined 参数与 serial step 分开处理；Group 的参与域需要明确来源。
- 动态边界和分支条件保持原求值次数与位置；重复 visit 有父实例内的序号，提前退出指向真实目标，不伪造正常收尾。不能保持这些语义时保留不完整证据，而非重新求值条件。
- 语法能力、对象打印策略和覆盖证明分别判定。scalar/local 不因覆盖未知而被整体禁止；协作打印仍检查完整祖先路径、实际参与域与数据就绪。源码分析不假装证明异步完成或日志完整。
- 目录拆分围绕源码分析、源码改写、前端打印、日志协议、运行导入和共享状态。离线协议不依赖 TileLang/emitter；两个打印路径共享唯一 session，避免复制旧 monitor 全局状态。
- 本轮明确只做保持行为的重构，未来动态循环、Group/Pipelined 和 transfer 协议尚需各自设计与验收。未要求本轮提前实现这些能力。

## 实施时需兑现的检查

这些是设计已有约束的具体检查点，不是新的范围或本次阻塞项。

1. 旧 `instrument.py` 也属于已有调用入口：搬移 reviewed 路径时保留导入兼容。其 `contracts.json` 路径不能继续机械地相对新子目录查找。
2. 不使用 `from capture_state import _active` 一类值拷贝保存可重新绑定的 session 全局变量。验证旧 monitor 入口、samples emitter、tiles emitter 访问同一活动状态，异常后清理以及嵌套 session 拒绝行为保持不变。
3. 对三个入口比较实际分析结果和注入源码；测试旧模块导入及已保存注入源码的 helper 路径继续可用。对 source loader 的异常恢复也应保留原检查。
4. 在无 TileLang 的进程中导入协议并重验保存日志；安装构建出的 wheel 后执行导入与 CLI 检查，不能只依赖源码目录测试确认子包存在。
5. 按计划运行 nested、local、Softmax samples、旧 source fragment、reviewed GQA 的 H200 回归，核对记录/reference。后续独立代码及验收审阅须审查实际结果，本设计 PASS 不等于重构已经通过。

未发现阻止按本轮范围实施的设计问题。
