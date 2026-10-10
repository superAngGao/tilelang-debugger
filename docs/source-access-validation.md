# 源码访问观察接入与验证

2026-10-09。访问观察通过 schema 3 的 `mode: access` 接入已有 run、源码作用域、插桩、打印、D／E 完整性协议和证据核对，不恢复旧 trace 命令，不增加 lowering pass。

本文件保留首次接入与统一修正时的验证范围。后续增加的 atomic/address、表达式适配、双版本兼容性及独立审阅见[后续验收](access-compatibility-validation.md)；下文当时尚未支持/审阅的说明不代表后续版本状态。

发现前端对象适配和符号绑定问题后，按用户要求先完成[统一分析与修正方案](source-access-stabilization-plan.md)，再按职责整理实现。下方保留首次接入的实验记录，最终结构调整验收另列。

## 实现范围

- `source_analysis/access.py` 选择源码元素读写或 T.copy 的一侧，提取语句内部条件；歧义通过 operation、明确索引表达式或 occurrence 消除。保留原语句，只在它之前采集参数。
- `frontend/access.py` 读取前端 Buffer 的逻辑索引、区域、shape、stride、dtype 和内存级别；T.copy 复用 TileLang 的前端区域归一化。`emitters/access.py` 只将操作数编码为 int64 local 元组并交给现有打印策略，不读取目标元素、不新增同步。
- `source_analysis/bindings.py` 生成循环参数求值计划，插入器统一消费该计划；`frontend/bindings.py` 按实际对象保留原 Var／Buffer 身份，不按 ast.Name 加分支，也不重新命名原对象。
- `protocols/access.py` 从已验证的完整元组生成 accesses.jsonl、access-summary.json 和 access-report.md，区分范围、掩码和采集完整性。偏移为相对逻辑 buffer 基址的推导值，不是物理地址、swizzle 或事务完成记录。
- 数值 reference 只接收数值点。混合采集时，访问报告单独重算校验，不要求 reference 自动解释访问参数，也不把数值匹配当作索引语义证明。

索引／条件中嵌套内存读取、有副作用的调用，以及可能在掩码关闭时未定义的除法，需要先显式绑定为 scalar。原子操作、地址取得和自定义 buffer 宏尚需语义适配，明确诊断，不能误报成普通读。运行仍需要原 kernel 正常完成；插桩不替代原访问或修正其索引。

## 验证环境与结果

H200、TileLang 0.1.12 固定 checkout，沿用统一采集验证环境。工作目录 `/home/ang.gao/tilelang-debugger-source-access-20261009`。

| 测例 | 独立核对内容 | 结果 |
| --- | --- | --- |
| access.py，shift=0／1 | 每个线程、循环的实际索引与掩码；分支到达；while 的 continue/break；global/shared 区域；故意偏移保持原值；混合数值 reference | 182／181 条逻辑请求，synccheck 及 reference 通过 |
| TileOPs MaxPool | 独立枚举 12 个输出的合法 kh／kw 访问，嵌套分支与四维索引 | 70 条，racecheck 通过 |
| TileOPs RoPE | 配对元素索引和预先加载 position 后的 cos 查表索引 | 96 条，racecheck 通过 |
| TileOPs Softmax | 前两块 T.copy 区域、最后一块 256 个候选索引及 255 个关闭的掩码 | 258 条，racecheck 通过 |
| runtime_expansion，三种 view | 每例两个 launch，n=17／39，stride=1／2／0；同编译、关键字传参、调用方 stream；逻辑索引与完整数值 reference | 三例 racecheck 及 reference 通过 |

所有通过的 capture 均检查 baseline／instrumented 输入、调用后参数和输出逐位一致，按独立枚举的期望索引核对访问记录。小测例额外比较前端中的目标 BufferLoad、copy 和同步节点数量，确认没有增加目标读取、搬运或同步；该比较仅用于验收，不是产品准入门禁。改变保存报告中的派生偏移后，独立证据重算必须报错，随后恢复原文件。

`tests/run_cpu.py --output artifacts/cpu-final`：12 个隔离 suite，共 99 项 CPU／TIR 测试通过，无 skip；其中 9 项为新的源码访问选择／报告测试，另有符号循环参数不改名的回归。

## 调试过程中修正的问题

前端区域调用实际名称为 tl.tileop.region，T.alloc_shared 在该版本使用 shared.dyn，现统一由前端适配层读取实际对象事实。动态测试发现循环别名和宏形式参数会重命名前端符号。早期曾绕过 Name 别名，统一方案已撤掉这类源码写法特例：所有参数仍按原顺序求值一次，由同一引用适配器保留已有符号；其他表达式使用正常前端绑定。访问元组通过闭包引用原操作数，避免用宏形参重新绑定。

此前未完成的运行参数扩展也在本轮实际验证：使用 T.StridedTensor 声明前端 stride，保留实际 tensor 传参，按 view 保存／恢复快照，并按 launch 绑定形状。早期失败产物保留，最终通过目录分别为 artifacts/access-verified、artifacts/tileops-racecheck、artifacts/runtime-access-fixed。

## 统一修正后的最终验收

新工作目录为 `/home/ang.gao/tilelang-debugger-source-access-stabilization-20261009`。先执行完整 CPU／TIR 回归，再构建 wheel 安装到独立 installed 目录，随后全部 GPU 验证使用该安装包；逐 capture 核对 environment.json 中的实际包路径。

- 13 个隔离 CPU／TIR suite，103 项测试通过，无 skip。新增真实前端测试验证原 Var 名称和身份、Buffer shape／stride 引用关系、返回已有 Var 的函数调用、求值次数、目标读取数量、copy 区域表示和真实 memory scope。
- H200 共 13 次成功 capture、21 对 baseline／instrumented launch。access 两例和 runtime 三例使用 racecheck；TileOPs 三例及既有数值 pipeline／group／negative／fragment／dynamic 五例使用 synccheck。
- 两个访问小测例分别得到 182／181 条逻辑请求；TileOPs 三例为 70／96／258 条；动态三例各有 n=17／39 两次 launch。独立索引枚举、数值 reference、输出逐位一致和报告扰动校验通过。
- 旧数值五例每例两次 launch，采集和离线 reference 均通过。
- wheel 中 54 个 Python／JSON 包成员与工作区源文件逐字节一致，无旧 trace 根模块。SHA256：`28c36738053848a2b464eccc75d0484505ded3b42e3bfaa93face807fb47eaa7`。

完整摘要在该目录的 artifacts/final-summary.json，本地副本为 artifacts/source-access-stabilized-summary.json。安装包本地副本为 artifacts/source-access-stabilized.whl。此次统一设计与实现由当前 session 完成，没有将历史独立审阅声明为本次审阅。
