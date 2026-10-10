# 用户 kernel 数值采集：源码插桩修改方案 v2

> 历史记录：旧 `trace` 已从当前产品移除；文中 trace 命令、专用测试和兼容性描述仅适用于[移除前提交](https://github.com/superAngGao/tilelang-debugger/tree/e838fc88d96a562ee6230ee154d6593bac6632f8)。保留当时结果，不代表当前产品能力。

本版替代 v1 寄存器直读方案。按用户纠正，本轮不设计、不挂钩、不解析 lowering pass；不读取 layout_map，不追踪 lowered buffer 身份，不插晚期 IR 日志。设备 IR/CUDA 仅原样导出供查看。既有 trace 保持原实现，不纳入本轮通用化，也不作为 run 的依赖。

## 已确认问题

当前 run 依赖源码/driver/观察点/布局摘要白名单，worker 将源码改名为 kernel.py，且漏掉 JIT compile 绑定，导致外部模块无法接入。限制来自工具自身。

打印实验已完成：默认自动同步和内部局部 patch 正对照无 hazards；关闭自动同步且只加外围同步的负对照有 hazards，见 phase1-validation.md。monitor.py 的 Python 宏已有四 dtype 位模式、H200 racecheck/synccheck 证据。本轮复用 fragment→shared→同步→单线程原位 printf→同步，普通 CTA 用全 CTA sync；旧分组同步保留受审路径。无需重做同步设计判断。

## 修改规则

1. run 默认 source 引擎，--engine reviewed 保留旧实现；trace 不变。不按摘要自动回退。旧 GEMM/GQA 命令显式 reviewed。source 摘要只用于来源证据及运行中变化检查。
2. 沿用 --source/config.source 和 line/when/buffer/block/loops。AST 在原语句前后插 capture 宏；不改计算、不重命名 buffer、不插属性/地址标记。用原循环变量条件选择次数，不从 lowered 循环恢复。
3. 选点在 T.Kernel 内、Parallel 等分布式循环外；最多两层 serial/range，常量 start/step。拒绝 kernel 内条件分支包围观察点、while、pipeline/ws；含显式异步矩阵/warp specialization 的目标函数留给 reviewed。不证明原算法正确或未初始化值合法。配置点必须真实构建；重复/缺失记录失败。
4. 宏展开直接从 Buffer/KernelLaunchFrame 得到 shape/dtype/grid/threads/block 绑定。支持静态 local.fragment、FP16/BF16/FP32/int32、一维 CTA。最多 8 点、65536 元素，构建时检查 grid/预算。打印采用既有 Python 宏结构，不添加 lowering。
5. 无 fragment 时允许只读 global 输入，由全 CTA 位置的线程 0 原位打印；前端参数/out_idx 确认是输入，源码直接写或 copy 目标拒绝。RoPE 只采输入 x，不声称采到了中间 scalar 或访问索引。全局输出/shared/Parallel 内 scalar 暂不支持。
6. 普通 SourceFileLoader hook 精确匹配指定 origin，保留 package/spec/file/相对导入；内存改写 source 和 linecache，原文件不动。source=driver 可直接执行。未加载选定模块或点未构建则失败，不寻找 kernel.py 兜底。
7. 新 worker 禁用缓存，拦截 tilelang.compile 和 tilelang.jit.compile；代理保留 artifact/属性。一次编译一次 positional tensor launch；out_idx int/list 可含负数，归一化后拒绝重复/越界/空列表，核对实际 Buffer shape/dtype。scalar 参数/多编译/多 launch 暂拒绝。finally 恢复 hooks/sys.path/argv/linecache 并保存计数/恢复状态，异常传播。
8. CLI -- 后传 driver 参数。两版独立进程固定 seed；输入/输出逐位一致、日志完整性、实际 sanitizer 完成日志继续核对。driver 无 reference 明确 not_provided / numerical_status=not_checked，可离线 analyze，不能声称数值正确。仅新引擎 schema 允许此状态，旧证据校验不放宽。
9. 示例 helper 在 TLDBG_OUTPUT 环境写 worker 结果/reference；数值不匹配正常完成使父 CLI 退出 2。任意 driver 异常/非零退出是执行失败，不吞掉。SystemExit(0/None) 正常结束。
10. 保存两版源码、frontend/device IR、CUDA、配置、环境、原始日志和 tensor 快照。source 不调用 ir.export/check_instrumented 的布局解析。明确不提供静态同步正确性证明。

## 审阅及验收

先独立设计审阅至 PASS 再实施；代码审阅修正后 H200 验收，最后独立复核证据。

CPU：源行/循环条件、普通包及相对 import、同文件 driver、异常恢复、错来源拒绝、参数分隔、out_idx、参考状态及记录完整性；旧 CPU 回归。

H200：Softmax row_max/row_sum/第三轮 tile_sum，RMSNorm sumsq/rrms/normalized，RoPE 输入；至少两种 shape、FP16/BF16/FP32、非零 block/serial 迭代。独立 CPU reference 核对选点和输出，racecheck/synccheck。源码空行/改名/合法错误计算仍采集，reference 报 mismatch；无 reference driver 为 not_checked，异常 driver 失败。旧 reviewed GELU/GQA 回归。

完成要求是通用 source 数值入口真实采集通过，不再仅交方案或 baseline。外部 trace 和 RoPE 中间 scalar 留待后续，不以输入打印替代其验收。

## v2 审阅补充：源码层确定参与域

- 每层 serial 的 start/stop/step 都作为宏参数在前端构建时求值；必须是静态整数且步长非零，按 Python range 确认所选 ordinal 存在。线程变量或运行期数据导致无法求值则拒绝；拒绝所选循环内 break/continue/return。因此打印同步的迭代条件是 CTA 一致的，不仅检查 start/step。
- global 输入只支持简单可检查源码：该 buffer 的全部 Name 使用必须是标量 Subscript 读取，不允许作为赋值目标、切片、取地址、alias 或传给不透明函数。目标 prim_func 的调用限于 Kernel/Parallel/serial/ceildiv/if_then_else/cast 等明确 DSL 构造及纯数学调用；未知 helper/extern/atomic 等拒绝 global 选点。前端真实 Buffer 身份还必须匹配非 out_idx 参数。RoPE 输入符合此域；不将这些规则称为任意 Python 程序的无写证明。
- global 采集 launch 前还核对 tensor storage：选定输入与其他输入的 storage 地址范围必须不重叠（保守拒绝共享底层 allocation，即使 view 不重叠）；输出由非空 out_idx 自动分配。禁止显式输出/in-place。防止 driver 用同一 storage 的另一个参数写入被观察输入。
