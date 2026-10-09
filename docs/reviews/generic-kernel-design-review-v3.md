# 用户 kernel 数值采集：独立设计复审 v3

**结论：PASS，无剩余设计阻塞项，可以实施源码打印宏方案。** v2 初审 FAIL 保留；本报告关闭其两个阻塞项，不恢复已撤销的寄存器直读/lowering 分析路线。

- 日期：2026-10-09。
- 受审方案：`docs/generic-kernel-plan.md`，含“v2 审阅补充：源码层确定参与域”。
- SHA256：`23bf84fe0e414238d5c2aa55d86d383c232953c905f70c6ca7a2884e6dfacd3d`。
- 本轮仅复审设计，没有修改产品或执行 GPU。

**B1 已关闭。** 每层 serial 的 start/stop/step 作为宏参数在前端求值，全部必须是静态整数，步长非零，所选 ordinal 在真实 range 中存在；拒绝线程/运行期相关上界及所选循环内 break/continue/return。结合观察点位于 Kernel 内且在 if/Parallel/pipeline/ws 之外，明确了本轮全 CTA 宏的参与条件。无需读取 lowering IR。

**B2 已关闭。** global 输入的 Name 仅允许标量 Subscript 读取，禁止赋值/切片/地址/alias/不透明调用；整个目标函数仅接受明确的 DSL/纯数学调用，未知 helper/extern/atomic 拒绝。实际前端 Buffer 身份必须对应输入。launch 前还核对选定输入与其他输入不得共享/重叠底层 allocation；输出由非空 out_idx 自动分配，显式输出/in-place 不支持。这覆盖源码别名与 driver 实参别名，无需做通用内存分析。

其余已接受设计不变：复用 fragment→shared→全 CTA 同步→原位 printf→同步；正常包导入/JIT；不以用户源码/driver/布局摘要准入；无 reference 为 not_checked；数值不匹配与执行失败分开；旧 reviewed 与 trace 限制明确保留。RoPE 本轮只打印只读输入，不冒充中间 scalar 或索引。

代码审阅需核对这些支持域检查在 launch 前执行。H200 仍须取得三类实际采集、选点 reference、racecheck/synccheck、源码变更/错误计算、无 reference/异常路径及 reviewed 回归证据。本设计 PASS 不替代实施验收。
