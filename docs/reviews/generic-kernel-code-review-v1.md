# 用户 kernel 源码数值引擎：独立代码审阅 v1

> 历史记录：旧 `trace` 已从当前产品移除；文中 trace 命令、专用测试和兼容性描述仅适用于[移除前提交](https://github.com/superAngGao/tilelang-debugger/tree/e838fc88d96a562ee6230ee154d6593bac6632f8)。保留当时结果，不代表当前产品能力。

**结论：FAIL，两处示例集成问题需修复；本轮未发现需要改变已通过的源码打印宏路线的问题。** 本报告不因尚未执行正式 H200 矩阵而拒绝代码。

- 日期：2026-10-09。
- 范围：新 `source_instrument.py`、`source_capture.py`，monitor/CLI/evidence/analysis 改动，TileOPs common/capture/capture_reference，以及新测试。
- 独立执行：source_engine 9、source_path 5、cpu 7、evidence 5、access 5，合计 31 tests PASS。
- 本轮只读产品，未执行 GPU。RMSNorm smoke 来自主 session 状态更新，不作为本报告独立验收证据。

## B1：instrumented worker 的示例产物被标为未插桩

`examples/tileops/common.py` 已在 `TLDBG_OUTPUT` 下导出实际编译对象的 IR/CUDA，并写到 worker 的 `example/` 子目录。但成功结果仍固定为 `instrumented=False`、`debugger_status=not_run`，包括 instrumented worker。这与实际执行和导出内容不符，后续审阅可能误把这份 IR/CUDA 当成 baseline。

请从真实 worker 请求获取 mode/engine，记录是否插桩及运行上下文。示例 helper 不负责最终设备日志完整性验收，因此可以使用“由父采集器验收”的状态，不应自行宣称完整 capture 通过。独立运行的基线仍应为未插桩。

## B2：数值不匹配会让 capture 示例跳过离线分析

`examples/tileops/capture.py` 对 run 命令使用 `subprocess.run(..., check=True)`。当前 driver reference 正常完成但不匹配时，父 run CLI 按设计退出 2；此处立刻抛异常，后续 reference provider 和 analyze 都不会执行。正是需要查看错误计算时，示例反而没有完整的逐点分析报告。

请区分正常采集的 mismatch 与执行失败：run 返回 2 时先验证捕获完整证据、状态及 numerical_status，然后继续离线 analyze；其他非零或证据不完整保持失败。最终保留数值 mismatch 的非零退出语义，不能把 run=2 或 analyze=2 改报 passed。

## 核心实现已核对

- source 准入不读源码/driver/layout 哈希白名单，AST 插入原 buffer 的 capture 调用，不重命名计算变量。
- `capture_source` 从实际 Buffer/KernelLaunchFrame 读取元数据，serial 完整域转静态整数并核对 ordinal，fragment 复用原宏且采用全 CTA 同步；global 独立只读分支保持原位编码。
- 全局源语法限制和真实前端输入 Buffer 身份校验存在；实际 launch 前检查选定输入与其他参数 storage 不重叠。
- 正常包/相对 import 的 hook 按真实 origin 工作，绕开选定文件的 pyc 并提供 linecache；finally 恢复 hook/path/argv/cache，原文件未改。compile 与 JIT 实际 compile 绑定都被覆盖，代理保留内核属性。
- positional tensor shape/dtype、out_idx 归一化、一次 compile/launch、原输入快照和两版输出一致继续校验。普通 driver 无 reference 被明确记录 not_checked，旧 schema 不接受这种缺省状态。
- 新代码原样导出前端/设备 IR/CUDA，没有新增 lowering hook、layout 解析或晚期 IR 日志。trace 与显式 reviewed 路径保留。
- 新中间 reference 覆盖 Softmax 的 max/sum/第三轮 tile_sum、RMSNorm 的 sumsq/rrms/normalized；RoPE 明确只比较真实输入 x。

## 正式验收仍需覆盖

线程相关 loop bound 应在构建时失败；global 输入源码别名/实际 storage 别名应在 launch 前失败。真实包/JIT、同文件 driver、无 reference、driver 异常、源码空行/改名、合法错误计算的完整采集与 mismatch，以及三类规定 shape/dtype/serial/block 和 racecheck/synccheck 不能用此次 CPU PASS 替代。旧 reviewed GELU/GQA 需要回归。
