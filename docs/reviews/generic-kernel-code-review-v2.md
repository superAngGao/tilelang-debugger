# 用户 kernel 源码数值引擎：独立代码复审 v2

**结论：PASS，无剩余代码阻塞项，可进入正式 H200 验收。** v1 两个发现均已修正；设计仍是源码插入既有 Python 打印宏，没有恢复 lowering 解析。

- 日期：2026-10-09。
- 复审独立重跑 `test_source_engine.py`：9 tests PASS；v1 已独立运行共 31 项相关 CPU 测试。
- 本轮没有执行 GPU，三类 smoke 不替代正式验收。

## 修复确认

**B1 关闭。** `examples/tileops/common.py` 从真实 worker `request.json` 获取 mode，instrumented 标记与实际编译产物一致；采集 worker 的示例结果标记 `worker_completed_pending_capture_validation`。独立基线保持未插桩，不自行冒充完整采集通过。

**B2 关闭。** `examples/tileops/capture.py` 接受 run 的 0/2，但继续之前必须通过 `verify_capture` 重新验证完整证据；其他非零直接失败。随后执行离线 analyze，要求 completed；仅两个命令都退出 0 且分析 matched 才标 passed，否则保存 numerical_mismatch 并退出 2。错误计算因此保留采集和逐点分析，而执行失败不被吞掉。

## 本次代码标识

| 文件 | SHA256 |
| --- | --- |
| `source_capture.py` | `e1a777719c1c917e1c58e5c90c4c7d45d6e6bbc28bf7f0c524f2b1e3b6acb53d` |
| `source_instrument.py` | `03f9f6997a4f2c29dbd5880ed05c28f10e3b974a4981644619a5053701527bc3` |
| `monitor.py` | `46a8122900d61ad5462369e334e619969b850a9a5ea1c9d6b3738e7d7bf589f5` |
| `examples/tileops/common.py` | `38c8abb42f555fea0b3160e2bd3caadaac71e2ebc7be6182f7ac0148cd5ef2dd` |
| `examples/tileops/capture.py` | `fe2c3c79d1f0a3e37f2a06aca814596f14f99e9f3791e15d008a1c32f6590c3b` |

正式验收继续覆盖 v1 列出的三类实际采集/中间 reference、shape/dtype/block/serial、sanitizer、源码变更/错误计算、无 reference/异常路径及 reviewed 回归。线程相关 loop bound 和 global storage/源码别名的拒绝应有实际负证据。通用 trace 和 RoPE 中间 scalar 不属于本轮完成范围。
