# 旧 trace 移除记录

2026-10-09。旧 trace 依赖固定样例的源码、行号、形状、descriptor 和 CUDA 契约，未形成通用编译结果分析能力。按产品范围取舍，移除独立 trace 路径，不再将其泛化列为交付缺口。

移除了 `trace`／`_access_worker` CLI、五个 access Python 模块、access-pins、原四例的五份访问配置、专用测试及两个实验脚本。测试运行器不再要求访问 fixture；TileOPs 历史准入探测只调用 reviewed run。TileOPs 的 access.json 是源码定位意图，仍被基线与数值观察点定位复用，保留并明确其用途。

数值 run、reference analyze、reviewed 兼容入口及 IR／CUDA 导出保留。源码“看访问”的专用采集和报告仍待实现；当前元素观察不能冒充原访问追踪。

历史实现及专用测试可在[移除前提交](https://github.com/superAngGao/tilelang-debugger/tree/e838fc88d96a562ee6230ee154d6593bac6632f8) 重现。历史文档增加移除提示，原验收结论仅适用于当时版本。

## 验证

使用该提交加本次清理构造独立快照，排除工作区尚未完成的运行参数扩展，远端目录为 `/home/ang.gao/tilelang-debugger-trace-removal-20261009`。

- `PYTHONPATH=src python tests/run_cpu.py --output artifacts/cpu`：11 个隔离 suite，89 项 CPU／TIR 测试通过，无 skip。比原 99 项少的 10 项来自移除的两个 trace 专用 suite。
- 本地源码路径与 TileOPs 测试共 13 项通过。
- wheel 构建成功，包内无 access 模块及 access-pins；保留数值模块、contracts 和导出实现。
- 直接从 wheel 调用 CLI：主帮助、run 帮助、analyze 帮助正常；trace 与 _access_worker 均按未知命令退出 2。
- wheel 中 source_capture.py 与移除前提交逐字节一致，IR／CUDA 导出未改动。
- 当前代码、测试和实验入口无旧 trace 模块导入残留。

验证 wheel SHA256：`c8bd57740f013391d8a6b9910b44755074d0e37b59f767497288dd76769d732c`。本轮没有重跑 GPU kernel，也没有新增独立 session 审阅；以上为清理后的回归与打包检查，不将旧 GPU 验收冒充本轮结果。
