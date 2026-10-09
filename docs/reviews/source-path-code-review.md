# 用户指定源码路径：独立代码审阅

**结论：PASS，无代码阻塞项，可以进入 H200 验证。** 当前支持的是用户指定已有受审源码的位置；源码/driver 契约及外部包/JIT 限制仍在。

- 日期：2026-10-09。
- 范围：`source.py`、两个 CLI/执行入口、两套 prepare、来源报告、更新后的 TileOPs 探测及 CPU 测试、README。
- `source.py` SHA256：`f8a3e4833199b8a3c0fda80c949f764555b7e1bfe4338c6dca4233f4fa98e9c9`。
- `tests/validate_tileops.py` SHA256：`34f5761a2e9df098b45e1a134709f14d1957f25000c6549643b638f99a872998`。
- 独立执行 5 个 suite：source_path 5、tileops_integration 8、cpu 7、evidence 5、access 5，合计 30 tests PASS。
- 本轮没有执行 GPU 或修改实现。

## 核对结果

1. `run` 和 `trace` 均向实际入口转发 `--source`。共享 loader 先读配置，再按显式 cwd 路径或配置目录相对路径选择文件；没有 driver sibling 推断或失败后回退。测试验证选定文件内容到达实际 prepare 调用。
2. effective 配置携带解析后的绝对路径；原始配置、原输入字符串和选择方式独立保存在 `source.json`。两个 prepare 已解除名称限制，但内容/driver gate 未放宽。
3. 两个入口都将读入的源码变量写入内部执行快照；capture 的插桩也基于这同一源码。原路径进入 run、request、point，再由相应报告显示。access manifest 通过 `dict(p, ...)` 保留新字段；旧产物通过 `kernel.py` 显示 fallback 兼容。
4. `probe_config` 从实际上游 AST 来源取得行号并生成现有字段结构，CLI 命令显式包含上游 `--source`。只有准确的 prepare 契约异常分类 unsupported；旧 FileNotFoundError、无关错误、超时、崩溃和未验证成功继续失败。
5. RoPE 的 y 全局 buffer 请求明确为准入探测，未假称 fragment 语义受支持。`--probes-only` 没有基线，完整汇总非零退出，不会借空集合标成功。
6. README 正确解释内部 `source/kernel.py` 别名与仍未解决的 driver/import/JIT 限制。

## 非阻塞注意及实测门槛

- 当前 prepare 拒绝发生在创建输出目录和保存 `source.json` 之前，所以这类早期拒绝没有产品级 provenance 文件；OSError 显示路径，而契约拒绝本身仍是通用消息。TileOPs 探测依赖外层保存的命令与请求追溯来源。后续可将选定路径加入契约错误上下文；不要声称每一种失败均已写出 run/source 元数据。
- H200 必须执行改名且移至独立目录的 GELU `run`、`trace`、离线 `analyze`，核对原路径、baseline 原文快照、数值/reference 和输出位一致；当前 CPU PASS 不能替代这项验证。
- 保留原调用兼容验证，复测六次 TileOPs 探测实际到达内容/driver 契约门槛。旧文档中的 sibling 缺失证据应保留为历史结果，新验证记录说明拒绝阶段已经推进。
- 代码没有新增源码/算子/CUDA 白名单项；不将本次修正表述为已支持任意用户 kernel。
