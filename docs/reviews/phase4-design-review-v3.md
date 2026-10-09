# 第四步设计独立复审 v3

**结论：PASS（设计通过，无阻塞项）。** 第 13 节可替代 v2 的 FFI hook 路径进入实施；它提供的是受审 host IR 与真实 launch 绑定支持的构造证据，不是 descriptor 内容或构造返回值的运行时直接观测。

- 日期：2026-10-09，Asia/Shanghai。
- 受审文件：`docs/phase4-plan.md`，第 13 节优先。
- 原始字节 SHA256：`70b055d41c21d127cf18e1e13de24a9111b6cc1f530db15e7ea4b0acdcf4a260`。
- TileLang checkout：`2d63708c8ad57196051c4636a1167c5453c73a48`，本轮再次核对。
- 本轮只读方案、实验代码、本地固定源码及远端保存的实验产物；只新增本报告，没有修改实现或启动 GPU。

## 复审依据

读取 `experiments/access_pipeline_probe.py`，以及远端 `/home/ang.gao/tilelang-debugger-phase4-20261009/artifacts/pipeline-probe-gqa` 和 `pipeline-probe-gqa-gc`。两处内部 `validation.json` 均为 passed、descriptors=4、restored=true，但目录没有外层进程状态文件。实验脚本在解释器退出之前写该文件，故它不能证明 worker 成功。本次未重跑实验；两次 exit=1 与 finalize segfault 来自主 session 的执行记录及方案记载，不能由这些内部文件独立重建。

第 13 节明确保留失败、拒绝绕过退出码，并停止使用已暴露生命周期故障的 Python FFI callback。这个处理正确；v2 的设计 PASS 从未免除该实验门槛。

远端 GQA `pipeline.py` 的实际证据支持新路径：

- 第 533 行 host `tma_descriptor_args` 保存 Q/K/V/O 四组构造参数。
- 第 536–547 行把四个 packed 参数位置分别绑定 Q/K/V/O；后续读取 shape，检查 dtype、stride、device 和 byte offset，并形成相应 tensor data 绑定。
- 第 640–647 行依次分配并构造四个 descriptor；第 650 行唯一目标 launch 按 `K_desc, O_desc, Q_desc, V_desc` 顺序传入。顺序不同于源码 tensor 顺序，必须使用方案规定的实参位置关联。
- 第 641–647 行实际构造参数与上述 metadata 的表达式对应。固定 `src/cuda/runtime.cc:540–559` 调用 `cuTensorMapEncodeTiled`，失败进入 `LOG_FATAL`，成功返回结果；不能虚构这个结果曾由产品 hook 读取。

以上为本轮文本/源码核对；正式实现仍须按方案对实际对象逐项执行 structural equality、支配关系与绑定验证，不能以本报告的行号匹配代替验证器。

## 接受理由

第 13 节要求构造 metadata 与真实 host call 参数逐项相等、每 descriptor 一次构造且支配 launch、没有重写、device 参数身份正确，并对未知 host 控制流和表达式拒绝。结合 baseline/pre-trace host 等价、固定源码摘要、真实 tensor shape/stride/data_ptr 绑定和 worker 整体成功，可以在当前四例域内支持“原路径按这些参数成功构造”。无需修改 host 或重复读取 opaque descriptor。

`derived_from_host_ir_and_launch_bindings` 明确区分了 descriptor 参数来源；GPU coords/shared offset/barrier index 仍取原 call 参数并标 runtime，区域展开仍是推导。输出 tensor 返回后的补充核对也是成功报告的必要条件。该修订没有重新打开 B1，也没有把 v1 的 B2 退回仅凭 descriptor 名称猜测参数的状态。

## 非阻塞实施注意

1. 成功必须由外层 runner 在子进程退出后决定；后续 4A 保存真实 `process.json`、stdout/stderr，不能沿用 probe 内部 passed 作为成功依据。
2. host 的 tensor 参数绑定要实际验证 packed 参数索引、shape/stride/data/byte_offset 提取链及动态 scalar 来源；仅识别 Var 名称不满足方案。
3. 对 constructor 参数求值使用受审表达式与原 dtype/cast，保留原表达式、真实绑定值及结果。未知节点、缺构造、descriptor 重写或对应关系歧义均失败。
4. 第 13 节第 5 点的 stride 文案应统一理解为 `cudaGlobalStrides = globalStrideRaw[1:]`，建议改成这个明确等式，避免误读为对已裁剪数组再切一次。
5. v2 第 12.3 节的“构造返回0”只属于已替换的 hook 方式；产品报告不得继续生成 observed constructor return、observed descriptor pointer 或 observed descriptor bytes 等未观测字段。

正式代码复审及 H200 验收仍按既定流程完成；本报告只批准这次证据方式的设计修订。
