# TileOPs 外部示例设计独立审阅 v1

> 历史记录：旧 `trace` 已从当前产品移除；文中 trace 命令、专用测试和兼容性描述仅适用于[移除前提交](https://github.com/superAngGao/tilelang-debugger/tree/e838fc88d96a562ee6230ee154d6593bac6632f8)。保留当时结果，不代表当前产品能力。

**结论：FAIL，1 项验收设计阻塞项。** 目录、上游调用范围与 21 组合成立；需要收紧 CLI 缺口证据的分类和退出规则后再实施。

- 日期：2026-10-09。
- 受审方案：`docs/tileops-integration-plan.md`。
- 方案 SHA256：`2849f99140e6c16c82d9d5268e59b1df5f4e257d9dfa429fe20b9d0bd63c5eae`。
- 核对上游：本地 TileOPs `95ba6cae78856ec2f610535fde3d44df198d39e8`。
- 本轮只读方案、上游 kernel/reference、当前 CLI 入口，只新增本报告；没有执行 GPU 验证。

## B1：接入失败必须分类，不能把任意非零退出都归为 unsupported

方案规定每类至少调用一次公开 CLI，并报告 `debugger_status=unsupported`，但没有规定如何识别这一状态，且未覆盖值采集与访问采集两个入口。这会允许依赖缺失、参数解析失败、错误配置、非 Linux 平台、超时或进程崩溃被误记为预期功能缺口，默认基线模式仍成功。

源码证据：`capture.run` 第 170–175 行与 `access.run` 第 77–81 行均先检查 Linux，再读取 driver 同目录 `kernel.py`，之后才读取配置；当前三个外部示例应在 Linux/H200 上产生目标路径明确的 `FileNotFoundError`。这只证明 sibling source 入口缺口，尚未实际触达源码摘要、layout/CUDA 白名单，不能把后者写成运行时观察。

修改要求：

1. 每类分别探测 `run` 与 `trace`，保存准确命令、工作目录、原始日志、退出码和超时标志；新意图配置与当前 CLI 适配探测输入应明确区别。
2. 仅把已识别的入口能力缺口分类为 `unsupported`，并记录阶段及具体原因。当前可识别条件是 Linux 上读取该示例目录下不存在的 `kernel.py`；若入口未来改变，不应把任意新的错误继续算作同一缺口。
3. 配置错误、依赖错误、超时、崩溃、其他异常分类为 `failed/error`，默认基线验收也非零退出。`--require-debugger` 对 `unsupported` 同样非零。缺失结果不能被成功汇总覆盖。
4. 意外零退出不能自动升级为 debugger 支持：必须核验实际采集产物与结果；本次范围内可标记需要重新审阅的未确认状态并非零退出。

## 已确认可接受

- Softmax 的两个 single 参数与 tiled N=513/tile_n=256 符合真实工厂定义。输入原 N，输出分别为 N_padded/total_cols；有限输入的 softmax padding 应为零。三个形状乘三个 dtype 为 9 例。
- RMSNorm 工厂确实要求 padded 输入与 weight，分母用原 N。两个形状乘三个 dtype 为 6 例。
- RoPE 原始 `_make_rope_neox_1d` 没有尾部 guard，计划的两组各 2048 元素且 block_size=1024，整除条件成立。两个形状乘三个 dtype 为 6 例。保留原标量路径正确。
- 通过真实包导入、检查 `__file__` 并保存来源，比复制 kernel 更适合这个外部接入测试。上游 `tileops.kernels.__init__` 会导入其他 kernel；由此产生的兼容/依赖失败应保留，不能制造空包绕过而仍宣称真实包导入成功。
- 不增加新算子白名单、不修改产品模块、将基线与 debugger 状态分开，范围表述诚实。本报告认可的是外部基线与缺口证据这一子任务，不是对原先通用用户 kernel 支持问题的关闭。

## 非阻塞实施注意

- 在执行前固化 dtype 容差和随机种子；记录误差、有限性、padding 与 dtype/shape，不能只输出 allclose。Softmax/RoPE 上游测试有 FP32=1e-5、FP16=1e-3、BF16=1.6e-2 的 atol/rtol 可作为依据；RMSNorm 的 FP32 需明确自定值。
- Softmax 同工厂中 softmax/log_softmax 分支都有同名嵌套函数和相同语句。定位器须识别所选 op_kind 分支及循环上下文；零匹配/歧义时失败，不能默默取第一条。
- reference 针对本次原始 kernel 边界：RoPE 使用真正传入的、已量化 cos/sin 表；RMSNorm 用未 padding 的有效 N。没有复用上游中间 reference 的证据，不应声称已验证中间采集。
- 三种 sanitizer 需要检查实际错误摘要，并让 sanitizer 错误触发非零退出，不能只依赖默认进程退出码。全部基线通过不等于工具可调试这些新 kernel。
