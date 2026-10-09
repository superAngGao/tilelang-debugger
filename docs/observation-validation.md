# 嵌套源码观察：首批实现与验证

日期：2026-10-09。产品基线 `8996d5f`，本轮新增 samples v2；所有数值插桩发生在 Python 源码／宏层，不新增或解析 lowering。

## 实现范围

- `schema: 2` 配置支持递归 serial/Parallel/if/elif/else，以及 `T.If/Then/Else` 规范化。scope 由原始源码摘要、原行位置和坐标维度标识；每层 induction 保存独立别名，保留同名嵌套绑定。
- 原位 PrimExpr／local Buffer 打印，允许在原元素写入语句之后观察同一 `buffer[index]`；索引必须为纯表达式，不额外读取索引 buffer。没有引入 collective barrier。旧完整 tile 路径和 TLDBG1 保留。
- 每条 TLDBG2 有独立身份、原位 bits 和固定 `|E` 结束标记。int64 分两段 uint32 传输，重组、JSON 和比较保持整数；bool 与 FP16/BF16/FP32/int32 明确编码。
- 根／分支见证支持静态线程／串行域的严格覆盖检查。Parallel 的 execution coverage 始终 unverified；有分支或 thread 筛选时 logical coverage 也保持 unverified。观察到的数值正确不能证明未收到的线程副本不存在。
- samples reference 独立声明 logical 或 execution 键集合；逐个比较所有收到的副本，缺失和多余键报告 mismatch。原 tensor reference 兼容。
- 最大 8 点，65536 事件保守预算（含见证），CUDA printf 参数预算限制最大坐标长度；不固定两层。边界仅允许静态纯整数表达式；未知 context、提前退出和新的 group/pipeline collective 尚不开放。

原始具体方案及其受审 SHA 保留在[方案审阅](reviews/observation-implementation-review.md)。实现中增加的“写后同一元素读回”由[代码审阅](reviews/observation-code-review.md)确认；它用于 Softmax 原语句的标量结果，没有扩展为任意地址读取或访问追踪。

## H200 主矩阵

产物：`artifacts/nested-final`。**16/16 通过，32 个隔离 worker，4304 条 DATA**。每例均重编译 baseline/instrumented、输入输出位一致、运行 reference 通过，再由离线 CPU reference 比较中间值及输出。

| 基础用例 | DATA | 已核对的内容 |
| --- | ---: | --- |
| nested | 149 | 三层 serial、同名内外 i、非零起点、if/elif/else、零次循环、编译期 inactive、Parallel 尾部 |
| local | 224 | 每线程两个 local 元素；bool；大于 2**53 的 int64；FP16/BF16/FP32 |
| selected | 1 | 来源 thread=0，内外同名循环分别选择第 2 次，原始坐标仍独立 |
| not-executed | 0 | thread=1 不进入所选分支；由根与分支见证闭合，非日志缺失 |
| pool | 210 | TileOPs MaxPool2D：Parallel → 外层 if → kh → kw → 边界 if，val／累计 max／bool |
| indices | 106 | 带索引池化：val、严格变大分支内的 int64 max_idx、初始 has_nan |
| rope | 144 | 原 THD position_ids RoPE：pos、val、paired_val，二维 Parallel 及尾部条件 |
| softmax | 768 | N=513、tile_n=256：前两块 full 分支及尾块 masked 分支的原写入元素 |

nested/local/pool/softmax 各再跑 racecheck 和 synccheck，共 8 次；**16 个 sanitizer worker 的原始日志均为 0 error / 0 hazard**。

池化输入为有限值，本轮没有验证 NaN 分支；bool true/false 由 local 测例覆盖。Softmax 原始输出含 768 列 padding，reference 同时验证前 513 列结果和其余零值，不只比较切片。Softmax 样例使用 32 线程，使含见证的保守预算落在限制内。

外部 kernel 均直接导入 TileOPs checkout `95ba6cae78856ec2f610535fde3d44df198d39e8`，没有拷贝改写算子或增加产品算子白名单。每个 worker 保存 `upstream.json`。

## 负例与旧功能回归

- `artifacts/nested-cpu-v1`：11 个隔离 CPU/TIR 套件，**78 项通过，无跳过**；其中 14 项新增作用域／协议／精确整数测试。
- `artifacts/nested-unroll`：内层同名 i 改为 `T.unroll(3,5)`，**149 条**原坐标样本与独立 reference 通过；两个 synccheck worker 为 0 errors，展开未改变观察身份。
- `artifacts/nested-negatives`：**10 项通过**，包括真实记录删除 root／branch／DATA／整个点、重复、截断、乱序重排；int64 reference 错一位时 32 个 wide 样本均报告 mismatch；真实 kernel 计算加 1 时采集保留、数值分析完成并报告错误，两个 CLI 均退出 2。
- 上述负例使用 `nested-matrix-v1` 的小测例证据（产品代码与最终矩阵相同）；初次外部 fixture 的预算拒绝和 Softmax 输出形状错误也保留在该目录。修正的是测试参数及 reference 输出形状，没有放宽产品门禁或数值容差。
- `artifacts/nested-regression-source`：原 **27/27** source 采集与离线 reference 回归通过，含 6 次 sanitizer 采集。
- `artifacts/nested-regression-reviewed`：旧 GELU **4096** 条与 GQA **16384** 条完整 tile 回归及独立 CPU reference 通过；四个 racecheck worker 为 0 hazards。

预算失败、未知作用域或源码副作用边界均明确报错。Parallel 分支样本可以完成数值比较，但报告继续保留 unverified 覆盖项，不把它交付为完整物理线程域的证明。

## 复现

使用 README 中的固定 H200 / TileLang 0.1.12 环境，配置 `PYTHONPATH=src`：

```bash
python tests/validate_nested.py --tileops /path/to/TileOPs \
  --output artifacts/nested-final --sanitizers
python tests/validate_nested_negatives.py --matrix artifacts/nested-final \
  --output artifacts/nested-negatives
python tests/validate_nested_unroll.py --output artifacts/nested-unroll
python tests/validate_source_engine.py --tileops /path/to/TileOPs \
  --output artifacts/nested-regression-source --sanitizers
python tests/validate_gpu.py --output artifacts/nested-regression-reviewed \
  --cases gelu gqa --sanitizer racecheck
```

独立方案、代码审阅和[最终验收](reviews/observation-acceptance-review.md)均为 **PASS**。验收 session 独立重算原始日志、快照、中间值及输出 reference，并核对上述覆盖边界。

本地证据归档：`artifacts/nested-observation-evidence.tar.gz`，SHA256 `929b2b45c20497a1ddb88f054e86d25383eb1c3bddfa9204ff37603405641dfa`。

构建产物：`artifacts/observation-wheel/tilelang_debugger-0.1.0-py3-none-any.whl`，SHA256 `d63ff5b9eec4ca87ab7eb37ddf264d7a9d263e23715af4007f42c364fc23fc17`。已逐字节核对 wheel 内 25 个产品文件与本地 src 一致；没有发布到包索引。原始日志及 wheel 不纳入 Git。
