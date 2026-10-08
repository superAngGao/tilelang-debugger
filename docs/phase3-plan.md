# 第三步：reference 接入与数值分析

状态：独立设计审阅 v1 PASS 后实施，代码 v1 问题修复后 v2 独立复审 PASS，CPU与H200验证完成。沿用原阶段编号，原第 1、2 步已经合并完成。见 [独立设计审阅](reviews/phase3-design-review-v1.md)、[最终代码复审](reviews/phase3-code-review-v2.md)及[验证记录](phase3-validation.md)。

## 1. 本轮交付

将现有四例 reference 变为可用的离线分析入口。用户提供 Python reference，从本次保存的实际输入计算观察点和最终输出的期望值。比较完整 tile，输出每元素 actual / expected / error / 坐标 / 是否匹配，并生成可直接阅读的 Markdown 摘要。无需重跑 GPU 即可更换 reference 再分析。

继续限定四例受审 kernel、构建参数和观察位置；不扩展源码采集安全范围。第四步的 runtime 索引采集、访问诊断、交互式 UI、任意程序自动 reference 不在本轮范围内。monitor、IR 同步检查、原始 bits 协议和已有布局授权不修改。

## 2. 捕获成功与数值正确分开

当前驱动在 reference 不匹配时立即 assert，baseline worker 会阻止 instrumented worker。这适合验收，不适合调试。

新增主机端 output-reference helper，四例仅替换最终 assert / reference.json 保存部分。正常独立执行 run.py 时仍 assert；在 capture worker 的 TLDBG_OUTPUT 环境中，helper 写出 passed=false、容差、误差与特殊值信息，但不因数值不匹配抛异常。reference 计算异常、shape 不匹配、dtype 不支持及无效容差仍是执行错误。绝不笼统捕获并吞掉 AssertionError 或其他驱动错误。

capture 保留原 run.json status=passed 的“执行、采集、输入输出一致性均通过”语义，新增 numerical_status=passed/failed 表示两版驱动 output reference 结论。两版原始输入必须相同、最终输出必须逐位一致；monitor 改变输出、launch 检查失败、sanitizer 错误、异常退出、超时、记录缺失等仍为捕获失败，不纳入正常数值分析。

运行 CLI 在捕获完成但 numerical_status=failed 时退出 2，所有已完成产物保留；执行失败退出 1。reference passed 不作为 instrumented launch 的前置条件。

evidence.py 提取 verify_capture：重验进程、门禁、快照摘要、两版字节一致、原始日志与完整记录，不要求 numerical passed。原 verify_success 在其上额外要求两版 reference passed，继续用于验收；不得使旧验收把错误结果标成通过。reference.json 必须存在且有布尔 passed，reference 异常不冒充 mismatch。

## 3. 最小 reference 接口

新命令（拟定）：

```bash
python -m tilelang_debugger analyze artifacts/gemm-001 \
  --reference examples/gemm/reference.py --output artifacts/gemm-analysis-001
```

输出目录必须不存在。分析可在有 PyTorch 的 CPU 环境运行，不加载 TileLang，不启动 GPU。reference 是用户显式选择的可信本地 Python 代码，不是沙箱。

provider 是自包含 Python 文件，仅依赖标准库和 PyTorch，定义：

```python
def reference(inputs, points):
    # inputs: baseline 实际输入的 CPU tensor tuple
    # points: points.json 的观察点元数据列表（含 shape、block、原循环值）
    return {
        "points": {
            # 必须覆盖全部选中点，不允许遗漏、多余 key 或广播。
            "gemm_0": {"tensor": expected_tile, "atol": 0.001, "rtol": 0.0001}
        },
        # 与本次输出列表顺序对应，全部覆盖。
        "outputs": [{"tensor": expected_output, "atol": 0.0625, "rtol": 0.002}]
    }
```

以复制后的 CPU inputs 和元数据调用 provider，防止 provider 的原地操作改变原始分析数据；调用一次，立即分离/复制全部 reference tensor。记录 provider 原文件、SHA256 和分析环境。第一版不支持 provider 相对导入本地其他文件；支持标准 import torch/math 等。保存实际期望值，保证报告可追溯，不承诺仅凭单文件哈希能重现任意外部依赖行为。

reference tensor 必须 CPU、非空、实数 strided tensor，shape 与 actual 严格相同；浮点支持 float16/bfloat16/float32/float64。int32 actual 要求 int32 reference，精确比较（atol=rtol=0）。不支持隐式广播、complex、sparse 或整数精度有损转换。

## 4. 比较规则与输出

有限浮点在主机以 float64 做比较：abs(actual-reference) <= atol + rtol * abs(reference)。容差必须非负、有限数值，不接受 bool。没有通过率阈值，任一元素不匹配即该 tensor mismatch。

NaN 默认总是不匹配（包括两边都是 NaN）；同符号 Inf 匹配，其余涉及 Inf 的情况不匹配。正负零数值相等，保留 bits 以便辨认。relative error 为 abs_error / abs(reference)；reference 为零时同值为0，其他为Inf。特殊值差异单独标明，不能被 NaN 汇总吞掉。JSON 不写非标准 NaN/Infinity token，使用字符串 NaN/+Inf/-Inf 或 null（不适用）。

analysis.json：schema、分析完成/失败状态、capture run_id、provider摘要、输入证据摘要、各点/输出的 shape、dtype、容差、计数、最大误差、NaN/Inf 数量、前若干错误及源码身份。区分 completed + matched=false（发现数值差异）与 failed（证据/接口/计算异常）。

elements.jsonl：全部观察 tile 和最终输出的逐元素记录，包含 point 或 output index、行优先多维坐标、actual bits、reference bits、实际值、参考值、abs/relative error、matched 和特殊值分类。提供人可读 report.md：总体结论、各 tensor 摘要、前20个错误坐标/数值及对应源码行/block/iteration。tensor完整数据存JSONL，摘要不打印数万个元素。

分析 CLI 全匹配退出0；分析完成有差异退出2；执行/证据/reference错误退出1。开始写running，成功或失败原子更新状态；失败时保留原因，不能留下旧的成功摘要。保存实际 provider 源文件及其输入 capture 文件的SHA256清单。报告输出独立目录，不改动原 capture。

## 5. 四例 reference

- GELU：x_reg 对照原始输入切片，严格相等；y_reg 直接对照 torch GELU 的 FP32 结果切片，使用已有最终输出容差0.03125/0.02。最终输出同原参考。区分数值误差与 phase1 的采集位保真测试。
- Sum：x_f32 对照原输入转换及显式补零；acc 对照FP32行求和（0.0001/0.0001）；最终输出0.03125/0.002。根据实际shape支持N256与257。
- GEMM：按block选输出tile，按原循环迭代取K前缀（64 * iteration），使用FP32 matmul，0.001/0.0001；最终输出0.0625/0.002。
- GQA：按block及consumer身份取Q/K切片，QK比较0.001/0.0001；最终完整attention比较0.01/0.01，保持已验证的noncausal/softcap=0配置。

reference 在CPU运行，复用既有数学表达式；现有H200验收仍保持独立入口。任何CPU/GPU reference差异如超过既定容差需定位，不自动放宽。

## 6. 代码布局及契约变更

新增 analysis.py（证据加载、provider、逐元素比较、报告）、numerics.py（无GPU依赖比较及驱动helper），四个 examples/*/reference.py，tests/test_analysis.py 及H200验证入口。扩展 cli.py 与 evidence.py，capture.py仅新增数值状态汇总。

四例run.py及Sum无padding驱动的哈希更新，需要独立代码审阅确认只调整主机侧reference处理，不改变输入、编译参数、kernel、launch次数或IR门禁。tests/build_contracts.py生成新驱动摘要，其余源码、观察点与layout契约必须与旧版完全相同。旧capture可离线分析，不因驱动哈希变化失效。

## 7. 验证和完成条件

1. CPU比较测试：有限值容差边界、reference为零、NaN/Inf/负零、int32极值、不合法容差、shape/dtype错误；精确验证多维坐标、错误计数及JSON可标准解析。
2. 四例历史真实capture离线分析：五条路径全部reference匹配，数值可重算；对已保存数据使用刻意错误的reference得到正确错误坐标和退出2。测试不得篡改原始证据冒充真实结果。
3. 错误处理测试：reference抛异常、返回缺失点/多余点/错误shape、改写输入；损坏snapshot/log、失败gate/process/sanitizer必须拒绝。旧verify_success仍拒绝reference失败，新的verify_capture只放行完整可靠capture。
4. H200重跑五条路径racecheck，确认新驱动的两版输入输出逐位一致和记录完整，使用新CLI分析全部产物；GEMM/GQA继续synccheck。核对kernel、monitor、IR gate和layout授权未变化。
5. 数值失败继续采集路径：使用独立测试夹具将输出reference有意偏移（仅测试环境、不得污染正式驱动契约），在真实H200 run中观察 baseline reference=false 后仍完成instrumented、完整records和状态；strict验收必须拒绝。测试夹具如果涉及临时驱动摘要，仅在测试进程替换内存契约，不添加产品绕过开关；独立审阅必须核查注入只改reference，不改kernel、门禁或比较规则。
6. 独立session方案PASS后实施；独立代码审阅、修复及上述验证通过后完成。提交方案、审阅原文、实现、README和精简验收记录；原始大产物留artifacts。
