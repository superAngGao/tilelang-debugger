# 离线报告第一版验证

日期：2026-10-10。基于主分支 `55b01b0` 增加报告功能。此次不新增 GPU 执行，使用之前已完成 H200 验证的 schema 3 采集证据；CPU/TIR 回归和实际浏览器测试覆盖本轮变更。本记录是实施验证，不是独立 session 审阅。

## CPU/TIR 与单元测试

Linux TileLang 0.1.12 环境执行：

```bash
CUDA_HOME=/usr/local/cuda PYTHONPATH=src python tests/run_cpu.py \
  --output artifacts/cpu-final
```

15 个测试套件、126 项测试全部通过，无跳过。其中报告新增 9 项，覆盖整数精度、NaN/Inf、零 reference、操作数缺失、转换 visit 隔离、HTML 字符转义、目录边界、无 GPU 库导入、跨采集分析拒绝，以及 partial/unlaunched 状态。报告单元测试也在 Windows Python 3.14 运行通过。

Linux 工作目录：`/home/ang.gao/tilelang-debugger-report-v1-20261010`。本地汇总：`artifacts/report-cpu-summary.json`。

## 既有证据集成

输入来自 `/home/ang.gao/tilelang-debugger-access-final-20261009/artifacts/v012`，下载到本地 `artifacts/report-inputs/`。

| 报告 | 输入 capture / analysis | 观察点实例数 | 结果 |
| --- | --- | --- | --- |
| mixed | access/shift-0 / access/analysis-0 | 8 | 通过 |
| fragment | values/fragment / values/fragment-analysis | 2 | 通过 |
| strided | runtime/strided / runtime/strided-analysis | 6 | 通过 |
| no-reference | expressions/logic / 无 | 7 | 通过 |

```bash
PYTHONPATH=src python tests/validate_report.py \
  --inputs artifacts/report-inputs --output artifacts/report-v1
```

生成后验证输入证据不变、复制证据逐文件一致、manifest 正确；把整个输出目录搬到临时位置后重新复核证据。损坏 records.jsonl 必须拒绝生成。四份报告全部通过。这里的实例数按 launch 展开，同一个配置观察点可以出现多次。

输入中旧 analysis 没有容差字段，页面明确提示，不猜测；新生成 analysis 会保存 dtype/atol/rtol。报告展示的是已关联的 reference 结果，本轮未重新执行 reference。

## 实际浏览器

Windows Playwright 1.63.0 使用本机 Edge，以 offline 模式打开真实 `file://` 文件：

```powershell
$env:PYTHONPATH='src'
python tests/reporting/browser_report.py --reports artifacts/report-v1 `
  --browser 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe' `
  --output artifacts/report-browser-final
```

四份报告逐个点击全部观察点和数值/访问/IR/上下文页签。验证线程筛选、launch 切换、CUDA 版本切换、矩阵空缺和误差视图、无 reference 提示和搜索空结果；600px 视口无页面横向溢出，1440px 截图保存供查看。所有页面无 JavaScript 错误。

另有一个明确标记为 `synthetic-renderer-only` 的渲染测试：显示 partial、精确显示超安全整数范围的整数，并确保带 `</script>` 的源码字符串不能执行脚本。它仅测试渲染，不是 GPU 采集证据。

## 安装包

本轮构建的 `tilelang_debugger-0.2.0-py3-none-any.whl` SHA256：

```text
72a8f84769d4281df4edc9d98c4d8abf02d928b7f6b8d6a5d8dda85f2215516c
```

使用 `pip install --no-deps --target artifacts/report-installed` 安装，确认导入路径来自安装目录。通过安装包 CLI 生成报告，并重新通过四组证据集成和五个浏览器场景。模板、CSS、JavaScript 均随 wheel 打包；报告导入不加载 torch、tilelang 或 tvm。

本地安装包验证产物位于 `artifacts/report-installed-example/`、`artifacts/report-installed-acceptance/` 和 `artifacts/report-installed-browser/`。这些测试产物不提交 Git。

## 尚未由本轮证明的能力

没有新做敏感算子或转换异常的 H200 故障注入；相关诊断经过单元验证，不声称已经完成协议全部数值异常验收。没有自动根因定位、失败 kernel 日志恢复、任意布局解析、多芯片验收或浏览器全版本兼容性结论。协议对应关系见[报告方案](report-plan.md)。
