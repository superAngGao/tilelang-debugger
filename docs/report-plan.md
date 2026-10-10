# 离线报告第一版

报告把现有 schema 3 采集、可选 reference 分析和编译产物组织成可交付的离线浏览目录。它不改变源码插桩、打印协议、同步策略或 lowering；不重新执行 kernel 或 reference。

## 输入与输出

输入是 `run` 目录，以及可选的 `analyze` 目录。执行 `report CAPTURE --analysis ANALYSIS --output REPORT` 后生成自包含 `report.html`、结构化 `report.json`、SHA256 `manifest.json`，以及 `evidence/capture/` 和可选的 `evidence/analysis/` 副本。文件名及目录含义保持固定；报告目录可整体移动。

`report.html` 内嵌样式、脚本、模型及源码/IR，无 CDN 或 fetch；单独复制 HTML 仍可查看，外部证据下载需要整个目录。JSON schema 为 `tilelang-debug-report-v1`。大于 JavaScript 安全整数范围的整数编码为十进制字符串，raw bits 为十六进制字符串，NaN/Inf 使用显式字符串，避免浏览器舍入。

## 组件职责

```text
src/tilelang_debugger/
  diagnostics/
    values.py           位模式解码、误差与数值统计
    operations.py       源码操作描述、显式操作数检查
  reporting/
    load.py             复核采集证据、关联已保存分析
    model.py            JSON 可移植数值编码
    build.py            将源信息和各类结果组织为报告模型
    render.py           证据副本、单文件 HTML、manifest
    templates/report.html
    assets/
      report.css
      app.js            总览、导航、共享筛选与分页
      source.js         源码与观察点关联
      values.js         原值/reference/误差与矩阵
      accesses.js       逻辑访问参数
      compiler.js       baseline/instrumented IR 与 CUDA
tests/
  reporting/            单元测试和浏览器验收脚本
  validate_report.py    既有 H200 证据的离线集成验证
```

源码插桩和控制流解析继续由已有组件管理；报告只用 AST 描述已选语句和有限的直接操作，不新建一套 kernel 语义分析器。

## 页面

顶部总览分开显示采集完整性、reference 结果、launch/点数量、非有限值和访问数量。协议覆盖、环境配置、操作检查可以展开。主体为观察点导航、源码定位、明细页签。执行筛选保留 launch、block、前端线程、嵌套循环坐标和序号、visit，不靠日志顺序匹配。完整性和摘要默认针对整个观察点，表格明确显示筛选后的条数。

矩阵按一次执行身份分组；缺失元素不补零，高维索引按扁平顺序展示。大矩阵最多显示 2048 个格子并提示，完整记录保存在模型和分页表格中。int64/uint64 原值与绝对误差由 Python 精确计算；超出浏览器安全整数范围的原值显示精确字符串，使用中性色，不用浮点强制着色。

## 与协议要求对应

| 协议要求 | 第一版提供的证据 | 边界 |
| --- | --- | --- |
| 2.1 范围、等级及结构化报告 | 配置、选区、预算、层级、memory scope、源码和操作描述 | 展示层的摘要/明细不替代分层采集验收 |
| 2.2 数值偏差 | 原值/reference、绝对/相对误差、容差、匹配结果、矩阵 | reference 由用户提供；旧分析没保存容差时明确标注 |
| NaN/Inf 产生传播 | 各观察点数量、位置和执行上下文 | 只说明观察分布；源码行顺序不证明运行顺序或因果 |
| 转换精度 | 直接 `y = T.cast(x, dtype)` 的显式 before/after scalar 配对及变化量 | 需同一行、同执行身份、明确操作数；变化不自动判错，也不证明误差放大 |
| 数值敏感操作 | `/ // %` 近零除数、sqrt 负输入、rsqrt 非正输入和非有限输入 | 仅检查语句前显式 scalar/local.var 观察；复合表达式缺操作数证据则未采集；条件内操作不一定执行 |
| 索引、范围、形状和布局 | origin/extent/shape/stride/active、边界分类、逻辑偏移、IR/CUDA | 不解析物理地址，不自动判定布局正确，不将 copy 尾部相交等同非法访存 |
| 2.3 上下文、统计及导出 | 原始配置、执行身份、统计、原始位、两版前端/设备 IR 和 CUDA | 不建立任意 IR 节点到源码行的映射，不自动分析编译变换 |
| 多硬件项目要求 | 记录实际设备；当前本组工作范围为 H200 | 其他芯片由合作组负责，不声明全项目多后端验收完成 |

## 证据与失败语义

1. 复用 `runtime.unified_evidence.verify`，重放原始日志检查，复核源码、编译产物、快照和访问派生结果。
2. 可选分析必须有相同 run_id、匹配采集文件清单、完整 launch/观察点覆盖和逐样本身份；实际值必须吻合 raw bits。新分析保存 dtype/atol/rtol，报告复算匹配状态；旧分析保留既有结果并提示缺少容差。reference 值仍是用户保存的分析结果，不代表报告独立证明 reference 正确。
3. 复制前后比较原目录和副本清单，避免生成过程输入变化。已有输出不覆盖；输出不得嵌套于输入或包含输入，证据内符号链接拒绝。
4. partial、truncated、unlaunched、无 reference、无操作数分别保留；空筛选和缺失元素不表示通过。损坏证据拒绝生成。
5. HTML 中 JSON 转义脚本结束字符，源码和用户文本使用 textContent。浏览器不执行证据中的 Python/reference。

本版本支持已有完整或合法 partial 采集，不支持失败 kernel 的日志恢复，不自动补插观察点，不提供任意程序 reference 或自动根因推断。协议要求在页面中的映射是证据索引，不是整体验收结论。
