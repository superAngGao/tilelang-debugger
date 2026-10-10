# 访问表达式独立设计与代码审阅

日期：2026-10-09。审阅 session：`access_expression_design_review`。

结论：**本审阅范围内通过**。修订后的实现未发现尚未解决的阻塞问题。该结论针对源码候选选择、表达式求值路径、操作参数适配和前端访问操作数检查；最终安装包、双版本完整测试及 GPU 结果由独立验收审阅确认，不由本结论替代。

## 范围与方法

独立阅读 `source_analysis/access.py`、`source_analysis/expressions.py`、`source_analysis/control_flow.py`、`instrumentation/unified.py`、`frontend/access.py` 及相应测试。先提出最小反例，再复核修订代码和新增回归。设计与实施补充均经过审阅后实施；审阅者未修改实现或测试，仅撰写本记录。

审阅期间独立查看固定 TileLang 0.1.12 的 eager AST/builder 源码，并实际构建前端函数验证可变标量的求值时机；没有新增或执行分析用 lowering pass。

## 审阅发现及修订

| 发现 | 修订及复核结果 |
| --- | --- |
| 收集候选时立即校验，安全 occurrence 被后续未选复杂访问阻止 | 先收集、再选择、再验证；不支持的操作参数绑定也保留为局部诊断。安全前项可独立选择 |
| `T.And`/`T.Or` 与链式比较丢失访问条件 | 统一传播访问路径条件，区分条件本身与条件后项 |
| 原子和取址适配按语义参数顺序处理，遗漏关键字源码顺序与参数内部调用 | 按实际参数求值顺序遍历；特殊操作角色不替代子表达式遍历 |
| 普通下标、增量赋值及非下标特殊目标漏掉基对象中的副作用 | 补齐基对象、索引和调用对象遍历；相应未知前缀明确拒绝 |
| masked copy 只检查所选端，另一端可能提前计算除零 | 对传入观察器的双方区域与 predicate 统一检查重放定义域 |
| access 显式 selector 仍受旧语法白名单限制 | access selector 复用统一 operand 分类；按表达式选择与按 buffer/occurrence 选择保持一致 |
| 链式及解包赋值可先修改 local.var，再计算后续目标索引 | 本轮明确拒绝该语句的访问观察，避免静默记录旧索引 |
| 原表达式后续 helper/宏可先向前端插入 local.var 写入，再运行此前构造的 BufferLoad | 增加后续副作用信息；前端操作数含 local.var 加载时拒绝此组合。不可变索引继续支持；所选 copy/atomic 自身不被误算为危险后续调用 |

最后一项通过真实前端复现：`value = x[j] + change(j)` 中，`j` 为 `T.alloc_var`，`change` 发出对该标量的写入。实际前端程序先修改 `j`，随后才读取 `x[j]`。仅凭 Python 参数或二元表达式的构造顺序，不能保证运行时索引值不变。该发现触发设计补充，补充获批后才实施。

## 独立验证与验收边界

- 独立执行 `PYTHONPATH=src python tests/test_source_access.py`：15 项全部通过。
- 独立构建上述 local.var 后续修改反例，核对实际前端语句顺序。
- 复核真实前端回归覆盖宏前缀、宏后缀、不可变索引、链式/解包赋值、既有 Var/Buffer 身份、目标读取数量和编码拒绝。
- 嵌套内存读取、未知操作内部访问、危险表达式重放等明确拒绝仍是设计边界；没有把拒绝改成静默推测。
- 双版本最终安装包、完整 CPU/TIR、GPU 数值与逐请求索引验收不在本次独立执行范围，应参见最终验收记录。

审阅快照 SHA-256：

```text
3ea9e6b685d9e482cf63ca1f77061b72cbd9565e85ad4958d1c3767f89289cb5  src/tilelang_debugger/source_analysis/access.py
d721370c2028bd1d5c3286c218e68b94b1b9a7f352c46ee0e5f91837fc3a5e80  src/tilelang_debugger/source_analysis/expressions.py
22f76a4b68328647921476a4b74684c03ab037bffe7431909b1f4d11ea682496  src/tilelang_debugger/frontend/access.py
a527282645d347d34130d574c7385c61d1afbfd9898d283e8827c1a8486762ea  src/tilelang_debugger/source_analysis/control_flow.py
75485fca479b9db660f243733e971031154e7dc303a1f02e6bd60ca10cf93c4e  tests/test_source_access.py
2ba60f9648f7c12dc407abe9754eb2d04a93b5dbe4207e5fde402c061db77b0d  tests/test_frontend_contracts.py
```
