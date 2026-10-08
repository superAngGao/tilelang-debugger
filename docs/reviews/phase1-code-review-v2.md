# 阶段 1 独立代码复审 v2

结论：**FAIL，剩余两个 launch 前检查缺口**。v1 的未知调用、原同步协议遗漏、GEMM fence 覆盖和已有证据复核问题已修复；正式 GQA 修正版及其 GPU 证据也已核对。剩余问题是 staging 元素配对关系未检查，以及 monitor 自身 collective 的循环上下文未检查。无需扩大到任意 kernel。

审阅日期：2026-10-08。本轮读取新版实现、测试、方案第 12/13 节、设计 v4 审阅，以及本地正式 artifacts；只运行 CPU 检查，没有启动 GPU，没有修改实现。报告针对本轮读取版本：

- `ir.py` SHA256：`756561FD3D953EF1DEB7A45DC494975AF7F79397DF6FECD4AA9F472C9D2C42EF`
- `evidence.py` SHA256：`5FE0CB27210DB13C07B117F51AF691FAD75ED9A8C9212BA058DE974A182D6ABF`
- `capture.py` SHA256：`DCDEFFF1F2578845C7318CB44FF816ECD580E9A53244262523E782E022A4B8C4`
- `tests/validate_gpu.py` SHA256：`95847B07D7A05AAB7BAD77601357FA3E3075719A7B7C782BF432B41A84FF1D52`

实施方在收到本轮发现后已开始下一轮修复。本文件保留上述版本的审阅结果，不因后续并行修改而覆盖。

## B1：分别证明源和目标覆盖，仍不能证明正确的元素配对

位置：`src/tilelang_debugger/ir.py:324`，尤其 `source_counts` 与 `counts` 的两个独立比较。

新版已经拒绝越界源索引、额外算术、错误 allocation 和 printf 读地址，修复了 v1 负例。但它将目标地址统计为一个 Counter，将 `(writer tx, local index)` 统计为另一个 Counter，两者之间的配对关系被丢弃。完整置换能通过所有检查，同时把正确 fragment 元素记录到错误逻辑 index。

具体负例可直接基于 `tests/test_ir.py` 的真实 TIR fixture：保持 local、shared、barrier、printf 和全部元数据不变，仅把默认 store 的目标索引从 `tx` 改成 `127 - tx`。

```python
f = fixture()
old = f["events"][0]["expr"]
tx = f["variables"]["threadIdx.x"]
f["events"][0]["expr"] = T.BufferStore(old.buffer, old.value, [127 - tx])
check_instrumented(**f)
```

两边独立 Counter 仍完全相同；打印仍按 `scratch[idx]` 输出，故门禁接受的 tile 被反向排列。该结论来自原函数控制流和 Counter 运算；本机没有 TVM，未将这段负例冒充本轮真实 TVM 执行结果。它也是 v3 “正确逻辑元素”要求仍未闭合的直接例子，不是要求通用布局推导器。

最小修复：保留并验证 `(目标逻辑/物理 index, 实际 writer tx, 源 local index)` 的配对。可以将四例已数值验证的实际 lowered 配对摘要纳入固定契约，未知布局拒绝；也可从实际可用的布局元数据推导。不能只把当前运行动态生成的摘要与自身比较。为完整置换增加负例，保留归约代表 writer 的正确路径测试，并对已验收 IR/CUDA 与绑定的映射作可追溯关联。

## B2：monitor barrier 的循环上下文可绕过参与域检查

位置：`src/tilelang_debugger/ir.py:198`、`:203`、`:283`。

原程序 protocol 已保存 guard 和 loop 上下文，这是正确修复；但 monitor 两个 barrier 被提前加入 `excluded`，各自只检查 count 和 `participants()`。后者仅求值 guards，不检查 `event["loops"]`。因此 monitor 自己的循环次数、最小值、变量和一致性没有进入任何拒绝条件。

同样可基于默认真实 TIR fixture，只给首个 barrier 附加一个 `For(j, 0, tx + 1, SERIAL, ...)` 的循环上下文，保持它的 guards 为空。`participants()` 仍返回全部 128 线程，两个 barrier 仍被从 protocol 中删除；printf 的独立元素循环检查也不受影响。这样的实际 IR 会让各线程到达 collective 的次数不同，不能等 launch 后超时再处理。即使额外循环是常数，两次 monitor barrier 与记录区间不匹配也不应静默放行。

最小修复：要求两次 monitor barrier 的完整循环上下文等于 point 的受审 enclosing serial loops，包含顺序、变量、min、extent 和 loop kind；验证所选迭代处于其范围。printf 只能在相同上下文上多一层已经检查的完整元素循环。staging/新增 fence 的 enclosing 上下文也必须与该观察区间一致，再允许把 monitor 操作从原协议比较中剔除。增加线程相关 extent、额外循环和两端循环不一致的拒绝测试。

## 已关闭的 v1 问题

- **同名 acc_s 绑定**：插桩对实际词法作用域做唯一 allocation 改名，前端记录实际 buffer，最终 allocation 要求唯一匹配，再对 staging/fence 使用对象身份核对。考虑到 lowering 克隆 Var，这一桥接方式在固定四例范围内可接受；两 consumer 的实际 CUDA 分别使用 `tldbg_buffer_0` 和 `tldbg_buffer_1`。
- **原同步协议和未知调用**：全部嵌套 Call 经过固定 intrinsic/extern 集合检查；原 Evaluate 协议保存参数、guard、loop，剔除经过认领的 monitor 操作后比较。CUDA 显式 named barrier 与 IR 交叉核对；AllReduce helper 的 0/1/2 资源保留，相关 header 固定摘要。v1 的 wait/loop 丢失和未知 extern 默认放行已关闭。
- **GEMM 完整 fence**：实际 CUDA local allocation 为128个 float32；wait后的原 fence64保留，新增 monitor fence128位于 staging 前，使用同一指针和 offset0。GQA 两组相应覆盖64。源码合同、实际分配与检查一致。
- **已有证据假阳性**：`verify_success` 现在检查成功状态和请求模式、执行计数、gate、进程退出、两版 reference、实际快照字节摘要、两版一致性、原日志解析及记录数量；验收 summary 原子更新 running/failed/passed。v1 的失败状态被重新标通过问题已关闭。本轮真实运行了其失败路径单测。

## GEMM 附加 proxy fence 例外

第13节说明的实际变化在产物中成立：`acceptance-final/gemm/instrumented/launch-gate.json` 记录一处无参数 `tl.fence_proxy_async`，guard 为 consumer 域，位于四轮 `ki` 循环；CUDA 中该 fence 出现在循环开头，monitor引入的 generic shared stores仍在该循环中。原始 wait、mbarrier、named barrier和其他协议操作没有被该例外整体忽略。

对固定受审四例，将这一**附加内存顺序 fence**与完成wait区分、要求无参数、证明域并保存差异是可接受的。当前协议匹配仍要求所有原操作按原顺序存在，不能用新增 fence 替换或删除原 wait。没有为此发现新的独立阻塞；B2 的monitor自身循环检查仍必须补齐。

## 独立 CPU 与本地 GPU 产物复核

本轮执行：`test_cpu.py` **7项通过**、`test_evidence.py` **3项通过**。已阅读5项真实TVM测试；本机无TVM，未重复执行，实施方报告的5项通过与本轮亲自运行的10项明确区分。

直接调用新版 `verify_success` 重查以下七个case，全部通过：

- `artifacts/acceptance-final`：GELU、Sum padded、Sum unpadded、GEMM、GQA；两版共10份racecheck零hazards，快照字节、运行/gate状态和原始记录均完整。
- `artifacts/synccheck-final`：GEMM、GQA；两版共4份synccheck零错误，同样完成快照与记录复核。

没有只信任旧summary：本轮又从baseline输入/输出字节及原始TLDBG日志，在CPU用NumPy独立重算全部五例reference和tile。

| Case | CPU独立输出reference最大绝对误差 | 采集tile最大绝对误差 |
| --- | ---: | ---: |
| GELU | 0.007768715920039693 | 0 |
| Sum padded | 0.00731658935546875 | 0 |
| Sum unpadded | 0.00292205810546875 | 0 |
| GEMM | 0.0229644775390625 | 0.00002288818359375 |
| GQA | 0.00017395615577697754 | 两组均0.0000095367431640625 |

全部通过预定容差，未放宽阈值。CPU与原PyTorch参考在GELU/GQA上的微小误差差异来自参考计算路径；不是修改已保存运行数据。五例baseline源码与当前受审example逐字相符。

另独立核对：

- `fidelity-final` 四dtype各128条原始记录等于expected bits，输入/输出文件逐字节相同，四份racecheck均零hazards。
- `print-regression` 八份原始stdout重新解析后都恰为256元素×3次且值正确；auto/patched四例零hazards，disabled/surround四例各3 hazards；execution均记录输出相等及helper恢复。负对照并未被当成工具失败或错值。
- `gqa-epilogue` 原件三份stderr确有527/352/345个数值不匹配及NaN，修正版三次reference成功，六份racecheck均零hazards。正式源码相对保留原件只有四条同步语句和两条说明注释。正式GQA两版都使用修正版，barrier3/4保留为基线协议，monitor另用15/14；原件没有被重新标成成功。

因此 v1 B4 所要求的正式GPU数值、同步和保真证据已闭合。此次仍判FAIL是因为 B1/B2 的前置拒绝缺口，**不是声称已验收的这些kernel出现了新的误采或死锁**。修复后需要证明新增门禁接受这些已核实布局、拒绝两个负例，再完成受影响复验；保留本轮证据与报告供下一轮对照。
