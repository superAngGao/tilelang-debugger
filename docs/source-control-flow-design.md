# 源码控制流与组件组织

2026-10-09。基线 `5172d3c`。本文件统一后续控制流设计，同时规定本轮保持行为的组件拆分；不宣称动态循环、Group、Pipelined 或提前退出已经接通。

## 1. 统一语义模型

一次解析源码，建立源位置、绑定、结构化作用域树和必要的控制转移边。观察点查询这个模型，不再各自扫描 AST 并罗列禁止节点。模型不依赖 lowered IR，也不推导寄存器布局。

| 结构 | 统一表示及职责 |
| --- | --- |
| Kernel / Function | 建立执行根；区分宿主构建代码、macro 和设备函数的语义阶段 |
| Sequence | 保留语句顺序、读写绑定和观察位置；前置兄弟语句的跳转也影响可达性 |
| Branch | 保存原条件、短路与各 arm；合流绑定保守合并，不重新求值条件 |
| Loop | 身份、绑定、迭代域、入口／回边／正常出口；serial/Parallel/unroll/Pipelined 是属性与适配器，不是独立嵌套算法 |
| Group | 保存明确的参与域及其来源；不把组 id 当成物理线程区间 |
| Transfer | break 指向最近循环出口，continue 指向最近循环下一次迭代，return 指向实际函数／macro 边界；不统一成“退出当前 scope” |
| Point | 源码位置、before/after 和对象；由模型计算到达路径和动态身份 |

结构化树负责嵌套；正常、分支、回边和提前退出的边负责控制转移。无需建立通用编译优化器或全程序分析器。未知 helper/context 保留不透明效果，只在影响所选点的路径、绑定或协作参与域时给出能力诊断。

### 循环域与身份

循环域分为 static、runtime-bounded、open/unknown。静态域可以枚举；动态边界在原求值位置处理，不能像当前静态实现一样统一搬到 Kernel 入口。

后续 adapter 必须保持条件／边界的求值次数与顺序。可直接复用前端已经构建的值；若必须源码绑定快照，应在原位置只求值一次并让原语句与记录共用。无法证明该替换保持 DSL 语义时，只记录实际 body visit，覆盖保持 unverified，不能为记录重新读取条件或移动副作用。

counted loop 保留原 induction value 和原始 ordinal；while／重复访问同一位置另需本次父 scope 实例内的 visit ordinal，不能用 induction value 代替唯一身份。ordinal 在进入迭代体时递增，continue 不得跳过递增；外层重新进入时子作用域 visit 重新绑定到新父实例。动态实例键包含源 scope/binding id、父实例、必要的 thread/group、原坐标和 visit。

Parallel 保留逻辑坐标与物理 thread 两种身份。Pipelined 的原逻辑迭代与 stage/slot 是不同字段，不能用 slot 取代迭代号；不修改用户原循环调度参数，也不声称插桩保持性能。

### 分支与提前退出

分支见证留在实际 arm。提前退出在原转移之前记录 transfer 的目标和当前父实例；不要在 break/continue/return 后插入必执行收尾。return 表达式有求值／副作用时，只有前端 adapter 能保证表达式值只求一次并保留原返回行为，才能提供完整的退出见证。

提前退出的影响由 Sequence 和转移边决定：例如循环前部的条件 continue 会使后续观察点不执行，而另一不相关函数内的 break 不应禁用该点。跳转跨越多个作用域时，记录其真实目标，可由模型推出被退出的作用域集合；不能伪造逐层正常结束。

## 2. 语法支持、打印能力和覆盖证明分开

每个点分别得到三份结果：

1. **前端语义**：该构造在固定 TileLang 版本的此种上下文中是否合法、绑定和转移如何解释。
2. **打印策略**：对象真实类型、memory scope、所需参与线程及数据就绪条件是否允许该 emitter。
3. **覆盖证据**：收到的记录能否证明该逻辑域／执行域完整；未知覆盖不自动禁止原位 scalar/local 打印。

已有 PrimExpr/local 直接打印只需要当前实例可执行。fragment/shared 协作打印另检查完整祖先路径对实际参与域是否一致，以及数据是否就绪。Group 改变参与域时重新检查整条祖先路径；不能清空已有分歧，也不能机械沿用全 CTA 的分歧结论。reader 筛选与参与域分开。

源分析只做常量／已知绑定／有限纯表达式依赖分类；不知道则保留 unknown。不能为了证明任意用户程序引入通用符号执行。源码作用域模型也不能凭空证明异步写入完成或 printf 不丢记录。

### 固定前端的实际差异

已阅读当前固定 TileLang checkout 的 `tilelang/language/eager/ast.py`、`builder.py` 和 `language/loop.py`：while/break/continue 有前端处理；return 根据 macro/控制流上下文存在限制；具体组合仍需基线编译与 H200 验证，不能从 Python AST 能表示就宣称设备支持。

`T.Pipelined(start, stop, num_stages, ...)` 的第三参数是 num_stages，不能照搬 serial 的 step 解析。循环 adapter 按固定前端函数签名解释参数；Group adapter 也须读取已有明确声明的域及同步契约，不猜测 `T.ws` 的数字含义。

## 3. 记录和预算的统一接口

从模型生成最小所需 enter/visit/arm/transfer/exit/data 事件计划。事件自带父实例身份和序号，不依赖线程间 printf 顺序。静态域仍用现有闭合规则；动态域只有具备根、visit、所有相关跳转和退出证据，且序列／计数一致时才能升级完整性。首版无法证明完整动态域时，记录样本并保留 unverified。

BEGIN/END 自身不足以发现整实例消失，动态计数也不能仅由收到的 DATA 回推。提前 return、超时、预算耗尽必须分别报告。后续预算机制只能停止调试输出，不能向用户循环插入 break 或改变计算；耗尽时保留不完整状态，不能截断后标通过。新事件需要版本化，现有 TLDBG1/TLDBG2 读取不变。

## 4. 组件目录与依赖

本轮实际拆分如下，旧模块保留薄的兼容导出，供已有脚本和已保存插桩源码继续导入：

```text
src/tilelang_debugger/
  source_analysis/
    normalize.py          # 语法规范化，不执行用户表达式
    scopes.py             # 当前 samples 源位置、绑定与作用域分析
    legacy.py             # 旧 full-tile 源码分析，待迁移到统一模型
  instrumentation/
    samples.py            # 根据分析结果插入别名、见证和打印调用
    legacy.py             # 旧 full-tile 源码插入
  emitters/
    samples.py            # PrimExpr/local 原位编码和输出
    tiles.py              # 已验证的 fragment/global 打印与同步实现
  protocols/
    dtypes.py             # 共用位宽定义，不依赖 GPU/emitter
    samples.py            # TLDBG2 解码与覆盖校验
    tiles.py              # TLDBG1 解码与完整性校验
  runtime/
    source_loader.py      # 真实包路径导入、临时 hook、恢复
  capture_state.py        # worker 内 observation/session 绑定状态
  source_capture.py       # 已有编译/运行/证据生命周期，暂不机械移动
  analysis.py             # 已有 tile reference 分析
  sample_analysis.py      # 已有 samples reference 对齐
  scopes.py, monitor.py, sample_monitor.py,
  source_instrument.py, records.py, sample_records.py  # 兼容入口
```

调用方向：运行调度 → 源码分析 → 插桩改写；插入调用在前端构建时进入 emitter → 共用 session 状态／协议类型。运行后的日志交给 protocols → reference 分析。离线 protocols 不得导入 emitter、运行 hook、TileLang 或 CUDA。

后续控制流模型、构造 adapter 与有限依赖分类放入 source_analysis；事件计划及源码变换放入 instrumentation；对象与同步策略放入 emitters。解析组件不生成 printf，插桩组件不再独自解析循环语义，emitter 不重新走 AST，协议解析器不推导源码控制流。

本轮不建立空的未来模块或通用插件框架。打印类型／控制流种类的登记按上述目录扩展，组合通过同一上下文递归完成。

## 5. 实施与检验

本轮先完成目录重构，产品行为、配置、注入后的源文本、记录协议、范围和 CLI 不变。旧导入名继续有效；拆分 session 时确保 samples/tile 共享同一状态，不复制模块全局变量。

实施前独立审阅本设计。重构验证：对原三种入口（旧 source、samples、reviewed）的实际 fixture 比较重构前后的分析结果与注入源码；现有 CPU/TIR 套件、离线已保存证据重验；H200 至少跑 nested、local、Softmax samples、旧 source fragment、reviewed GQA，并核对原始日志／reference。wheel 必须含子包。再独立代码及验收审阅。

后续能力按模型落地：先在统一模型中接通 Group/Pipelined 普通 samples，再接动态 visit/transfer 协议及协作对象策略。每轮均先落实 adapter/协议，再独立审阅、实现和测试；本轮目录重构不冒充这些能力已经完成。

控制流新增验收矩阵以语义规则为单位：负／非单位步长，外层相关动态边界，while 重复坐标，前置兄弟 continue/break，多层退出，macro return 边界，Group 外层分歧与组内一致，pipeline 多 stage 同 slot 不同迭代，短路条件不重复读取。每种规则验证不同对象策略的必要交互；保留现有静态路径作为回归。
