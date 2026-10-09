# 组件拆分验证

日期：2026-10-09；基线 `5172d3c`。本轮只重组源码分析、插桩、打印、记录协议与源码加载；不改变配置、注入源码、记录格式或能力范围。后续完整控制流模型见 [source-control-flow-design.md](source-control-flow-design.md)。

## 行为等价

重构前保存 `artifacts/component-refactor/source-before.json`，重构后比较 11 组完整分析结果与插入后的源文本：nested/local、四个外部 TileOPs 场景、旧 source 两层循环 fixture、reviewed GELU/Sum/GEMM/GQA。Windows 源码运行和 Linux 实际安装 wheel 均完全一致。

[独立代码审阅](reviews/component-code-review.md)另外比较了 21 个移动函数的 AST，除模块引用与原有 schema 分发的归属调整外保持等价。旧 `instrument.py`、契约数据路径和 reviewed IR 检查没有移动。旧入口仅转发，新旧打印器共享 capture_state 的同一个 session。

新增两项行为测试：旧／新导入共用状态、退出与异常恢复；在新进程中仅导入协议解析时，不加载 emitters/runtime/TileLang/TVM/torch。

## 安装包与运行

重构 wheel 安装在独立目录，不覆盖远端旧 src 或原 TileLang。所有本轮 CPU/H200 命令的 `PYTHONPATH` 指向该安装目录；注入代码仍引用原兼容入口，因此真实编译也检验了兼容路径。

产物根目录为 `artifacts/component-refactor`：

| 验证 | 结果 |
| --- | --- |
| `cpu` | 12 个隔离 CPU/TIR 套件，80 项通过，无跳过 |
| `samples/nested` | 149 条，嵌套身份、分支／inactive／零次域及 reference 通过 |
| `samples/local` | 224 条，local／bool／int64／浮点原位与 reference 通过 |
| `samples/softmax` | 768 条，真实 full/tail 分支写后元素采集与 reference 通过 |
| `source/rms-n257-float16` | 514 条，旧 source fragment 路径及 reference 通过 |
| `reviewed/gqa` | 16384 条，旧 reviewed 组内打印及 CPU reference 通过，两个 racecheck worker 均 0 hazards |

共 5 次 H200 采集、10 个 worker、18039 条 DATA。独立验收使用 installed wheel 再重建这 5 份原始证据，并重验旧 46 份 capture（16 nested + unroll + 27 source + 2 reviewed）及全部中间／输出 CPU reference，均通过。单纯目录移动没有改变已有记录的解释。

外层执行命令明确把 `PYTHONPATH` 设为独立 installed 目录。原 worker 的 environment.json 尚未记录 debugger 自身模块路径；独立复核进程另验证了各组件 `__file__` 均来自 installed，二者分别作为运行命令和安装包复核证据，不伪称原环境文件已经包含模块路径。

wheel：`tilelang_debugger-0.1.0-py3-none-any.whl`，SHA256 `e04f5d2502495573f40a6bc55a45aa6c935def52ea9d3bb80fd1098b4400c3f2`。独立核对 40 个 Python 成员与本地受审 src／实际 installed 逐字节一致，5 个子包齐全。未覆盖原安装环境、未发布包索引。

最终审阅见[组件独立验收](reviews/component-acceptance-review.md)；该结论限于保持行为的组件重构，不证明未来完整控制流／协作对象策略已经实现。

独立方案、代码和验收均 **PASS**。证据包 `artifacts/component-refactor-evidence.tar.gz`，SHA256 `bdb311ee15493c4512005d5eb80cae117846e5a30dbaddaaccea347909ab66e1`；包含新测试、安装包与实际安装目录、重构前的源文本基线。被重验的历史证据仍在上一轮归档中。

复现（先在固定 H200 环境准备好独立 wheel 安装目录）：

```bash
export PYTHONPATH=/absolute/path/to/installed-wheel
python tests/validate_nested.py --tileops /path/to/TileOPs \
  --output artifacts/refactor-samples --case nested --case local --case softmax
python tests/validate_source_engine.py --tileops /path/to/TileOPs \
  --output artifacts/refactor-source --case rms-n257-float16
python tests/validate_gpu.py --output artifacts/refactor-reviewed \
  --cases gqa --sanitizer racecheck
```

前后快照对比在同一个 checkout 路径下执行（Windows 加 `python -X utf8`）：

```bash
# 重构前的版本
python tests/validate_component_refactor.py --tileops /path/to/TileOPs \
  --baseline artifacts/source-before.json --create
# 重构后／安装包
python tests/validate_component_refactor.py --tileops /path/to/TileOPs \
  --baseline artifacts/source-before.json
```

这份重构辅助脚本在 `5172d3c` 尚不存在；需要先带入该脚本再生成基线。本轮在两端均先以旧产品生成 baseline，再改用重构产品比较，没有用重构后的实现自建期望值。
