# 样例来源与调整

新增 [TileOPs 外部示例](tileops/README.md) 放在 `examples/tileops/`，直接加载指定 checkout 的 Softmax、RMSNorm、RoPE。下面四个已有回归样例保持原样；外部示例的数值基线与调试接入状态单独记录。

四个 kernel 来自 `tilesight-delivery-docs-20260923/examples/kernels` 的 TileOps 独立样例，保留 [LICENSE.TileOps](LICENSE.TileOps)。GELU、Sum、GEMM 的算法源码未改；`run.py` 缩小问题规模，保留至少两个 block，GEMM 有四轮 K 迭代，GQA 有三轮 KV tile。

GQA 原件保留于 `gqa/kernel.original.py`。正式 `gqa/kernel.py` 仅在两个 consumer 的 `T.copy(acc_o, Os[group,:,:])` 后加入：

```python
T.fence_proxy_async()
T.sync_threads(3, 128)  # 另一组为独立 ID 4
```

这两行分别在两组执行，共四条新增语句；每个写线程先发布 shared 写入，再由组内 barrier 保证 leader 发起 TMA store 前所有写入完成。原算法、线程划分、named barrier 1/2 和 mbarrier 协议保留。对应的 NVIDIA 协议说明见 [异步拷贝文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)。

`experiments/gqa_epilogue_probe.py` 从原件生成两版，在同一 seed 和 racecheck 环境比较。此次原版连续三次 reference 失败（含 NaN），修正版三次通过；六次 racecheck 均报告零 hazards。因此这是数值对照与生成代码支持的修正，不能说 racecheck 已捕获原缺陷。

`monitor.json` 由受审锚点生成，行号指向原始 `kernel.py`。维护者可用 `tests/build_contracts.py` 重新生成，但更新源码/驱动摘要必须重新审阅；该脚本不是跳过契约审查的用户入口。Sum 另有 `run_unpadded.py`，覆盖 N=256 的非 padding 路径。
