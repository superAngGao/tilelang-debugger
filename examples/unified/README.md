# 统一采集机制测例

这些 kernel 不含 debugger 调用。测试脚本从源码语句生成 schema 3 配置，通过公共 CLI 插桩，再用独立 CPU reference 检查中间值、参数变化和返回值。

| 文件 | 覆盖 |
| --- | --- |
| `kernel.py` / `run.py` / `reference.py` | Parallel 尾部、Pipelined、显式 Group、二维 CTA/local、动态边界、while+break/continue、负步长、fragment/shared/global 选区、in-place；每例两次 launch |
| `advanced_kernel.py` / `advanced_run.py` / `advanced_reference.py` | 组内 reader 与参与者分离、流水线 fragment、多重循环/分支、重复值、末尾 return、helper compile 与 alias、13 dtype × scalar/local/fragment/shared |
| `pipeline_copy.py` / `pipeline_copy_reference.py` | 真实 global→shared→fragment 的三 stage 流水线，自动 TMA warp specialization；活跃及编译期 inactive 多桩 |

从仓库根目录执行：

```bash
python tests/validate_unified.py --output artifacts/core --sanitizer racecheck
python tests/validate_unified_advanced.py --output artifacts/advanced --sanitizer synccheck
python tests/validate_unified_advanced.py --types --output artifacts/types
python tests/validate_unified_pipeline_copy.py --output artifacts/pipeline-mixed --mode mixed
python tests/validate_unified_pipeline_copy.py --output artifacts/pipeline-inactive --mode inactive
```

每个脚本生成配置、capture、analysis、命令日志及 summary。advanced 包含预期失败：scalar 与 collective 预算截断必须保持 partial；越界观察必须跳过额外读取并失败，不能把它们计为普通成功采集。完整性扰动和独立审计见仓库验证记录。

`pipeline_copy` 的源码线程为 32，固定编译器自动添加 128 个 TMA producer。协议核对的仍是独立已知的 32 个**前端线程**；不将额外 producer 冒充用户所选线程，不需要解析 lowered IR。
