# TileOPs 外部示例代码独立复审 v2

> 历史记录：旧 `trace` 已从当前产品移除；文中 trace 命令、专用测试和兼容性描述仅适用于[移除前提交](https://github.com/superAngGao/tilelang-debugger/tree/e838fc88d96a562ee6230ee154d6593bac6632f8)。保留当时结果，不代表当前产品能力。

**结论：PASS，无剩余代码阻塞项，可以进入 H200 验证。** 这不是数值基线已通过或 debugger 已接入的结论。

- 日期：2026-10-09。
- 复审 `tests/validate_tileops.py` SHA256：`477be303887ef457e5f1857244c177490c70f676794956e92e9cf15eb2abe825`。
- 对应测试 `tests/test_tileops_integration.py` SHA256：`65bd11f33610c55a03b2e0b29fb05aed977e977641eaa4ceb1f6a8bb28a00c7f`。
- 再次独立执行 CPU 测试：8 tests PASS。

v1 的 B1 已关闭：`debugger_status` 按 `run`/`trace` 分别匹配 `capture.run` 的 `source_file.read_text` 与 `access.run` 的 `driver.with_name(...).read_text`；仍要求精确缺失路径、正确入口模块、FileNotFoundError 尾行和不存在的 sibling 文件。测试分别覆盖两条 traceback，并验证入口交叉匹配失败。超时、崩溃、无关错误和未验证的零退出继续失败。

补充加固已独立核对：`baseline_status` 要求完整的必需证据文件集合并逐个核对摘要，拒绝空清单与目录穿越文件名；`suite_status` 对空 probes 明确失败。新增第 8 个测试覆盖空证据清单，原测试增加空探测断言。两项均不改变真实上游 kernel 路径。

v1 中 README 链接建议是审阅者误判，已明确撤回：`../../docs/tileops-integration-plan.md` 实际解析到仓库内正确文件，无需修改。观察点 `applies_to` 保持计划意图，不表示已判断执行分支。

其他实现沿用 v1 审阅范围。H200 需核对 21 个基线组合、每类三种 sanitizer、六次实际 CLI 缺口探测及严格模式非零退出，保留未通过项。普通基线成功不得表述为已采集上游 kernel 中间 tile/runtime 索引。
