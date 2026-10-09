# Component refactor independent acceptance review

2026-10-09. Reviewer: independent `nested_acceptance_review` session.

## Verdict: PASS for the component refactor

The completed evidence supports the behavior-preserving component split. All five new H200 captures passed, and the installed refactored package successfully revalidated 46 previously saved captures. No new control-flow or observation capability is claimed by this verdict.

The reviewer performed CPU-only verification and did not modify the product or run new GPU kernels. Every remote verification process explicitly used:

```text
PYTHONPATH=/home/ang.gao/tilelang-debugger-source-engine-20261009/artifacts/component-refactor/installed
```

The original `src` directory on the remote machine was not used for these checks.

## Wheel and component integrity

- Independently verified wheel SHA256: `e04f5d2502495573f40a6bc55a45aa6c935def52ea9d3bb80fd1098b4400c3f2` for `tilelang_debugger-0.1.0-py3-none-any.whl`.
- All 40 Python members exactly match both the local reviewed source files and the corresponding remote installed files. The wheel includes all five subpackages and their initializers: `source_analysis`, `instrumentation`, `emitters`, `protocols`, and `runtime`.
- Checked actual `__file__` paths for the package, compatibility entry point, analysis, instrumentation, both emitters, both protocols, source loader, and capture state. They all resolve inside the absolute installed directory.
- Independently reran the two component tests through that installation: offline protocol imports avoid GPU/emitter/runtime dependencies; compatibility names and new emitters share the same session state and restore it on normal and exceptional exits.
- Independently regenerated and compared the 11 source plans and injected source texts against `source-before.json`. All are identical, including samples, legacy source and four reviewed examples.

The GPU worker `environment.json` format records TileLang and CUDA provenance, but not the debugger's own module path. The original GPU commands were launched with the installed directory by the implementation session; the independently verified module paths above apply to this review's reconstruction processes. This report does not treat the original summaries as module-path evidence.

## Original evidence and numerical reconstruction

Revalidation used original device stdout, point metadata, persisted records, coverage, worker process and launch metadata, binary snapshot hashes, baseline/instrumented equality, and original sanitizer logs. Point and output references were recomputed on CPU from saved inputs; matrix success summaries alone were not sufficient.

| Evidence under `artifacts/` | Captures | DATA records | Result |
| --- | ---: | ---: | --- |
| `component-refactor/samples` | 3 | 1,141 | nested 149, local 224, Softmax 768: all reconstructed and reference matched |
| `component-refactor/source` | 1 | 514 | RMSNorm fragment path and output reference matched |
| `component-refactor/reviewed` | 1 | 16,384 | GQA tile and complete output reference matched; both racecheck workers clean |
| `nested-final` | 16 | 4,304 | All old samples and coverage reconstructed identically; all references matched |
| `nested-unroll` | 1 | 149 | Original induction coordinates and reference preserved |
| `nested-regression-source` | 27 | 19,745 | All old TLDBG1 source records and references matched |
| `nested-regression-reviewed` | 2 | 20,480 | Both old reviewed records and references matched |

Thus the new wheel reconstructed all 46 historical captures and all five new captures. Existing sanitizer evidence was checked under the capture's declared mode. No compatibility change was required in the saved records or captured source imports.

The new CPU evidence in `component-refactor/cpu` contains 12 isolated suites and 80 tests. Original unittest logs all report `OK`, without skips; suite counts and successful return codes agree with the summary. The reviewer additionally reran the component suite and the 11-fixture source comparison rather than relying solely on their stored results.

## Scope

This acceptance covers module organization, wheel packaging, compatibility imports, shared capture state, unchanged source generation, and preservation of existing capture/reference behavior. Group/Pipelined observations, dynamic bounds, early transfers and further collective strategies remain future work. The control-flow design is not evidence that those capabilities have been implemented.
