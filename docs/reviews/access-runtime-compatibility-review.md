# Source access runtime and compatibility independent review

Date: 2026-10-09. Reviewer: independent session `access_runtime_independent_review`.

**Result: PASS for the reviewed runtime, protocol, parameter-expression and frontend-compatibility changes. No remaining blocking finding in this scope.** This is a code-review result; the final installed-wheel GPU matrix is being checked separately by acceptance review.

## Scope

Reviewed the uncommitted source-access and runtime expansion changes in `runtime/unified_capture.py`, `runtime/unified_evidence.py`, `runtime/parameters.py`, `runtime/point_bindings.py`, `runtime/tensor_views.py`, `protocols/access.py` and the associated analysis/snapshot path. The follow-up review covered `frontend/compatibility.py`, both supported frontend region names in `frontend/access.py`, public-print-module fingerprinting in `capture.py`, and the new parameter/frontend contract tests.

Checked launch argument binding and snapshots, per-launch symbolic point resolution, view aliases and overlapping bytes, access tuple completeness/coordinate identity, derived logical bounds and offsets, and offline evidence recomputation. Failed-kernel partial-evidence preservation remains the explicitly deferred work item; this review does not approve that unimplemented behavior.

## Findings and resolution

1. **P2 — incorrect truncating arithmetic. Resolved.** The original parameter evaluator mapped TIR `Div` and `Mod` to Python floor division and remainder. Independently reproduced with TileLang 0.1.12: for `n = -3`, frontend `Div(n, 2)` is `-1` while the evaluator returned `-2`; frontend `Mod(n, 2)` is `-1` while the evaluator returned `1`. Such expressions can affect otherwise positive dimensions or grid extents. The evaluator now separates truncating and floor operations, retains integer dtype/Cast semantics, rejects division by zero and undefined signed overflow, validates variable ranges, and preserves unsigned wraparound.
2. **Test oracle defect. Resolved.** The first narrowing-Cast test asked the TileLang Analyzer to simplify out-of-range unsigned constants. The 0.1.12 Analyzer itself rejects such values, so it cannot supply that oracle. The final tests use independent byte interpretation for narrowing casts, while retaining real frontend Analyzer comparisons for division and remainder.
3. **Nonblocking resource observation. Deferred.** `restore_storage` allocates data and validity bytearrays sized to the original backing storage. A small view into a large allocation can therefore require substantial CPU memory during evidence verification. This does not invalidate the tested values or alias semantics; optimizing sparse overlap validation is outside this change.

The suspected missing contiguous-input check was not promoted to a finding: the default TileLang Cython adapter validates ordinary contiguous tensor declarations. Explicit strided declarations are covered by the runtime expansion tests.

## Independent verification

- Read both complete CPU/TIR summaries under `/home/ang.gao/tilelang-debugger-access-followup-20261009/artifacts/cpu-012-final` and `cpu-015-final`: each contains 14 passing suites and 117 tests. Read the parameter and frontend-contract logs for both versions; all passed.
- Independently reran `test_runtime_parameters.py` against the final installed package at `/home/ang.gao/tilelang-debugger-access-final-20261009/installed`, using each version's Python interpreter. Both runs passed all 5 tests, covering frontend arithmetic, narrowing casts, invalid arithmetic, unsigned wraparound, and overlapping cross-dtype views with conflicting-byte rejection.
- Verified the following reviewed source files match the final installed-wheel files byte for byte by SHA-256:

| File | SHA-256 |
| --- | --- |
| `runtime/parameters.py` | `21d3490e9b1d077898c8f7ed3289d3551372c5075eab19104a6310630024269c` |
| `frontend/compatibility.py` | `022f2d1844a4edffb24c180a9c2e217ffde9b7598dee47d87777676e72ac70d5` |
| `frontend/access.py` | `22f76a4b68328647921476a4b74684c03ab037bffe7431909b1f4d11ea682496` |

Compatibility probes establish the required frontend interfaces and copy-region behavior, not blanket support for arbitrary TileLang releases. The reviewed evidence covers TileLang 0.1.12 and 0.1.15; final GPU acceptance is a separate requirement.
