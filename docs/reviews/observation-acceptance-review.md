# Nested observation independent acceptance review

Date: 2026-10-09. Reviewer: independent `nested_acceptance_review` session.

## Verdict: PASS for the scalar/local increment

The final H200 matrix completed before this verdict: 16/16 nested runs, the additional unroll run, 27/27 existing source captures, and both reviewed GELU/GQA captures passed. This accepts the scoped implementation, not every combination in the broader system design.

I read and reconstructed the evidence on `tileops-dev`, under `/home/ang.gao/tilelang-debugger-source-engine-20261009/artifacts`. Verification used the raw device logs, point metadata, binary input/output snapshots, process and launch records, sanitizer logs, and independently generated CPU references. It did not rely on matrix summaries alone. No new GPU runs or product edits were performed by this reviewer.

## Evidence independently checked

| Evidence directory | Completed evidence | Independent checks |
| --- | --- | --- |
| `nested-final` | 8 base runs plus 8 sanitizer runs; 4,304 DATA records | Reconstructed all records and coverage; checked snapshot hashes and bitwise baseline/instrumented equality; recomputed every point and output reference on CPU |
| `nested-unroll` | 149 records; baseline and instrumented synccheck | Confirmed the captured source uses `T.unroll(3, 5)` and original induction coordinates survive; recomputed all references |
| `nested-regression-source` | 27 captures; 19,745 records | Revalidated raw evidence and recomputed all intermediate/output CPU references |
| `nested-regression-reviewed` | GELU 4,096 and GQA 16,384 records | Revalidated raw evidence and both racecheck workers per case; recomputed intermediate and complete output CPU references |
| `nested-cpu-v1` | 11 isolated suites, 78 tests | Checked original unittest logs and summary: all passed, no skipped tests |
| `nested-negatives` | 10 positive/negative controls | Checked protocol mutations, exact int64 mismatch report, the actual wrong-kernel source mutation, and completed numerical mismatch reports |

The eight base DATA counts are nested 149, local 224, selected 1, not-executed 0, pool 210, indices 106, RoPE 144, and Softmax 768. The nested matrix contains 16 clean sanitizer worker logs; unroll adds two; the source regressions contain 12 and the reviewed regressions four. Each sanitizer log reports a completed clean run, and worker command metadata names the corresponding tool.

## Semantic and negative checks

- Independently checked the small fixture's integer formulas against raw decoded records. The outer and inner variables both named `i` retain distinct original coordinates: outer 2/3, middle 1, inner 3/4. Selecting each `i` loop's second iteration yields exactly thread 0, coordinates `[3, 1, 4]`, value 104. The unroll variant preserves the same coordinate set.
- Verified zero-trip and build-time inactive points have no DATA and justified `not_executed` status; the inactive point is separately `inactive_at_build`. The selected non-entering thread case has zero records with closed execution coverage. These are not inferred merely from empty logs.
- Verified all 32 bool samples directly against thread parity, and all 32 int64 values against `9007199254740993 + thread` using integer arithmetic. The incorrect reference differing by one reports exactly 32 `wide` mismatches; other local points remain matched.
- Reversed the actual nested log and obtained identical parsed records/coverage. Independently removed a root, a branch witness, one whole scalar instance, and all events for a point; duplicated a DATA record and truncated its terminator. All six corruptions were rejected. The CPU suite additionally checks Parallel coordinate loss versus replica loss and contradictory/out-of-domain events.
- Checked that the wrong kernel actually changes the arithmetic by `+1`, rather than merely forcing a status. Capture remains usable with `numerical_status: failed`; offline analysis completes with 64 mismatches at `value`. The stored command results are the expected exit 2.
- Inspected the real TileOPs fixtures and references. MaxPool references enumerate window prefixes independently; RoPE references enumerate logical elements without assuming a thread mapping; Softmax observes the same element immediately after its write, covering two full tiles and the masked third tile. Expected key sets are not derived from received DATA.
- Every Parallel branch case retains unverified logical and physical execution coverage even when its independently specified numerical keys all match. Numerical success does not promote coverage. Static serial/local cases have complete execution coverage.

## Acceptance boundaries

This increment validates source scalar values, thread-private local buffers, and constrained scalar element read-back immediately after the same source element is written. It does not validate general arbitrary memory probing, new collective fragment/shared/global strategies, group/pipeline observation, dynamic loop bounds, early exits, or complete physical thread/replica coverage under Parallel. Existing reviewed GELU/GQA success is a regression result, not evidence that the new generic strategy supports their collective observations.

The current MaxPool inputs are finite. They exercise running maxima, boundaries and index-update branches, but do not establish NaN-path behavior. Both bool values are covered by the separate local fixture. The selected RoPE case uses identity rotation while checking intermediate source values; it is not broad RoPE numerical conformance testing.

The four core product SHA256 values match the independently reviewed code:

| File | SHA256 |
| --- | --- |
| `scopes.py` | `b4db95b0f165f0355f1b3fba25c55c93388fc72801516efa38073891b4c146cb` |
| `sample_monitor.py` | `60cf489d387798b8cbe7902dcd101966b130feefbb17ee3e1b390b688001a926` |
| `sample_records.py` | `c280824045d7ae8f4580246eef738b011614d64a37298b734e1f326efa5c8f77` |
| `sample_analysis.py` | `2c69d710c4f8522675a197e4da0af46a13d4c5e2a3071b0206dbc17a0fcf8da8` |

## Documentation and evidence package

Reviewed the final `README.md` and `docs/observation-validation.md` against the evidence. Counts, source-level implementation, exact dtype handling, constrained write-back observation, conservative coverage states, and remaining collective/group/pipeline limits agree. The validation document explicitly separates finite MaxPool testing from untested NaN paths and preserves the initial fixture failures rather than presenting those attempts as passes.

Independently checked the remote evidence archive SHA256: `artifacts/nested-observation-evidence.tar.gz` = `929b2b45c20497a1ddb88f054e86d25383eb1c3bddfa9204ff37603405641dfa`.
