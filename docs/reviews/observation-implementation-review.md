# Nested observation implementation plan review

Date: 2026-10-09. Independent design review; no product changes reviewed.

Reviewed: `docs/observation-implementation-plan.md` and the accepted `docs/observation-strategy-design.md`.

## Verdict: PASS (revised plan)

Accepted plan SHA256: `779d557a6e5fad547bb5d2e44177d49f03c43243b45c2ee57cf959cffe4d269a`.

The revised plan resolves both blocking findings below. A Parallel path with any branch or explicit thread selection now has unverified logical coverage, without inferring uniformity from the witnesses received. Only a branch-free, unfiltered Parallel path can close the independently enumerated logical coordinate set; execution coverage remains unverified in all Parallel cases. Early exits anywhere in the Kernel body are rejected for this increment, including exits in preceding sibling branches. These conservative boundaries permit useful capture and numerical comparison without claiming unavailable execution-domain proof.

No design blockers remain. The implementation and acceptance requirements below remain applicable. This PASS authorizes implementation of the scoped plan; it is not code or GPU acceptance. The new `examples/nested_scopes/kernel.py` and `run.py` were inspected as candidate fixtures; no independent GPU evidence acceptance was performed in this review.

The scalar/local first increment is appropriately scoped. It does not need new lowering, a layout parser, or support for collective observations in this increment. Keeping the existing tile implementation and its regressions is sound. Binding snapshots, typed raw bits, self-identifying records, independent references, and separate logical/execution coverage are the right foundations.

The initial review requested two executable decisions; both are resolved by the revision described above:

1. **Specify the recursive state after a Parallel scope.** Before Parallel, the static thread domain can be enumerated. After Parallel, a Cartesian product of threads and coordinates is not an expected execution domain. The plan acknowledges this, but does not say how recursive branch witnesses are matched when multiple threads report the same coordinate, or when a selected thread covers only part of the logical domain. Define exactly when logical coverage may close and when it remains unverified. Arm disagreement is contradictory only for the same dynamic identity including thread; different threads can legitimately take different arms at the same coordinate. Include concrete cases for `Parallel -> thread-dependent if -> serial -> point`, a branch before Parallel, and explicit thread selection. The implementation may conservatively leave these ambiguous combinations unverified; it must not infer missing mappings from observed DATA, silently union away a missing required record, or report a filtered Parallel domain as not executed solely because no sample appears.

2. **Reject relevant nonlocal exits, not only ancestor AST node kinds.** A preceding sibling `if cond: continue` inside an enclosing loop can skip a point even though the point's ancestor list contains no early-exit node. A preceding conditional return/break has a similar effect. Either conservatively reject such exits in the relevant enclosing function/loop regions, or define their reachability witnesses. The former is sufficient for this increment. Add negative tests where the exit is in a sibling branch before the selected line. Unrelated gemm/ws/pipeline calls still need no function-wide ban.

## Implementation / acceptance requirements

These clarify the existing plan and do not expand the feature scope:

- References declare their expected key domain independently of captured DATA. Missing keys must not be accepted by shrinking the expected reference domain to the records received. Correct numerical values with unverified execution coverage remain explicitly partial evidence.
- Preserve exact integers through decoding, serialization, alignment, and comparison. The int64 greater-than-2**53 test must reach offline analysis, including a one-bit erroneous reference; testing only the printf codec is insufficient.
- Bind loop snapshots with generated identifiers that cannot collide with source names. Test the same-name nested binding on the compiled H200 path, not only the rewritten AST.
- Budget all emitted witnesses, including ancestors outside a deeper selected iteration. Test a zero-trip loop and a build-time inactive branch with no DATA: evidence must justify not-executed/inactive, or the run must remain incomplete/unverified.
- Missing-root, missing-branch, missing-one-D, and missing-whole-instance mutations must be separate tests. Also remove all events for one Parallel coordinate and all events from one replica: the former must not retain logical completeness when that coordinate is independently expected; the latter must never be presented as complete physical execution coverage.
- Preserve the existing 27 source captures and reviewed GELU/GQA regressions. New sample strategy coverage is not established by those old tests.

This review is limited to the implementation plan. Code review and independent H200 evidence acceptance remain required after implementation.
