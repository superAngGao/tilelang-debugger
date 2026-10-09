# Unified installed-wheel independent acceptance

2026-10-09. Reviewer: `/root/unified_acceptance_review`.

Status: **PASS** for the final installed wheel and documented supported scope.
All final runners have completed. The reviewer independently reverified raw
evidence and recomputed numerical references; no running or expected-failure
capture is counted as a complete successful capture.

## Final artifact and provenance

Artifact: `tilelang_debugger-0.2.0-py3-none-any.whl`, SHA256:

`c0d703825a43e158022c790e4f2c3db04acfe69764e24a8b28185071ab588456`.

All 48 Python wheel members match the reviewed local source and the remote
installed files byte for byte. Independent verification imports the product
from the new, isolated final installation:

`/home/ang.gao/tilelang-debugger-unified-reader-final-20261009/installed`.

Every audited schema-3 baseline/instrumented GPU worker records this exact
installed `tilelang_debugger/__init__.py` path in its own `environment.json`,
along with NVIDIA H200 and the pinned environment. Final acceptance does not
reuse earlier schema-development or reader-interface candidates. The old
reviewed entry retains its older environment format without a debugger-path
field; its regressions were launched under the same installed environment and
their raw evidence and references were independently reverified here.

## Completed matrix and numerical audit

The reviewer independently ran `runtime.unified_evidence.verify` and numerical
`analyze` against raw device logs, source/compilation artifacts, typed argument
and output bytes, aliases, launch identities, and reference providers. Runner
summary lengths and every expected outcome were also checked.

| Matrix | Valid capture directories | Launches | DATA records | Mode |
| --- | ---: | ---: | ---: | --- |
| core | 11 | 22 | 2,004 | racecheck |
| core-sync | 11 | 22 | 2,004 | synccheck |
| advanced | 12 | 13 | 817 | racecheck |
| advanced-sync | 12 | 13 | 817 | synccheck |
| types-direct | 26 | 26 | 1,248 | ordinary execution |
| types-collective | 26 | 26 | 832 | ordinary execution |
| TileOPs | 5 | 5 | 1,742 | racecheck |
| pipeline-mixed | 1 | 1 | 256 | racecheck |
| pipeline-inactive | 1 | 1 | 0 | racecheck |
| pipeline-sync | 1 | 1 | 256 | synccheck |
| **Schema-3 valid total** | **106** | **130** | **9,976** | |

The complete schema-3 matrix contains **116 attempts**: **102 complete**, **4
deliberately partial budget-truncated captures**, and **10 expected rejected
attempts**. The latter are eight configuration rejections and two OOB-read
diagnostics, each exercised under racecheck and synccheck. The four partial
captures remain partial and their analysis remains incomplete/unmatched. They
are valid truncation evidence, not complete captures. The other 102 captures
match independently generated sample keys/values and argument/output references.

The core covers Parallel tails, Pipelined, Group, two-dimensional CTA/local,
dynamic bounds, while with break/continue, negative steps, fragment/shared/global
regions, in-place mutation and repeated launches/builds. Advanced cases cover
group and pipeline fragment capture, nested dependent loops/branches, repeated
while values, final bare return, unselected helper kernels, same-object aliasing,
thread/iteration selection, no selected visit, and direct/collective budget
exhaustion. The 52 dtype cases cover 13 dtypes across scalar, local, fragment and
shared; they do not claim a separate sanitizer run for every dtype combination.
The five TileOPs captures use real pinned pool, indices, RoPE, Softmax and RMS
kernels with independent CPU references.

Legacy compatibility adds **7 complete captures and 71,172 DATA records**:
GELU, Sum, unpadded Sum, GEMM and GQA under racecheck; GEMM and GQA under
synccheck. The reviewer revalidated raw evidence and recomputed intermediate
and final-output references using the installed package. Across both schemas,
113 valid capture directories contain 81,148 DATA records; the four partial
schema-3 captures retain their explicit partial status.

All **13 CPU/TIR suites, 99 tests**, pass. The reviewer checked each original
suite log for its actual test count, successful completion and absence of skips.

## Automatic pipeline correction

The initial unpublished candidate exposed a real root-domain error: a constant
R printf ran on compiler-added producer threads while DATA/E used the frontend
consumer domain. That candidate failed and is not final-release evidence.

The independently reviewed final D/E/X protocol omits R and still requires one E
for every independently known frontend thread and point, including zero visits.
No received packet enumerates the expected domain. Records explicitly say
`thread_space=frontend`; source-thread coordinates are not advertised as physical
CUDA IDs after automatic specialization.

Fresh final-wheel pipeline captures pass with one active point plus two
compile-time inactive points, with only the two inactive points, and under
synccheck. The exported mixed-case CUDA still has 160 hardware threads for the
32-thread frontend domain and TMA operations; `pass_configs` remains empty.
Thus the acceptance was not obtained by disabling automatic specialization or
adding a product-side lowering/layout parser. Generated CUDA was inspected by
the reviewer as experiment evidence, not consumed by the capture policy.

## Reader and source-thread semantics

Normal whole-buffer/region capture needs no thread selection. The final API
reserves `thread` for scalar/local/element selection and uses optional
`collective.reader` only to override the output reader of collective capture.

Six successful group-reader cases, three each under racecheck and synccheck,
were checked against independent fixture expectations: default reader 128,
explicit reader 129, and XYZ reader `[129,0,0]`. Every point retains
`thread=None`, all 128 logical elements are present, and the participation
domain stays 128 threads with named barrier 15. A different reader does not
reduce transfer/synchronization participation.

All eight configuration rejections were checked for the specific diagnostic in
the instrumented worker's raw stdout/stderr and for **zero instrumented
launches**, not merely exit code 1: whole-buffer `thread`, out-of-CTA reader,
out-of-group reader, and a collective contract attached to a direct observation.
The pinned frontend `warpgroup.py` independently confirms that Group uses the
same x/y/z linearization as the implementation.

## Protocol and evidence negative checks

Twenty-five independent checks pass, using real capture logs/evidence where
applicable. They include accepting arbitrary packet order and rejecting missing
DATA, duplicate DATA, truncated packets, a missing whole thread, a missing whole
visit, missing END, changed END count, out-of-domain thread, an obsolete candidate
R packet, and a removed host launch boundary.

Evidence mutations reject changed snapshot bytes, removed persisted records,
changed CUDA, changed scalar argument, missing compile identity, missing source,
both workers' emptied snapshot lists, invalid alias group and invalid scalar
type. An int64 reference value above `2**53` changed by one produces exactly one
mismatch. Removing every X from the actual OOB capture still fails from E's
persistent invalid-read state.

An otherwise legitimate collective budget truncation with one admitted element
removed fails. A false overflow flag fails; a simulated saturated uint64 end
counter is classified as incomplete. This is a protocol simulation, not a claim
of executing `2**64` GPU iterations. Original capture evidence was never modified;
mutations use temporary copies or in-memory logs.

## Snapshot and alias roundtrip

Nine further installed-product CUDA/Torch roundtrip checks pass: a
zero-dimensional int32 tensor, overlapping contiguous int64 views, distinct
nonzero storage offsets, values above `2**53`, isolated deepcopy retaining mutual
aliasing, negative floating zero, a specific NaN payload, an int64 Python scalar,
and a bool scalar. These establish snapshot/reconstruction behavior, not a
zero-dimensional kernel or offset-view kernel integration claim. The separate
same-object-alias GPU case exercises the runtime launch path.

## Accepted boundaries and evidence location

The final README accurately separates supported behavior from remaining
boundaries: manual Pipelined schedule arrays, conditional/nested/value returns,
loop-else and unknown contexts need explicit adapters; only the final bare kernel
return is supported. Collective readiness/uniformity and named-barrier ownership
remain explicit user contracts. The tool does not prove arbitrary asynchronous
readiness, cross-block snapshots, arbitrary hardware-thread layouts or algorithm
correctness without a reference. Runs require static buffer shapes, contiguous
CUDA tensors and sequential default-stream launches. Access-trace generalization
is outside this numerical-capture implementation.

No unresolved acceptance blocker remains within that documented scope.
Independent scripts and results are under the final remote root's
`artifacts/independent-acceptance/`: `audit.py --final`, `audit.json`,
`wheel-manifest.json`, per-capture `reanalysis/` and `legacy-reanalysis/`, and
`snapshot-roundtrip/result.json`. `audit.json` records
`final_matrix_complete=true`. Earlier candidate audits are historical evidence
only; all final statistics above come from the reader-final root and wheel.