# Automatic pipeline root lifecycle design review

2026-10-09. Reviewer: `/root/unified_core_review`.

Scope: design only; no product code modified. Evidence inspected directly from
`artifacts/pipeline-copy/capture/instrumented/compiles/0/kernel.cu` and
`points.json` under the final H200 workspace.

## Finding

The frontend specifies 32 threads. Automatic TMA warp specialization launches
160 hardware threads, leaves the constant R printf outside specialization, and
places counter initialization, DATA, and E in the consumer region. R therefore
prints hardware IDs 0..159, while DATA/E print remapped frontend IDs 0..31
(consumer hardware IDs 128..159). Root domain validation correctly rejects this
inconsistent record stream. The debugger must not infer a larger domain from
the received R rows or add a lowered-IR parser to repair identities.

## Recommendation: one mandatory E per frontend thread

**PASS as a proposed design**, subject to the acceptance cases below. Keep the
counter initialization and final E, omit R from this new lifecycle variant.
The independent expected participant domain remains the frontend launch
geometry. Require exactly one E for every point and every expected thread in
the selected block, including zero-visit threads. E records contain attempts,
emitted visits, overflow, and invalid-read state. DATA must match every emitted
visit's complete element set. Missing whole threads, whole visits, or final
records must still fail.

This removes a redundant observation rather than weakening completeness:
constant R does not independently count observation attempts. Mandatory E,
independent participants, visit numbering, complete element sets, and host
launch boundaries already supply the relevant evidence. No DATA/E rows are
used to enumerate expected participants. A missing E is never classified as
zero execution.

The budget reserves one mandatory E per point/thread before allocating DATA
capacity, replacing the two-record reservation. Root counter
initialization remains outside the observation's source branches/groups/loops.
The E call remains outside DATA selection and uses counter state even for zero
visits. The unpublished TLDBG3 candidate can be finalized as D/E/X without R
compatibility; published TLDBG1/2 behavior stays unchanged. Reject rather than
silently ignore R rows under the final TLDBG3 contract.

## Smaller alternative

Passing `counter[0..2]` into R may let dependency-driven specialization place
it beside initialization and E. This is a reasonable mechanism experiment and
minimal code change. It is less attractive as the lasting protocol because
those values are initially constant and its placement remains susceptible to
the compiler's classification/optimization order. A successful single kernel
does not establish zero-visit or inactive-point behavior. Do not declare that a
local-buffer operand universally forces placement without testing that claim.
The implementation worker subsequently reports this experiment failed: R and
initialization were moved outside specialization, while DATA/E remained in the
consumer region. It therefore does not fix the issue and is not the selected
design.

## Identity terminology

In this kernel DATA/E identify **frontend logical threads**, not physical CUDA
thread IDs. Expose/document the thread coordinate space accordingly (for
example `thread_space=frontend`). Source thread filtering then follows the
source binding. Automatically added producer threads are not members of this
source observation domain. Do not claim hardware thread coverage or compute
physical IDs from a guessed offset of 128. This does not require changing the
source program's scheduling.

## Required acceptance

- Real TMA pipeline with active DATA: exact frontend thread domain and values.
- Runtime-zero iterations and a branch selecting no visits: all expected E,
  zero attempts, no DATA.
- Compile-time inactive observation alongside an active pipeline point: all
  independent lifecycle domains remain valid; no domain inferred from logs.
- Multiple points and different thread/iteration filters: each keeps its own
  counter and mandatory participant set.
- Remove an entire thread, E, a complete visit, or one element: each fails.
- Budget truncation, invalid-read state without X, and overflow retain their
  existing distinctions.
- Existing explicit Group and multidimensional-thread cases still pass.

If automatic specialization moves or duplicates the required E outside the
frontend domain in one of these cases, it is an unresolved adapter issue. It
must not be hidden by relaxing expected participants. Final behavior needs
installed-wheel H200 reference and sanitizer evidence, not product inspection
of generated CUDA.
