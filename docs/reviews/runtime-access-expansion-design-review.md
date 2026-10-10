# Runtime and source-access expansion: independent design review

2026-10-09. Reviewer: `/root/unified_core_review`. Scope: proposed direction,
not implementation acceptance. Existing generic runtime, snapshot verifier,
and numerical source emitter were inspected. No implementation was changed.

**Direction approved**, with the following correctness requirements to include
in the audit plan. Static/contiguous/positional/default-stream checks currently
exist in the debugger wrapper and must not be attributed wholesale to T.print.
Removing wrapper restrictions does not by itself demonstrate that every
underlying TileLang frontend/backend combination supports the same inputs.

## Runtime compatibility

- Remove exact TileLang/helper/H200 fingerprint admission from the generic
  path, retaining recorded versions, source hashes, device information, and
  tested combinations. Diagnose missing concrete frontend/backend features.
  Keep exact reproduction checks only in explicitly chosen legacy reviewed
  paths, without silently routing ordinary user kernels into them.
- Normalize keyword arguments using the compiled adapter's actual input
  contract, with original frontend names as the source map. Account for
  omitted outputs, inferred/hidden symbolic dimensions, scalar arguments,
  aliases and duplicate/missing names. Snapshot normalization must not clone
  or reorder the values actually passed incorrectly. Arbitrary launch options
  must not accidentally be bound as kernel input tensors or silently dropped.
- Accept a non-default stream on its actual device. Synchronizing the current
  stream is sufficient only if the launch really uses that stream; explicit
  stream arguments and multi-device tensor inputs require deliberate binding.
  CUDA graph capture is a different execution model and must not be silently
  broken by a synchronization/host-log wrapper.
- Device printf is a shared logging resource. Verify that the chosen stream
  synchronization produces complete launch boundaries and cannot attribute
  another concurrent launch's records to this launch. Supporting a non-default
  stream is different from preserving multi-stream concurrency/timing. Do not
  silently replace current-stream synchronization with global serialization
  while claiming the latter property.

## Strided values and aliases

Logical tensor bytes plus shape/stride/storage offset can describe ordinary
views, but the current `tensor.copy_` reconstruction is insufficient for
zero-stride expansion and internally overlapping `as_strided` views. Construct
shared byte storage and populate addressed bytes with overlap consistency
checks, or use an equivalent mechanism that actually supports those layouts.
Mixed-dtype aliases use byte offsets derived from each view's element width.

Storage span validation must use the reachable offsets, not `offset + numel`;
for nonnegative strides it is based on
`offset + sum((shape[d] - 1) * stride[d])`, with a separate empty-view case.
Do not group unrelated empty storages solely because their data pointer is
zero. Reference inputs must preserve alias relationships while remaining
isolated from saved evidence. Logical snapshots do not define unobserved gaps
in the underlying storage; do not imply otherwise.

Required examples: transpose, strided slice, offset views, two overlapping
views, expanded zero-stride input, empty tensors, scalar tensor, and mixed-dtype
views where supported by the underlying backend. Preserve argument aliases in
the actual launch; snapshotting may make CPU copies but must not replace the
user's argument objects.

## Symbolic shape contract

Retain frontend expressions for shapes and strides, plus concrete per-launch
values/bindings. Check repeated symbols consistently across inputs and outputs;
do not merely skip all shape validation after removing `int(...)`. Distinguish
parameters passed explicitly from dimensions inferred by a TileLang adapter.
A same-compiled-kernel test with at least two concrete shapes is needed.
Observing dynamic-size whole buffers additionally needs bounded enumeration
and per-visit shape/element-count evidence; wrapper acceptance alone does not
remove the emitter's static-region assumptions.

## Source logical access path

A default source path is appropriate for requested logical accesses: identify
the source operation/buffer, evaluated index or region, shape/strides, enclosing
conditions, and validity/mask information without reading the target data.
Record it as a **source logical access request**, not a hardware transaction.
Parallel layout distribution, vectorization, implicit boundary guards and TMA
transactions cannot be inferred solely from the requested source region.
Keep the old explicitly reviewed physical/lowered trace as a separately named
mode; neither trace kind should masquerade as the other.

The principal semantic risk is evaluating indices or masks twice. Expressions
may read index tensors, invoke helpers, or depend on values mutated by the
observed operation. Reuse existing bindings or introduce a source adapter that
evaluates each expression once in its original order. Before/after observation
must refer to the indices used by that operation, not a later recomputation.
For opaque operations, record the known request and unknown mask separately
rather than inventing effective-access evidence. A skipped branch produces no
request, and a zero-volume region is distinct from missing logs.

Reuse TLDBG3 counters and mandatory frontend-thread E domains, but explicitly
define access-record payloads and expected element cardinality. Do not decode
index/region records as captured numerical values. Loss of a complete access
instance or all records from one participating frontend thread must still fail.

## Manual pipeline schedules and release evidence

Run a real manual `order/stage/sync/group` example. Those arrays correspond to
scheduled statements and inserted logging can change statement cardinality.
Do not discard the user's schedule, convert the pipeline into a different loop,
or use an automatic-pipeline smoke test as manual-schedule evidence. A source
adapter must preserve schedule semantics or report the precise unsupported
form and retain a reproduction.

The audit should map each restriction to its actual owner (wrapper, protocol,
source adapter, emitter, frontend, backend, legacy reproduction), then give the
replacement and a positive/negative test. Final evidence should include an
installed artifact, independent references, malformed/missing access records,
non-default stream launch boundaries, strided alias reconstruction, and
dynamic-shape reuse. This review approves that direction; it is not a blanket
claim that all restrictions can be safely removed by deleting checks.
