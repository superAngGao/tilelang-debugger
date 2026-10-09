# Unified runtime independent review

2026-10-09. Scope: `runtime/builds.py`, `runtime/unified_capture.py`,
`runtime/unified_evidence.py`, and the frontend decorator wrapper in
`instrumentation/unified.py`. The schema 3 offline numerical analysis component
is outside this review. No product code was changed by this reviewer.

Status: **PASS** for the runtime/build-lifecycle scope. The four initial findings
are closed. Final verification used the installed frozen wheel with SHA-256
`c0d703825a43e158022c790e4f2c3db04acfe69764e24a8b28185071ab588456`.
This is not a substitute for the separate full-product acceptance report.

## Initial findings and fixes

1. **Unselected kernels cannot run in a multi-kernel driver.** The compilation
   hook intercepts every `tilelang.compile`, but `building(...)` wraps only
   frontend functions owning selected points. An ordinary helper kernel in the
   same selected source, or in another module, consequently has no
   `tldbg_build_id` and is rejected. Preserve these compilations and launches with
   an empty point set, and distinguish an actually unobserved kernel from a
   selected frontend function whose instrumentation failed to attach.
   **Fixed:** unobserved compilations keep empty point sets; configured but
   unlaunched points are explicitly reported and make the run partial.

2. **Valid zero-dimensional tensor arguments fail during snapshotting.**
   `value.detach().view(torch.uint8)` raises for scalar tensors whose dtype has
   more than one byte. Reproduced in the pinned Torch environment with int32 and
   float32 tensors: `self.dim() cannot be 0 to view ... as Byte`. Flattening the
   detached contiguous tensor for byte serialization preserves the original
   shape and alias metadata without replacing the kernel argument.
   **Fixed:** byte serialization reshapes the detached tensor to one dimension;
   kernel arguments and their saved original shape/storage information remain
   unchanged.

3. **Offline verification accepts missing snapshots and incompatible argument
   manifests.** On independent temporary copies of the successful H200
   `artifacts/unified-smoke-2/parallel` capture:
   - Replacing every baseline/instrumented `before.json` and `after.json` with
     `[]` passes verification despite nonzero launch argument/output counts.
   - Changing both workers' first launch `arguments` to 100 (and updating the
     corresponding launch identity) also passes despite the actual compile
     signature and snapshot counts.
   Validate snapshot lengths and roles against the launch and compilation
   manifests, including actual dtype/shape, scalar representation, and valid
   alias/offset/stride metadata. The verifier must reject incomplete evidence
   even if the two workers have identically incomplete snapshot lists.
   **Fixed:** before/after cardinalities are checked against compile parameters
   and output roles; dtype, shape, scalar representation and alias metadata are
   verified.

4. **Compilation and source identity files are not verified.** Deleting either
   `instrumented/compiles/0/identity.json` or
   `instrumented/source/original.py` from separate temporary copies still passes.
   Validate saved compile identities, selected source/driver hashes and expected
   compiler artifacts, so the advertised source/compile provenance remains
   reviewable offline. This does not require parsing lowered IR.
   **Fixed:** the verifier checks source/driver, saved plan and staged source
   hashes, compilation identities and hashes of all six exported compiler
   artifacts. Saved-source hashes avoid coupling old evidence to future source
   generator implementations.

## Positive observations

- Build sessions deep-copy per-build points and restore capture state in
  `finally`; compilation copies the frozen point metadata again.
- Kernel arguments are passed through without cloning, preserving in-place
  behavior and alias relationships. Normalized storage groups avoid comparing
  addresses across worker processes.
- Launch boundaries use pre/post CUDA synchronization and C/Python stdout
  flushing. Parsing checks exact sequential begin/end identity and rejects
  device packets outside a launch. The default-stream and graph checks state
  the supported execution contract explicitly.
- The original two-launch H200 parallel evidence passed independent offline
  verification before all mutation checks. Tests used temporary copies and did
  not alter the original evidence.
- CPU audits passed for two-launch segmentation, missing/end/duplicate/mismatched
  boundaries, device packets outside a launch, and exceptional frontend session
  cleanup.
- The suspected `tilelang_out_idx` conversion issue was checked against the
  pinned frontend: its attribute values are ordinary Python integers. This is
  **not** a finding and needs no patch.

## Final frozen-wheel acceptance

The reviewer imported from
`/home/ang.gao/tilelang-debugger-unified-reader-final-20261009/installed`, verified
the wheel SHA-256, and compared the four reviewed modules' installed bytes with
their wheel entries. All audited GPU worker environment records identify that
installed package location.

At the review checkpoint, **82 completed schema 3 captures / 103 launches** passed
independent offline verification with this exact installed product. This includes:

- Core cases with two sequential launches, including an in-place case with two
  independent frontend builds and two compilations of the selected kernel.
- The alias case, with both kernels compiled before launch: an unselected helper
  has an empty point set, then the selected kernel launches with two arguments
  sharing the same storage. Original argument mutation and output evidence are
  checked without replacing the input tensors.
- Scalar runtime parameters, pipeline/group cases and source-selected TileOPs
  cases included in the saved audit manifest.

Ten negative tests on temporary copies all rejected: missing compile identity,
missing source, both workers' emptied snapshot lists, false argument cardinality,
corrupted CUDA artifact, invalid scalar dtype, noncanonical alias group, changed
monitor selection, changed staged source and missing host launch end. No original
capture was edited, and no compatibility adapter was used for final verification.

An additional direct CUDA snapshot check using the previous `f42e165a...b95297`
candidate's installed runtime passed
for zero-dimensional int32/float32/float64/bool tensors and overlapping contiguous
views with shared alias group and offsets 1 and 4. This establishes the snapshot
API behavior; the final snapshot/runtime module bytes are identical as verified
below. It does **not** claim that the pinned frontend supports 0D kernel
tensor parameters. The alias kernel uses a one-element tensor parameter.

Reproducible evidence in the final root:
`artifacts/runtime-review-reader-final.py`, `artifacts/runtime-review-final.json`,
and `artifacts/runtime-module-equivalence.json`. The additional snapshot check is
saved in the previous candidate root
`/home/ang.gao/tilelang-debugger-unified-release-20261009/artifacts/runtime-snapshot-review.json`.
Counts above describe this review's
checkpoint; later completed captures belong to the full acceptance inventory.

The user-requested thread/reader semantics change required a new final wheel.
The reviewer compared the following modules in both wheel archives and the final
installed directory; their bytes are identical. Thus previous runtime-only
snapshot checks remain applicable, while all 82 captures and ten evidence
negative tests above were independently verified using the new final wheel.

| Reviewed module | Identical SHA-256 in previous and final wheels |
| --- | --- |
| `runtime/builds.py` | `c622f7aae37523d63f7ab5f0c5e14b3e8c6e155eb315e49914ddaf793f8e250e` |
| `runtime/unified_capture.py` | `5e79d9a83bd32e670a6dc2bd1e2a941fb5a611a94250e0ffe6e36775a93da51d` |
| `runtime/unified_evidence.py` | `712d986f931f6e477f009d2084b301387dde59155cc0a67c453e8e43e1b30769` |
| `instrumentation/unified.py` | `0801768ffd8f65288ab6fbbc055940225ff18ba0a866415b30cf37613fa4eccc` |

## Subsequent evidence audit

Independently audited all 11 captures (two launches each) under
`/home/ang.gao/tilelang-debugger-unified-20261009/artifacts/unified-reference-1`.
Their numerical lifecycle/raw evidence passed. These development captures
predated a control-flow traversal-order change: their transfer metadata lists
continue before break, whereas the revised implementation traverses in reverse
order. For this audit only, an in-process compatibility adapter sorted transfer
metadata by source line; it did not alter saved evidence, injected code, runtime
records or other validation. This is mechanism evidence, not frozen-release
acceptance. The final hash-based provenance implementation removes this
generator-version coupling.

On temporary copies, all seven mutations were rejected: missing compilation
identity, missing original source, both workers' emptied snapshot lists,
inconsistent argument cardinality, corrupted CUDA artifact, invalid scalar dtype,
and noncanonical alias group. Original capture folders remained untouched.
