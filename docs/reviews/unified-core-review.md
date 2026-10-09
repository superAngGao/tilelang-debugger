# Unified core independent review

Current-interface note: final TLDBG3 uses **D/E/X**, with an independently known
frontend thread domain and one mandatory E per thread. Thread identities are
`frontend`, not necessarily physical CUDA IDs. `point.thread` filters
scalar/local/element execution; whole-buffer capture uses `region` for logical
elements and the separate optional `collective.reader` for its output reader.
See the [automatic-pipeline revision](automatic-pipeline-root-design-review.md)
and the final lifecycle/reader appendices below. Earlier R/E and physical-thread
descriptions are preserved candidate-review history, not the current contract.

2026-10-09. Reviewer: `/root/unified_core_review`.

Status: **PASS — core and reference-analysis code review**, including subsequent
bounds-guard fixes and final D/E/X lifecycle revision. All six findings are resolved. Final installed-wheel
runtime/reference/sanitizer acceptance remains separate.

Scope: `source_analysis/control_flow.py`, `instrumentation/unified.py`,
`emitters/unified.py`, `protocols/unified.py`. Runtime integration and final H200
acceptance are separate reviews. Missing unfinished peripheral modules are not
treated as findings against this core review.

## Original findings — resolved in second review

1. **Collective uniformity check omits dynamic counted loops.**
   `observe()` asks for `collective.uniform=true` only when an ancestor is a
   branch, Parallel, or while. A serial/unroll/pipeline bound can depend on a
   physical thread; those ancestors can also diverge in how often the inserted
   barrier is reached. For example, a shared-buffer observation inside
   `T.serial(tx)` currently receives a CTA barrier with only `ready=true`.
   Require an explicit participation-uniformity contract for all potentially
   divergent enclosing loops, or establish uniformity from a supported source
   rule. Reader selection correctly remains inside, rather than around, the
   barriers; that does not repair divergence of the original enclosing loop.

2. **Loop ordinal adapter ignores legal keyword arguments.**
   Local executable reproduction using `prepare()` and `inject()`:
   `T.serial(2, 8, step=2)` generates ordinal `(k - 2) // 1 + 1`, so original
   iterations `k=2,4,6` become ordinals `1,3,5`; selecting iteration 2 captures
   nothing. `T.Pipelined(2, stop=8)` similarly gets an incorrect start of zero.
   The pinned TileLang `language/loop.py` confirms that serial, Serial, unroll,
   Unroll, and Pipelined accept the relevant keyword arguments. Bind each
   frontend signature and preserve positional/keyword argument evaluation order
   when creating snapshots. Validate unsupported/duplicate argument forms
   explicitly rather than silently computing the wrong identity.

3. **Protocol accepts impossible coordinate payloads as complete.**
   A CPU reproduction with one root, one DATA, and one end packet accepts a
   coordinate `2**80` and Parallel ordinal `17`, returning
   `capture_integrity=complete`. The emitter sends signed int64 coordinates and
   ordinals and fixes Parallel ordinal to zero. Check signed field width and
   the per-loop ordinal invariant. Do not infer dynamic loop bounds from the
   received data; width and emitter invariants are independently known.

## Additional compatibility validation required

Pinned `Pipelined(order=..., stage=..., sync=..., group=...)` metadata indexes
scheduled executable statements. Inserting printf/counter/synchronization
statements while preserving the original arrays may invalidate that manual
schedule. This review has not run a GPU reproduction of this combination.
Exercise it; either implement its source adapter or provide an explicit
diagnostic and accurately record the limitation. Automatic `num_stages` tests
alone do not establish manual schedule compatibility.

## Positive observations

- R/E expected participants come from frontend geometry, not observed log rows.
- DATA identities use per-point/thread visits and complete element sets;
  cross-thread log order is not required.
- Missing DATA within an emitted visit fails independently of budget truncation.
- Collective reader/iteration/budget filters do not wrap the inserted barriers.
- Protocol imports remain CPU-only; the implementation does not inspect lowered
  IR to choose the numerical capture strategy.

These observations are not a final acceptance verdict. Re-review will check the
fixes and associated tests; H200 evidence remains necessary for pipeline/group
counter semantics and collective synchronization.

## Second review

The implementation now requests uniformity for all enclosing loop kinds,
snapshots positional and keyword arguments, computes ordinals using the actual
start/stop/step binding, and represents signed coordinate/ordinal values as two
uint32 words. Manual pipeline arrays receive an explicit unsupported-adapter
diagnostic. The original three findings are resolved.

Independent CPU checks passed for positional/keyword mixed calls, all-keyword
serial calls, Pipelined keyword stop, and explicit `stop=None`. Six signed64
boundary values (`-2**63`, `-2**32-1`, `-1`, `0`, `2**32+1`, `2**63-1`) round-trip
through the two-word protocol; an out-of-range word and a missing end record
are rejected. Parallel ordinal zero is now checked explicitly.

### Second-review findings — resolved in third review

4. **Conditional return does not have runtime-return semantics in this frontend.**
   A real pinned-frontend reproduction (`artifacts/unified-return-review.py`;
   remote `/tmp/tldbg-unified-return-review.py`) builds:
   `A[tx] = 1; if tx == 0: return; A[tx] = 2`.
   Its frontend TIR has `A[tx] = 1` and `if tx == 0: evaluate(0)`, with the
   final store absent for every thread. The eager AST translates return into a
   Python return from the construction function. Thus inserting finish before
   the conditional return emits E only inside that branch and the appended
   kernel finish is never constructed. The remaining threads have no end
   record. Diagnose these unadapted nested returns explicitly rather than
   claiming general runtime early-return support. An unconditional final return
   may be supported separately with actual lifecycle evidence. Nested helper
   function/macro returns must not be mistaken for exits from the kernel root.

5. **Inactive points bypass root/end budget reservation.**
   Budget calculation occurs only in `observe()`, which is absent for a
   compile-time inactive point; `start()` still emits R/E. Protocol validation
   counts budget per point rather than per launch. Two inactive points with
   32 threads each produce 128 lifecycle events; with launch budget 100, each
   point has only 64 events and both are accepted. Reserve and validate root/end
   cost independently of DATA construction, and enforce the aggregate launch
   event limit in parsing. This must not change legal zero-visit semantics.

The second review does not claim H200 acceptance of the revised word encoding;
that evidence is being gathered by the implementation/acceptance worker.

## Final core review

The return adapter now rejects nested, conditional, value-bearing, or non-final
returns inside the selected kernel. A final bare return receives its end packet
before the frontend exits; the ordinary tail finish is not constructed in that
case. This is a deliberately bounded frontend contract, not an implementation
of arbitrary GPU early returns. Manual pipeline schedule arrays likewise remain
an explicitly diagnosed limitation.

`start()` now checks the aggregate known root/end reservation independently of
whether a point's DATA is constructed. The parser enforces a single aggregate
launch event budget and bounds E's emitted count before materializing expected
visit sets. The inactive-point over-budget reproduction is therefore rejected.

The reviewer independently ran `python -X utf8 -m unittest discover -s tests
-p test_unified.py -v`: **16/16 passed**, including both new regressions,
keyword loop snapshots, malformed words, Parallel ordinals, missing entire
threads/visits, truncation versus missing DATA, and exact integer references.
The earlier independent signed64 boundary checks and fixed-frontend conditional
return experiment remain supporting evidence.

No unresolved blocking finding remains in the four reviewed core modules.
This PASS does not substitute for full H200 acceptance: collectives under the
documented readiness/uniformity contract, supported loop/group combinations,
final bare return, dtype emission, and installed-artifact provenance still need
the corresponding acceptance evidence. The implementation worker reports
11 H200 smoke cases with two launches each passing; this report does not count
that report as independently audited sanitizer/reference acceptance.

## Subsequent delta review

Nested function/macro transfer traversal now stops at function boundaries;
insertion resets and restores the enclosing root context when visiting a
function. This prevents ordinary nested function returns from inheriting the
outer kernel finish. Bounding region cardinality before allocating the index
list also addresses the large-buffer allocation concern.

6. **New out-of-bounds diagnostic can disappear without invalidating E — resolved.**
   The new scalar BufferLoad guard emits X for an invalid index but does not
   update the attempt count or any persistent error state. If X is dropped by
   printf, R/E still report zero visits and parsing accepts a complete empty
   observation. Repeated invalid iterations also emit unlimited X packets,
   bypassing capacity. Persist the invalid-read status in the independently
   required E record and bound diagnostic output. Dropping every X must still
   make the capture fail; budget exhaustion must not suppress the error state.
   This does not require performing any invalid memory read.

The added `unified_analysis.py` comparison path preserves integer exactness,
checks independently provided key sets, compares all replicated logical values,
and uses finite nonnegative floating tolerances. Runtime alias reconstruction
was checked independently on torch as described below.

### Delta resolution and independent checks

Invalid selected BufferLoad attempts now set a persistent error bit in E and
advance the saturating attempt counter. At most the first invalid attempt emits
X, and only when it falls within capacity. E causes an explicit error even if
all X records disappear, including invalid attempts after output capacity was
exhausted. The valid read remains under its bounds guard.

The reviewer reran the expanded suite: **17/17 passed**, including the new
lost-X/end-error regression. Separate checks confirmed a nested macro return
does not inherit an outer finish or transfer record, a billion-element buffer
without a bounded region is rejected before allocating indices, and a small
region of that buffer still works.

An independent torch experiment (`artifacts/unified-alias-review.py`) ran both
against a separate source snapshot and the current remote candidate source.
Two overlapping contiguous int64 views with nonzero offsets reconstructed the
same storage; `deepcopy(tuple(inputs))` retained alias relationships while
isolating original inputs. Comparing values above `2**53` succeeded exactly,
and changing one integer by one produced a mismatch. The reference comparison
and alias loader have no unresolved blocking finding from this review.

## Final automatic-pipeline lifecycle revision

The approved design in `automatic-pipeline-root-design-review.md` is now
implemented. Final TLDBG3 emits D/E/X, not R. The independent root domain is the
frontend launch geometry, each point/thread requires exactly one E, and budget
reservation accounts for those mandatory E packets. Counters remain initialized
at the source kernel root. Parser checks still reject missing/duplicate E,
missing visit elements, invalid-read state, impossible counters, and over-budget
logs. Old candidate R packets are rejected. TLDBG1/2 remain separate.

Thread identity is explicitly `thread_space=frontend` in metadata and records.
It is not misrepresented as a CUDA physical thread ID when automatic TMA warp
specialization introduces producer threads and remaps consumer bindings.
No product lowered-IR parser was introduced for this revision.

The reviewer independently reran the local suite: **18/18 passed**. The reviewer
also audited actual H200 evidence under the remote
`tilelang-debugger-unified-root-revision-20261009/artifacts` workspace using
`artifacts/unified-root-review.py`:

- `pipeline-mixed`: one active fragment observation and two compile-time
  inactive observations (before/after the pipeline). Each has exactly 32 E
  records with the independently expected frontend domain 0..31.
- `pipeline-inactive`: only the two inactive observations. Both have zero DATA,
  zero counters, and exactly 32 E records covering that same complete domain.
- All five point domains were checked directly from raw packets. Deleting E
  for thread 31 of each point causes `missing end`, including every zero-DATA
  point; participants are not inferred from surviving records.
- Offline evidence verification passed for both captures, including saved
  baseline/instrumented racecheck logs, snapshots, compile/launch manifests,
  and protocol recomputation. Independent reruns of each saved reference
  provider returned `matched=true` and `capture_complete=true`; results were
  saved as `core-review-analysis` alongside those captures.

No blocking issue remains in this revision. This evidence approves the
source-level core change and the tested zero-DATA automatic-pipeline domains;
the complete final installed-wheel matrix and provenance audit remain a
separate release acceptance gate.

## Thread selection versus collective reader — interface design review

**Design PASS** (implementation not yet reviewed in this appendix). For schema
3, `point.thread` selects the frontend thread execution being observed for
scalar/local/element capture. Whole fragment/shared/global buffer or region
capture selects logical elements through `region`; it must reject a non-null
`point.thread` rather than reinterpret it as the output reader.

An optional advanced `collective.reader` selects which participating frontend
thread emits a cooperative capture. The default is the first participant and
does not change the requested logical region. An integer or `[x,y,z]` reader
must be normalized against frontend extents and validated both against the CTA
and the actual explicit group domain. Scalar/local/element capture rejects a
collective contract, whose fields otherwise have no meaning for that mode.

The protocol may still record the emitter's frontend identity. All expected E
records remain mandatory independently of either selection; reader choice must
not constrain cooperative participation or barrier reachability. No physical
thread mapping or lowered-IR parser is required. Group-fragment reader 129
should move to `collective.reader=129` and preserve its existing reference.

Required implementation checks: default and explicit readers, equivalent XYZ
reader, outside-CTA and outside-group reader rejection, whole-buffer `thread`
rejection, scalar/local `collective` rejection, and unchanged logical values
with different valid readers. Final evidence must use a new frozen artifact
directory, preserving the previous frozen run.

### Reader interface implementation review

**Code PASS.** `frontend_thread()` consistently validates and normalizes a
linear integer or XYZ coordinates in the frontend domain, rejecting bool,
negative, malformed, and outside-CTA selectors. Cooperative capture rejects
`point.thread`, obtains its optional reader only from `collective.reader`, and
checks membership after intersecting enclosing explicit group domains. Direct
scalar/local/element capture rejects an inapplicable collective contract.
Reader/default selection remains inside the DATA condition; it does not guard
barriers or mandatory E output. The parser retains its separate reader and
thread filters.

The reviewer independently reran the expanded CPU suite: **19/19 passed**.
README describes source-thread selection and cooperative output reader as
different concepts and explicitly names the frontend coordinate space. The
reviewed emitter file SHA256 is
`e57e57306c566669c74af7a997db5590093def9bf4f42e84397e9995e2ed8e0e`;
the same bytes were independently confirmed in the new remote installed
artifact at `tilelang-debugger-unified-reader-final-20261009/installed`.

The advanced GPU matrix includes default reader 128, integer/XYZ reader 129,
and four invalid-configuration cases covering whole-buffer thread selection,
reader outside CTA, reader outside group, and direct capture with collective
settings. At this code-review point that matrix is still running, so this
appendix does not preempt its final independent acceptance result.
