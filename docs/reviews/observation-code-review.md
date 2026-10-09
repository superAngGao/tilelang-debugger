# Nested observation independent code review

Date: 2026-10-09. Reviewer: independent `nested_code_review` session.

## Verdict: PASS after revisions

No remaining blocking finding in the reviewed scalar/local increment. This is code review, not GPU acceptance. The H200 matrix, independent references, sanitizer evidence, and existing-path regressions require their separate acceptance review.

Reviewed the working-tree diff against main, the accepted implementation plan, and the new `scopes.py`, `sample_monitor.py`, `sample_records.py`, and `sample_analysis.py`. Also inspected integration changes in source capture, monitor session validation, evidence reconstruction, numeric comparison, schema dispatch, and the nested fixture.

## Findings resolved during review

1. **Additional evaluation of loop bounds.** Initial injection copied arbitrary original bound expressions into the kernel-entry root witness while retaining the original loop expression. A bound such as `next_extent()` therefore acquired an extra, earlier invocation. The revised preparation rejects calls, buffer reads, and names assigned in the kernel body; supported bounds are pure static integer expressions over closure constants. Independently confirmed the impure-bound rejection.

2. **Numeric-tail truncation accepted as a complete event.** Initially the last field was the numeric high word. Truncating `2097152` to `209715` retained valid field count and bit width while changing an int64 value; parsing still reported complete instance integrity. Every TLDBG2 event now ends with the fixed `|E` terminator, which the parser requires. Independently confirmed tail truncation is rejected.

3. **Unconstrained element expressions expanded the memory-read boundary.** The first scalar-element extension accepted arbitrary subscripts. The revised implementation only accepts an element observation immediately after the source statement writes that same syntactic element. Its buffer base must be a name, and indices cannot contain calls or nested buffer reads. Independently confirmed `x[x[0]]` is rejected. This supports the intended Softmax write-back observation without presenting arbitrary memory probes as already supported.

## Independent checks

- Ran `python -m unittest discover -s tests -p test_scopes.py -v`: all 14 tests passed.
- Ran separate in-memory protocol probes, rather than relying only on those tests: reordered logs; missing root; missing whole serial instance; duplicate event; truncated final numeric field; missing ancestor branch witness; a DATA record on the opposite branch; Parallel coordinate loss versus replica loss; exact int64 reference values above `2**53` with a one-unit incorrect reference. All produced the expected result after revisions.
- Reviewed the decoder and comparisons: int64 combines two uint32 words and remains an integer through decoding, JSON serialization, reference alignment, and equality; integer/bool tolerances must be zero.
- Confirmed the parser distinguishes observed-instance integrity, logical coverage, and execution coverage. Any Parallel path keeps execution coverage unverified. Parallel with a branch or thread filter also keeps logical coverage unverified. No observed-arm agreement is used as proof about missing threads.
- Confirmed branch witnesses remain inside original arms, conditions are not duplicated, and loop aliases preserve separate same-name ancestor bindings. Unsupported early exits in sibling statements are rejected throughout the selected kernel body.
- Confirmed schema-2 records dispatch to their own parser and analysis. Legacy TLDBG1 paths retain their original completeness requirements; the shared integer comparison change preserves exact int32 behavior.
- `git diff --check` passed.

## Acceptance boundaries

This review does not claim arbitrary collective fragment/shared/global observation, pipeline/group support, dynamic loop bounds, or complete physical execution coverage under Parallel. A successful numerical reference does not upgrade unverified coverage. Source-level read-back still relies on the original write and execution context being valid; it is not a general memory-safety proof.

GPU acceptance should explicitly include build-time inactive and zero-trip points, original-iteration selection, selected threads, the same-name nested loop fixture, local/bool/int64 values, real TileOPs branches, and the existing source/reviewed regressions. Reference-error and log-loss negative cases remain part of acceptance, not just positive numerical equality.

Reviewed core-file SHA256 values:

| File | SHA256 |
| --- | --- |
| `scopes.py` | `b4db95b0f165f0355f1b3fba25c55c93388fc72801516efa38073891b4c146cb` |
| `sample_monitor.py` | `60cf489d387798b8cbe7902dcd101966b130feefbb17ee3e1b390b688001a926` |
| `sample_records.py` | `c280824045d7ae8f4580246eef738b011614d64a37298b734e1f326efa5c8f77` |
| `sample_analysis.py` | `2c69d710c4f8522675a197e4da0af46a13d4c5e2a3071b0206dbc17a0fcf8da8` |
