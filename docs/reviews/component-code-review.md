# Component refactor independent code review

2026-10-09. Reviewer: independent `component_code_review` session.

**PASS for the behavior-preserving component refactor.** No blocking code findings. This is a code review, not the subsequent H200/wheel acceptance review.

## Scope

Reviewed the uncommitted product changes relative to `5172d3c72687478e6037a36001f8473bdc71690f`: source analysis, source insertion, emitters, protocol readers, source loader, shared capture state, and compatibility modules. Also reviewed `tests/validate_component_refactor.py`, `tests/test_components.py`, and its addition to the isolated CPU runner.

The new control-flow design describes future capabilities. This refactor does not implement Group/Pipelined, dynamic bounds, or early transfers, and this PASS makes no such claim.

## Independent checks

- Compared all 21 moved top-level function ASTs against HEAD. They are identical after only normalizing tile emitter `_active`/`_buffers` access to the shared state module and removing the old schema dispatch before comparing the extracted legacy functions. No executable-body differences remain. The new dispatch preserves both schema routes.
- Compared TLDBG1 constants and the extracted TLDBG2 dtype-width map against HEAD: unchanged.
- Independently ran the saved pre-refactor comparison with Python 3.14 and `-X utf8`: all 11 source-plan and injected-source fixtures are identical, including samples, legacy source, and four reviewed examples.
- Ran the existing source/scope suites: 14 scope tests and 9 source-engine tests passed. Ran the new component suite: 2 tests passed.
- Independently checked that compatibility imports resolve to the same emitter/session objects. Tile and sample emitters share the single `capture_state` module, including mixed point state and buffer bindings. Nested sessions are rejected; normal and exceptional exits restore both state variables.
- In a fresh Python process, imported both compatibility protocol readers and checked that neither GPU libraries nor emitters/runtime hooks/capture state were imported.
- Checked setuptools discovery: all five new packages are discovered by the existing `src` configuration. The directories have package initializers. Actual wheel contents remain an acceptance check.
- Inspected generated-source imports and runtime callers: old public module paths remain available, preserving existing saved instrumentation and CLI execution paths. Private `_active` and `_buffers` intentionally live only in `capture_state`; no repository caller still accesses them through the old monitor module.

## Limits and follow-up acceptance

The first local fixture command without UTF-8 mode encountered the existing default-codepage behavior in the external example configuration reader. Repeating with `-X utf8` passed; this is not introduced by the component move.

This review did not run TileLang compilation or H200 kernels. Before release, complete the design's H200 regression set (nested scalar, local, Softmax samples, source fragment, reviewed GQA), independent persisted-evidence checks, and actual wheel-content verification. Existing limitations and record semantics must remain described as unchanged.
