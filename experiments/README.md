# Fragment print synchronization experiment

`print_sync_probe.py` isolates the shared-memory staging inside `T.print(fragment)`.
It is an experiment, not a production patch or a full GQA validation.

## H200 results — 2026-10-08

Environment: NVIDIA H200, driver 595.71.05, TileLang 0.1.12, PyTorch
2.10.0+cu129, nvcc 13.2.78, Compute Sanitizer 2026.1.1.0.
The installed `print_op.py` SHA-256 was
`55d1d925f24f2f0567744191d1af1089da74dea70d62fb32fd52d37bd6800f5b`.
Installed-source snapshots are retained with the local raw evidence.

Two execution domains were tested:

- `cta`: one block of 128 threads.
- `wg1`: one block of 384 threads, with the experiment confined to `T.ws(1)`
  (threads 128–255), printing from thread 128. The other groups do no work;
  this isolates the subgroup behavior, not the complete GQA protocol.

Each case prints 256 known int32 values, repeated over three kernel launches.
The parser checks all 768 `(element index, value)` records as a multiset,
including duplicates and missing records. The kernel also copies its fragment
to a separate global output which is checked against the input.

| Mode | Automatic ThreadSync | Explicit synchronization | CTA racecheck reports | WG1 racecheck reports | Printed values |
|---|---|---|---:|---:|---|
| `auto` | on | none | 0 | 0 | 768/768 correct in each case |
| `disabled` | off | none | 3 | 3 | 768/768 correct in each case |
| `surround` | off | before and after the whole T.print call | 3 | 3 | 768/768 correct in each case |
| `patched` | off | after staging copy and after printing, inside replacement helper | 0 | 0 | 768/768 correct in each case |

The three error reports correspond to the three launches. Each groups multiple
shared-memory write/read hazards; this is not a count of three individual
conflicting element accesses. Racecheck exits with the configured status 86
for the two unsafe modes. All global output checks passed.

No incorrect printed value was observed in these runs. The evidence is a
detected synchronization hazard, not a reproduced visible corruption. Default
automatic synchronization passed this experiment; these results do not show
a general failure of TileLang's default printing behavior.

## Generated code evidence

In `surround`, the relevant generated order is:

```text
bar.sync
fragment -> shared stores
leader reads shared and prints
bar.sync
```

In `patched`, it is:

```text
fragment -> shared stores
bar.sync
leader reads shared and prints
bar.sync
```

The default WG1 case automatically generates a partial synchronization with
128 participants between the stores and the reads. The patched case explicitly
uses the otherwise unused barrier 14 with 128 participants.

## Monkey patch

The experiment temporarily replaces
`tilelang.language.print_op.print_fragment_buffer_with_condition` with a Python
helper invoking a `T.macro`. The public `T.print` call remains unchanged. The
replacement emits the same shared staging and printing, with explicit existing
`T.sync_threads` operations. It uses the original `shared` storage scope.

The replacement is installed before building/compiling the kernel and restored
in a `finally` block afterwards. No TileLang installation files or lowering
passes are modified. Every worker uses a separate process. A compiled kernel
retains its compiled instrumentation after the Python helper is restored.

Barrier 14 and the count 128 are valid only for these isolated cases. Applying
this to arbitrary kernels requires checking participant domains, uniform
control flow, barrier availability and original asynchronous completion. This
experiment does not validate those decisions for the complete GQA kernel,
concurrent compilations, multiple monitor sites or shared scratch reuse within
a kernel loop. The trailing barrier is present but its reuse behavior is not
independently stressed here.

## Reproduce

Run in an environment with TileLang, PyTorch, CUDA and an available H200:

```bash
CUDA_VISIBLE_DEVICES=2 python experiments/print_sync_probe.py matrix \
  --output artifacts/print-sync/plain

CUDA_VISIBLE_DEVICES=2 python experiments/print_sync_probe.py matrix \
  --sanitizer /usr/local/cuda/bin/compute-sanitizer \
  --output artifacts/print-sync/racecheck
```

Select an available device; the number above is only an example. Each case has
a 120-second timeout. Output includes the source snapshot, frontend IR,
generated CUDA, raw stdout/stderr, environment, execution checks, racecheck log
and a machine-readable summary. The matrix driver records child failures in
`summary.json`; its own exit status is not an overall pass/fail verdict.

Local evidence from this session is under `artifacts/print-sync-20261008/`:

- `race-cta/` and `race-wg1/`: automatic, disabled and surrounding-sync cases.
- `race-patched-shared/`: final patch using the same shared scope as upstream.
- `plain-cta/`: initial non-sanitized observations.

Earlier `patched` subdirectories in `race-cta/`, `race-wg1/` and `plain-cta/`
used the allocation helper's default `shared.dyn` scope. They are retained as
history but are not the patched evidence used in the table above. The final
patch was rerun with explicit `shared` to remove that confounding difference.
