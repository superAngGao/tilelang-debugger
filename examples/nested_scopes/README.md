# Nested source observations

`kernel.py` contains ordinary TileLang kernels with no debugger API calls. `configure.py` resolves fixture statements to original source lines; the user-facing input remains a source filename and line numbers.

The main kernel exercises three serial loops with mixed branches, shadowed induction names, nonzero starts, zero-trip and inactive scopes, and a Parallel tail. `--local` exercises a private local buffer, bool, int64 above 2**53, and three floating dtypes. `reference.py` independently constructs expected keys and values.

From the repository root:

```bash
python examples/nested_scopes/configure.py /tmp/nested.json
python -m tilelang_debugger run examples/nested_scopes/run.py \
  --monitor /tmp/nested.json --output artifacts/nested
python -m tilelang_debugger analyze artifacts/nested \
  --reference examples/nested_scopes/reference.py --output artifacts/nested-analysis
```

For the second kernel add `--local` to configure, and append `-- --local` to the run command. A source thread can be selected with configure's `--thread`; the tail reference deliberately does not guess compiler thread ownership, so use the local fixture or a serial-only point selection for thread-specific offline comparison.

See `tests/validate_nested.py` for all point configurations, including independent selection of both shadowed loop iterations and a proven not-executed branch. Physical execution coverage under Parallel remains unverified, even when all received values match their reference.
