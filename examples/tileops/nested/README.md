# Nested TileOPs samples

These fixtures import the real upstream modules through their package names. `configure.py` resolves source assignments into ordinary schema-2 monitor files; there is no product-side operator dispatch or source hash admission list.

| Case | Selected intermediate values |
| --- | --- |
| pool | Window input, running maximum, NaN flag after each valid element |
| indices | Window input, int64 index inside strict-greater update arm, initial NaN flag |
| rope | Position id, original value, paired value inside the valid THD tail branch |
| softmax | Original fragment element immediately after its write in full/tail branches |

Pool uses a 3×4 finite input and a 3×3 padded window; it does not test NaN propagation. RoPE uses 48 elements in a 64-coordinate Parallel domain. Softmax has 513 columns and 256-column tiles; its padded output is compared in full. CPU references enumerate expected logical coordinates independently from the capture.

Run from the repository root in the documented H200 environment:

```bash
python tests/validate_nested.py --tileops /path/to/TileOPs \
  --output artifacts/nested-validation --sanitizers
```

Use `--case pool`, `--case indices`, `--case rope`, or `--case softmax` to select individual cases. These branch-containing Parallel captures report unverified logical/execution coverage; an exact independent reference key set adds a separate numerical check, without pretending to know physical replica ownership.
