# Generated reference outputs

These files were generated from the committed public implementation on 2026-09-04.

- `reproduction_summary.json`, `field_profile.csv`, and `pareto_points.csv` come from `configs/quick_reproduction.yaml`.
- `freeform_comparison.json` is explicitly exploratory because its conductor bases are not cost-equivalent.
- `ml_benchmark.json` records a small local timing benchmark; speed depends on hardware and software versions.
- `figures/` contains views of the same generated data.

Regenerate the core experiment with:

```bash
magfield reproduce --config configs/quick_reproduction.yaml
```

