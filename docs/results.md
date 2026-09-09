# Results guide

Run `magfield reproduce --config configs/quick_reproduction.yaml` to regenerate the committed artefacts. The JSON summary is canonical; charts are views of the same stored data. The summary includes weighted-operator rank, nullity, condition number and the relative singular spectrum; `figures/singular_spectrum.png` visualises the same values.

Do not compare a regenerated result with a historical thesis number unless the geometry, conductor length, number of turns, sampling grid, constraints, and baseline definition are matched.
