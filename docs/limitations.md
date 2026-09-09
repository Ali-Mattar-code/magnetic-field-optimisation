# Limitations

- The default experiments are simulations, not a substitute for benchtop measurement.
- The conductor model assumes thin wire and quasi-static, linear media.
- Driver voltage, thermal transients, skin effect, nearby ferromagnetic materials, and eddy currents are not included.
- Pose-error robustness uses a tractable approximation for the quick workflow.
- The quick reproduction samples one axial field profile; its 10-dimensional numerical null space means field-equivalent current patterns are not uniquely identifiable from that observation set. Regularisation selects a stable solution but does not create missing information.
- Free-form comparisons use different conductor bases and must not be interpreted as equal-manufacturing-cost claims.
- Surrogate benchmarks are small and hardware-dependent; speedup numbers should be regenerated locally.
- The historical thesis results are not treated as reproduced evidence.
