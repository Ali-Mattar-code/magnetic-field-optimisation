"""Interactive demonstrator for the inverse-design pipeline."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from magfield.electrical import ohmic_power, resistance_matrix
from magfield.geometry import planar_array
from magfield.metrics import fwhm, leakage_ratio
from magfield.optimisation import solve_currents
from magfield.physics import influence_matrix

st.set_page_config(page_title="Magnetic Field Inverse Design", layout="wide")
st.title("Physics-Constrained Magnetic Field Inverse Design")
st.caption(
    "Clean-room research demonstrator - all candidates are verified by "
    "Biot-Savart simulation."
)

with st.sidebar:
    st.header("Design")
    grid = st.slider("Grid size", 3, 7, 5, step=2)
    radius_mm = st.slider("Coil radius (mm)", 12.0, 28.0, 18.0)
    spacing_mm = st.slider("Coil spacing (mm)", 20.0, 38.0, 26.0)
    depth_mm = st.slider("Target depth (mm)", 35.0, 90.0, 60.0)
    target_ut = st.slider("Target field (uT)", 10.0, 80.0, 50.0)
    current_limit = st.slider("Channel limit (A)", 4.0, 20.0, 12.0)

coils = planar_array(
    grid,
    grid,
    spacing_mm / 1000,
    radius_mm / 1000,
    turns=10,
    segments=72,
)
x = np.linspace(-0.1, 0.1, 201)
points = np.column_stack([x, np.zeros_like(x), np.full_like(x, depth_mm / 1000)])
matrix = influence_matrix(points, coils, component=2)
target = target_ut * 1e-6 * np.exp(-0.5 * (x / 0.018) ** 2)
weights = np.where(np.abs(x) <= 0.018, 1500.0, 30.0)
row = influence_matrix(np.array([[0.0, 0.0, depth_mm / 1000]]), coils, component=2)[0]
resistance = resistance_matrix(coils)
result = solve_currents(
    matrix,
    target,
    weights,
    resistance,
    target_row=row,
    target_value=target_ut * 1e-6,
    max_current=current_limit,
    regularisation=3e-8,
)
profile = matrix @ result.currents

metric_columns = st.columns(4)
metric_columns[0].metric("Target achieved", f"{profile[len(x)//2] * 1e6:.1f} uT")
metric_columns[1].metric("FWHM", f"{fwhm(x, profile) * 1e3:.1f} mm")
metric_columns[2].metric("Ohmic power", f"{ohmic_power(result.currents, resistance):.2f} W")
metric_columns[3].metric("Peak current", f"{np.max(np.abs(result.currents)):.2f} A")

left, right = st.columns([1.5, 1])
with left:
    figure, axis = plt.subplots()
    axis.plot(x * 1e3, target * 1e6, "--", label="Target")
    axis.plot(x * 1e3, profile * 1e6, label="Optimised")
    axis.set(xlabel="Lateral position (mm)", ylabel="Axial field (uT)")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    st.pyplot(figure)
with right:
    st.subheader("Current map")
    st.dataframe(np.round(result.currents.reshape(grid, grid), 2), use_container_width=True)
    st.write(f"Leakage ratio beyond +/-40 mm: **{leakage_ratio(x, profile, 0.040):.3f}**")
    st.info(
        "Simulation only. Driver voltage, thermal transients, and nearby materials "
        "are outside this model."
    )
