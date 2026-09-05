# Upload this repository correctly

The GitHub repository should be named:

`magnetic-field-optimisation`

## Recommended method: GitHub Desktop

1. Extract `magnetic-field-optimisation.zip`.
2. Open GitHub Desktop and choose **File -> Add local repository**.
3. Select the extracted `magnetic-field-optimisation` folder.
4. If prompted, choose **Create a repository** using that folder.
5. Set the description to:

   `Physics-constrained inverse design of spatial magnetic fields using validated simulation, constrained optimisation, robustness analysis and ML acceleration.`

6. Publish the repository to `Ali-Mattar-code` and keep it public.

## Browser upload method

1. Create a new empty GitHub repository named `magnetic-field-optimisation`.
2. Do **not** initialise it with a README, `.gitignore`, or licence.
3. Extract the ZIP on your computer.
4. Open the extracted folder, select everything inside it, and drag those items onto GitHub's **uploading an existing file** page.
5. Confirm that `README.md`, `pyproject.toml`, `src`, `tests`, `results`, and `docs` appear at the repository root before committing.

## Important check

The repository homepage must start with the heading:

> Physics-Constrained Magnetic Field Optimisation

If it starts with Python imports or `solve_geometry`, the contents of `experiments/ml_acceleration.py` were accidentally pasted into `README.md`. Replace the root README with the supplied `README.md` file.

## Suggested topics

`physics` `magnetostatics` `inverse-design` `convex-optimization` `machine-learning` `scientific-computing` `biot-savart`

