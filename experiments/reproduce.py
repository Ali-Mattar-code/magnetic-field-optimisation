"""Regenerate the reference experiment from the repository root."""

from magfield.experiment import run_experiment


if __name__ == "__main__":
    run_experiment("configs/quick_reproduction.yaml", output_dir="results")

