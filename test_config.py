from pathlib import Path

from magfield.config import load_config
from magfield.experiment import build_problem


def test_quick_config_is_complete():
    path = Path(__file__).parents[1] / "configs" / "quick_reproduction.yaml"
    config = load_config(path)
    assert config["geometry"]["turns"] == 10
    assert config["field"]["target_tesla"] == 5e-5


def test_configured_leakage_weight_is_applied():
    path = Path(__file__).parents[1] / "configs" / "quick_reproduction.yaml"
    config = load_config(path)
    problem = build_problem(config)
    outside = abs(problem["x"]) > config["field"]["target_sigma_m"]
    assert set(problem["weights"][outside]) == {config["optimisation"]["leakage_weight"]}
