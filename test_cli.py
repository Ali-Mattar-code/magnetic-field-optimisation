from magfield.cli import main


def test_validation_cli():
    assert main(["validate", "--segments", "192"]) == 0

