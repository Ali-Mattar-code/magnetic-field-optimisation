from magfield.validation import run_validation


def test_reference_validation_passes():
    summary = run_validation(segments=256)
    assert summary.passed
    assert summary.single_loop_max_relative_error < 0.01

