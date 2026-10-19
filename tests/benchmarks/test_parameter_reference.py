"""The parameter audit must reject errors that preserve aggregate gradient norms."""

import pytest

from benchmarks.parameter_gradient_reference import compare_tensors


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def test_equal_norm_opposite_gradient_is_rejected(torch):
    gradient = torch.tensor([1.0, -1.0])
    with pytest.raises(AssertionError):
        compare_tensors(gradient, -gradient)


def test_optimizer_state_scale_error_is_rejected(torch):
    moment = torch.tensor([0.2, 0.4])
    with pytest.raises(AssertionError):
        compare_tensors(moment, moment * 2)


def test_reference_checks_dtype_and_nonfinite_values(torch):
    value = torch.tensor([0.25])
    assert compare_tensors(value, value.clone())["exact"]
    with pytest.raises(ValueError, match="layout"):
        compare_tensors(value, value.double())
    with pytest.raises(AssertionError):
        compare_tensors(value, torch.tensor([float("nan")]))
