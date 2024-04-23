import numpy as np
import pytest

from activation_functions import ActivationFunction


def test_activation_function_methods():
    """
    Creates a list of the expected methods and checks that each one
    is present in the ActivationFunction class.
    If any are missing, it fails the assertion and reports which attributes are not found

    """
    expected_methods = [
        "sigmoid",
        "softmax",
        "relu",
        "sigmoid_derivative",
        "relu_derivative",
    ]

    missing_methods = [
        method for method in expected_methods if not hasattr(ActivationFunction, method)
    ]

    assert (
        not missing_methods
    ), f"ActivationFunction is missing methods: {missing_methods}"


@pytest.mark.parametrize("z_value", [0, 1000, -1000, np.inf, -np.inf, np.nan])
def test_sigmoid_stability(z_value):
    z = np.array([z_value], dtype=np.float64)
    result = ActivationFunction.sigmoid(z)
    assert not np.isnan(result).any(), "Sigmoid should not produce NaN"
    assert not np.isinf(result).any(), "Sigmoid should not produce Inf"
    assert np.all(result >= 0) and np.all(
        result <= 1
    ), "Sigmoid output out of expected range [0, 1]"


@pytest.mark.parametrize("z_value", [0, 1000, -1000, np.inf, -np.inf, np.nan])
def test_relu_stability(z_value):
    z = np.array([z_value], dtype=np.float64)
    result = ActivationFunction.relu(z)

    if np.isinf(z_value) and z_value > 0:
        assert (
            result[0] == np.finfo(np.float64).max
        ), "ReLU should replace positive inf with max float"
    else:
        assert not np.isinf(result).any(), "ReLU should not produce Inf"
    assert np.all(result >= 0), "ReLU should not produce negative values"
