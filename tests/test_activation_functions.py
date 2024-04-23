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
    """
    Tests the stability of the sigmoid activation function across a range of extreme values.
    The sigmoid function is expected to map any input to the interval [0, 1].
    Test here for extreme inputs like very large positive/negative values or NaN/Inf,
    the function does not produce NaN, Inf, and stays within the [0, 1] range.
    """
    z = np.array([z_value], dtype=np.float64)
    result = ActivationFunction.sigmoid(z)
    assert not np.isnan(result).any(), "Sigmoid should not produce NaN"
    assert not np.isinf(result).any(), "Sigmoid should not produce Inf"
    assert np.all(result >= 0) and np.all(
        result <= 1
    ), "Sigmoid output out of expected range [0, 1]"


@pytest.mark.parametrize("z_value", [0, 1000, -1000, np.inf, -np.inf, np.nan])
def test_relu_stability(z_value):
    """
    Tests the ReLU activation function's response to extreme values and NaN/Inf inputs.
    ReLU should output 0 for any negative input, and for non-negative inputs, it should output the input itself.
    Special cases:
    - Positive infinity (Inf) is expected to be capped at the maximum floating-point value representable by np.float64,
      preventing overflow issues.
    - Negative values and negative infinity should result in zero output.
    - NaN inputs should also ideally result in zero to prevent propagation of NaN values through the network.
    """
    z = np.array([z_value], dtype=np.float64)
    result = ActivationFunction.relu(z)

    if np.isinf(z_value) and z_value > 0:
        assert (
            result[0] == np.finfo(np.float64).max
        ), "ReLU should replace positive inf with max float"
    else:
        assert not np.isinf(result).any(), "ReLU should not produce Inf"
    assert np.all(result >= 0), "ReLU should not produce negative values"
