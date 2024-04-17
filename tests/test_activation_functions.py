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
