import numpy as np
import pytest

from activation_functions import ActivationFunction
from my_neural_network import SimpleNeuralNetwork


def test_neuron_imports():
    """
    Creates a list of the expected methods and checks that each one
    is present in the SimpleNeuralNetwork class.
    If any are missing, it fails the assertion and reports which attributes are not found
    """
    expected_attributes = [
        "forward_propagation",
        "compute_loss",
        "backward_propagation",
        "update_parameters",
        "train",
    ]

    missing_attributes = [
        attr for attr in expected_attributes if not hasattr(SimpleNeuralNetwork, attr)
    ]

    assert not missing_attributes, f"Neuron is missing attributes: {missing_attributes}"


@pytest.fixture(
    params=[
        [5, 10, 1],  # Binary classifier configuration
        [5, 10, 3],  # Multi-class classifier configuration
    ]
)
def create_network(request):
    layer_dims = request.param
    network = SimpleNeuralNetwork(layer_dims)
    return network


def test_initialization(create_network):
    network = create_network
    assert all(
        key in network.parameters
        for l in range(1, network.L)
        for key in (f"W{l}", f"b{l}")
    ), "Weights and biases are not initialized for all layers."
    assert all(
        isinstance(network.parameters[f"W{l}"], np.ndarray)
        and network.parameters[f"W{l}"].shape
        == (network.layer_dims[l], network.layer_dims[l - 1])
        for l in range(1, network.L)
    ), "Weights are not initialized correctly."
    assert all(
        isinstance(network.parameters[f"b{l}"], np.ndarray)
        and network.parameters[f"b{l}"].shape == (network.layer_dims[l], 1)
        for l in range(1, network.L)
    ), "Biases are not initialized correctly."


def test_forward_propagation(create_network):
    network = create_network
    X = np.random.randn(5, 10)  # Example input (5 features, 10 samples)
    AL, caches = network.forward_propagation(X)
    assert AL.shape == (
        network.layer_dims[-1],
        X.shape[1],
    ), "Output layer dimensions are incorrect."
    assert len(caches) == network.L - 1, "Cache not correctly formed."


def test_sigmoid():
    Z = np.array([0, 2, -2])
    A = ActivationFunction.sigmoid(Z)
    assert np.allclose(
        A, [0.5, 0.88079708, 0.11920292]
    ), "Sigmoid function is incorrect."


def test_softmax():
    Z = np.array([[0, 1], [2, 1]])
    A = ActivationFunction.softmax(Z)
    expected_output = np.array([[0.11920292, 0.5], [0.88079708, 0.5]])
    assert np.allclose(A, expected_output), "Softmax function is incorrect."


def test_backward_propagation(create_network):
    """
    Checks that all expected gradients are present in the returned gradients dictionary.
    This ensures that the backward propagation logic is functioning correctly
    and that it is providing all necessary information for parameter updates.
    """
    network = create_network
    X = np.random.randn(
        network.layer_dims[0], 10
    )  # Generates a random input matrix X with a shape of the network’s input layer dimension. 10 samples/batch size
    num_classes = network.layer_dims[-1]

    # Prepare labels Y based on number of output classes
    if num_classes == 1:
        Y = np.random.randint(
            0, 2, (1, 10)
        )  # single row of Binary labels with vals of [0,1]
    else:  # num_classes is the number of different classes the network can classify.
        # Generate random labels for multi-class, and convert to one-hot
        labels = np.random.randint(0, num_classes, (10,))  # Generate random labels
        Y = np.zeros((num_classes, 10))
        Y[labels, np.arange(10)] = 1  # Proper one-hot encoding
        # Together, `labels` and `np.arange(10)` form a set of index pairs
        # that specify which elements of the array Y to set to 1.
        # For each column  j (ranging from 0 to 9), the element at row labels[j]
        # is set to 1. This places a 1 in the position corresponding to the class label for each sample.

    AL, caches = network.forward_propagation(X)
    grads = network.backward_propagation(AL, Y, caches)
    assert all(
        f"dW{l}" in grads and f"db{l}" in grads for l in range(1, network.L)
    ), "Gradient keys missing."
