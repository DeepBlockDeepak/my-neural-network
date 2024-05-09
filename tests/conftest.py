import numpy as np
import pytest

from my_neural_network import SimpleNeuralNetwork


@pytest.fixture
def input_data():
    # Creates test data and labels with a specified dtype
    X = np.random.randn(40, 100).astype(np.float64)  # 40 features, 100 samples
    y = np.random.randint(0, 2, (1, 100)).astype(np.float64)  # Binary labels
    return X, y


@pytest.fixture(
    params=[
        [5, 10, 1],  # Binary classifier configuration (** 1 ** output neuron)
        [5, 10, 3],  # Multi-class classifier configuration
    ]
)
def create_network(request):
    layer_dims = request.param
    network = SimpleNeuralNetwork(layer_dims)
    return network
