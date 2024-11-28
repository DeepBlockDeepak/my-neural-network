import numpy as np
import pytest

from my_neural_network import NeuralNetworkConfig, SimpleNeuralNetwork


@pytest.fixture
def input_data():
    # Creates test data and labels with a specified dtype
    X = np.random.randn(40, 100).astype(np.float64)  # 40 features, 100 samples
    y = np.random.randint(0, 2, (1, 100)).astype(np.float64)  # Binary labels
    return X, y


@pytest.fixture(
    params=[
        {
            "layer_dims": [5, 10, 1],
            "optimizer": "gradient_descent",
        },  # original optimizer
        {"layer_dims": [5, 10, 1], "optimizer": "adam"},
    ]
)
def create_network(request):
    params = request.param
    config = NeuralNetworkConfig(
        layer_dims=params["layer_dims"],
        optimizer=params["optimizer"],
        learning_rate=0.01 if params["optimizer"] == "gradient_descent" else 0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )
    network = SimpleNeuralNetwork(config=config)
    return network
