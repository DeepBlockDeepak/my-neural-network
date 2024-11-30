import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from activation_functions import ActivationFunction
from my_neural_network import NeuralNetworkConfig, SimpleNeuralNetwork
from tests.titanic_data.preprocess import preprocess_data


def test_initialization(create_network):
    """
    Ensure that weights and biases are initialized correctly.
    Checks that each layer has the appropriate dimensions and is stored as numpy arrays.
    """
    network = create_network
    # Verify that all expected weight and bias matrices exist and are of the correct shape
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
    """
    Test the forward propagation function to ensure it computes correctly.
    Checks output dimensions and cache integrity.
    """
    network = create_network
    X = np.random.randn(5, 10)  # Example input (5 features, 10 samples)
    AL, caches = network.forward_propagation(X)
    assert AL.shape == (
        network.layer_dims[-1],
        X.shape[1],
    ), "Output layer dimensions are incorrect."
    assert len(caches) == network.L - 1, "Cache not correctly formed."


def test_sigmoid():
    """
    Tests the sigmoid activation function.
    """
    Z = np.array([0, 2, -2])
    A = ActivationFunction.sigmoid(Z)
    assert np.allclose(
        A, [0.5, 0.88079708, 0.11920292]
    ), "Sigmoid function is incorrect."


def test_softmax():
    """
    Tests the softmax activation function
    """
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


@pytest.mark.parametrize("dtype", [np.float64, np.float32, float, np.int64, np.int32])
def test_data_types(input_data, dtype):
    X, y = input_data
    X = X.astype(dtype)
    y = y.astype(dtype)
    # Set up the network configuration using the new config class
    config = NeuralNetworkConfig(layer_dims=[X.shape[0], 10, 1])
    nn = SimpleNeuralNetwork(config=config)
    try:
        nn.train(X, y, epochs=10)
    except Exception as e:
        pytest.fail(f"Training failed with dtype {dtype}: {str(e)}")


@pytest.mark.parametrize(
    "shape",
    [
        ((40, 100), (1, 100)),  # correct shape
        ((40, 101), (1, 100)),  # mismatched shapes
        ((39, 100), (1, 100)),  # incorrect feature size
    ],
)
def test_input_shapes(shape):
    feature_shape, label_shape = shape
    X = np.random.randn(*feature_shape).astype(np.float64)
    y = np.random.randint(0, 2, label_shape).astype(np.float64)
    # Use the config model to define the network
    config = NeuralNetworkConfig(layer_dims=[X.shape[0], 10, 1])
    nn = SimpleNeuralNetwork(config=config)
    if feature_shape[0] != nn.layer_dims[0] or feature_shape[1] != label_shape[1]:
        with pytest.raises(ValueError):
            nn.train(X, y, epochs=10)
    else:
        nn.train(X, y, epochs=10)  # Should pass without errors


@pytest.mark.parametrize("optimizer", ["gradient_descent", "adam"])
def test_breast_cancer_classification(optimizer):
    # load
    data = load_breast_cancer()
    X = data.data  # shape: (m, n_features) -> (569, 30)
    Y = data.target  # shape: (m,) -> (569,)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # transpose X to match my nn input shape
    X_train = X_train.T  # shape: (n_features, m_train)
    X_test = X_test.T  # shape: (n_features, m_test)
    Y_train = Y_train.reshape(1, -1)  # shape: (1, m_train)
    Y_test = Y_test.reshape(1, -1)  # shape: (1, m_test)

    # configure the network
    config = NeuralNetworkConfig(
        layer_dims=[X_train.shape[0], 64, 32, 1],
        learning_rate=0.01 if optimizer == "gradient_descent" else 0.001,
        optimizer=optimizer,  # 'adam' or 'gradient_descent'
        seed=42,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )
    nn = SimpleNeuralNetwork(config)

    # train, predict, evaluate
    nn.train(X_train, Y_train, epochs=1000)
    predictions = nn.predict(X_test)
    accuracy = accuracy_score(Y_test.flatten(), predictions.flatten())
    # Run with -s flag to see std out
    # poetry run pytest -s tests/test_neuron.py::test_breast_cancer_classification
    print(f"{config.optimizer.upper():>10} Accuracy: {accuracy:.4f}")


@pytest.mark.parametrize("optimizer", ["gradient_descent", "adam"])
def test_titanic_classification(optimizer):
    # load
    train_data = pd.read_csv("tests/titanic_data/train.csv")

    # process data -> obtain pd.DataFrames of feature engineered train/test
    X_train = preprocess_data(train_data)
    y_train: pd.DataFrame = train_data["Survived"]  # extract labels
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=0
    )

    # must use numpy!
    X_train = X_train.to_numpy().T  # features as rows, samples as columns
    X_val = X_val.to_numpy().T

    y_train = y_train.to_numpy().reshape(1, -1)  # reshape to (1, number of samples)
    y_val = y_val.to_numpy().reshape(1, -1)

    # configure the network
    config = NeuralNetworkConfig(
        layer_dims=[X_train.shape[0], 64, 32, 1],
        learning_rate=0.01 if optimizer == "gradient_descent" else 0.001,
        optimizer=optimizer,  # 'adam' or 'gradient_descent'
        seed=42,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )
    nn = SimpleNeuralNetwork(config)

    # train, predict, evaluate
    nn.train(X_train, y_train, epochs=1000)
    predictions = nn.predict(X_val)
    predictions = np.asarray(predictions).flatten()
    predictions = (predictions > 0.5).astype(int)  # convert probabilities to 0 or 1
    accuracy = accuracy_score(y_val.flatten(), predictions.flatten())
    # Run with -s flag to see std out
    # poetry run pytest -s tests/test_neuron.py::test_titanic_classification
    print(f"{config.optimizer.upper():>10} Accuracy: {accuracy:.4f}")
