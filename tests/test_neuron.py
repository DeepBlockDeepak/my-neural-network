import numpy as np

from src.my_neural_network.neuron import SimpleNeuralNetwork


def test_compute_loss():
    # Setup
    y_true = np.array([[1, 0, 0]])  # Assuming three classes and one example
    y_pred = np.array([[0.7, 0.2, 0.1]])  # Sample prediction

    # Expected output
    expected_loss = -np.log(0.7)  # Only the first class contributes to loss

    # Initialize the network and compute loss
    net = SimpleNeuralNetwork(layer_dims=[3, 3, 3])
    loss = net.compute_loss(y_pred, y_true)

    # Assert
    assert np.isclose(
        loss, expected_loss
    ), f"The computed loss is incorrect: {loss:.2f} vs {expected_loss:.2f}"
