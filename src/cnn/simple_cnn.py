from cnn import Conv2D
from my_neural_network import SimpleNeuralNetwork


class SimpleNeuralNetworkWithConv(SimpleNeuralNetwork):
    def __init__(self, config, conv_params):
        super().__init__(config)
        self.conv_layer = Conv2D(**conv_params)  # add a convolutional layer

    def forward_propagation(self, X, mode="train"):
        conv_output = self.conv_layer.forward(X)

        # flatten output of convolutional layer (needed for input to FFNN?)
        flat_output = conv_output.reshape(conv_output.shape[0], -1).T

        # pass to fully connected layers
        return super().forward_propagation(flat_output, mode)
