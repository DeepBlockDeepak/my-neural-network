import numpy as np


class SimpleNeuralNetwork:
    def __init__(self, layer_dims):
        self.parameters = {}
        self.L = len(layer_dims)  # number of layers in the network
        self.layer_dims = layer_dims

        # Initialize parameters
        np.random.seed(1)  # Seed the random number generator for consistency
        for l in range(1, self.L):
            self.parameters["W" + str(l)] = (
                np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            )
            self.parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def compute_loss(self, AL, Y):
        # Categorical cross-entropy
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(AL)) / m
        cost = np.squeeze(cost)  # Ensure cost is a scalar value
        return cost

    def forward_propagation(self, X):
        caches = []
        A = X
        L = self.L - 1  # Number of layers

        # Implement [LINEAR -> RELU]*(L-1)
        for l in range(1, L):
            A_prev = A
            W = self.parameters["W" + str(l)]
            b = self.parameters["b" + str(l)]
            Z = np.dot(W, A_prev) + b
            A = np.maximum(0, Z)  # ReLU activation
            caches.append((A_prev, W, b, Z))

        # Implement LINEAR -> SOFTMAX
        W = self.parameters["W" + str(L)]
        b = self.parameters["b" + str(L)]
        Z = np.dot(W, A) + b
        AL = self.softmax(Z)
        caches.append((A, W, b, Z))

        return AL, caches


# Example usage:
layer_dims = [
    5,
    4,
    3,
    2,
]  # 4-layer model with input of size 5, two hidden layers, and an output of size 2
net = SimpleNeuralNetwork(np.array([5, 4, 3, 2]))


print(net.parameters.keys())
