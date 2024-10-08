import numpy as np
from pydantic import BaseModel

from activation_functions import ActivationFunction


class NeuralNetworkConfig(BaseModel):
    layer_dims: list[int]
    learning_rate: float = 0.01
    seed: int = 1


class SimpleNeuralNetwork:
    def __init__(self, config: NeuralNetworkConfig):
        self.config = config
        self.parameters = {}
        self.L = len(config.layer_dims)  # number of layers in the network
        self.layer_dims = config.layer_dims

        # Initialize parameters
        np.random.seed(
            self.config.seed
        )  # Seed the random number generator for consistency
        for l in range(1, self.L):
            self.parameters["W" + str(l)] = (
                np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * 0.01
            )
            self.parameters["b" + str(l)] = np.zeros((self.layer_dims[l], 1))

    def compute_loss(self, AL: np.ndarray, Y: np.ndarray) -> float:
        """
        Computes the binary cross-entropy loss or categorical cross-entropy loss depending on network type.

        Binary cross-entropy is the loss function for binary classification problems.
        It measures the performance of a classification model whose output is a probability value between 0 and 1.

        Cross-entropy -> loss function for multi-class classification problems.
        It measures the performance of a classification model whose output is a probability value between 0 and 1.

        Args:
            AL (np.ndarray): The activation from the last layer's activation function; the probabilities predicted by the model for the positive class.
            Y (np.ndarray): The true labels of the data as a binary vector (e.g., containing 0 for negative class and 1 for positive class).

        Returns:
            float: The average binary cross-entropy loss across all examples in the dataset.
        """
        m = Y.shape[1]  # Number of samples
        if Y.shape[0] == 1:  # Binary cross-entropy
            # 1e-8 is added for numerical stability to avoid log(0)
            cost = -np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8)) / m
        else:  # Categorical cross-entropy
            cost = -np.sum(Y * np.log(AL + 1e-8)) / m
        return np.squeeze(cost)  # Reduce the dimensionality of the cost to a scalar

    def forward_propagation(self, X: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Performs the forward pass through the network for binary or multiclass classification.
        Computes the activation at each layer by applying the linear transformation followed by the activation function.

        Iterates over all layers except the last one using ReLU activation.
        Applies sigmoid or softmax activation at the final layer to output a probability for the binary class or multi-class, respectively.

        Args:
            X: Input data (features) where each column represents an example.

        Returns:
            tuple: A tuple containing:
                - AL: The sigmoid or softmax probability of the output layer, which is the
                                predicted probability for the positive class for each example.
                - caches: A list of tuples, where each tuple contains:
                                (A_prev, W, b, Z) for each layer. These are cached for
                                use in the backward pass:
                                - A_prev: activations from the previous layer
                                - W: weights matrix of the current layer
                                - b: bias vector of the current layer
                                - Z: linear component (W*A_prev + b) of the current layer
        """
        caches = []
        A = X
        L = self.L - 1  # Number of layers excluding the output layer

        for l in range(1, L):
            A_prev = A
            W = self.parameters["W" + str(l)]
            b = self.parameters["b" + str(l)]
            Z = np.dot(W, A_prev) + b
            A = ActivationFunction.relu(Z)  # ReLU activation
            caches.append((A_prev, W, b, Z))

        # final activation/output layer
        W = self.parameters["W" + str(L)]
        b = self.parameters["b" + str(L)]
        Z = np.dot(W, A) + b

        if np.isnan(Z).any() or np.isinf(Z).any():
            print("NaN or Inf detected in Z")

        if self.layer_dims[-1] == 1:
            AL = ActivationFunction.sigmoid(
                Z
            )  # Binary classification (single neuron in output layer)
        else:
            AL = ActivationFunction.softmax(
                Z
            )  # Multi-class classification (more than one neuron)

        caches.append((A, W, b, Z))

        return AL, caches

    def compute_initial_gradient(self, AL: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Differentiate between binary cross-entropy and multi-class cross-entropy.

        Args:
            AL: Probability vector, output of the forward propagation (forward_propagation()).
            Y: True "label" vector (for example: containing 0 if non-cat, 1 if cat).

        Returns:
            dAL: Initial gradient of the loss with respect to the activation layer.
        """
        if (
            AL.shape[0] == 1
        ):  # Binary classification - sigmoid activation for binary cross-entropy loss
            return -(np.divide(Y, AL + 1e-8) - np.divide(1 - Y, 1 - AL + 1e-8))
        else:  # Multi-class classification- softmax activation
            return AL - Y

    def backward_propagation(self, AL: np.ndarray, Y: np.ndarray, caches: list) -> dict:
        """
        Perform the backward pass using gradient descent to compute the gradients of the loss function
        with respect to the weights and biases.

        Args:
            AL: Probability vector, output of the forward propagation (forward_propagation()).
            Y: True "label" vector (for example: containing 0 if non-cat, 1 if cat).
            caches: List of caches containing every cache of forward_propagation()
                           with "relu" (it's caches[l], for l in range(L-1) i.e., l = 0...L-2).

        Returns:
            grads: A dictionary with the gradients.
                         grads["dA" + str(l)] = ...
                         grads["dW" + str(l)] = ...
                         grads["db" + str(l)] = ...
        """

        grads = {}
        L = self.L
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # Ensure Y has the same shape as AL

        dAL = self.compute_initial_gradient(AL, Y)

        # Loop from l=L-1 to l=0
        for l in range(L - 1, 0, -1):
            current_cache = caches[l - 1]
            A_prev, W, b, Z = current_cache

            # Lth layer (SIGMOID -> LINEAR) gradients.
            if l == L - 1:
                dZ = dAL * ActivationFunction.sigmoid_derivative(Z)
            else:  # lth layer: (RELU -> LINEAR) gradients.
                dZ = dAL * ActivationFunction.relu_derivative(Z)

            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            if l > 1:
                dAL = np.dot(
                    W.T, dZ
                )  # This will be the dAL for the next iteration (previous layer)

            # Save the gradients for the current layer
            grads["dW" + str(l)] = dW
            grads["db" + str(l)] = db

        return grads

    def update_parameters(self, grads: dict) -> None:
        """
        Update the parameters using gradient descent.

        Args:
            grads: Dictionary with gradients.
            learning_rate: Learning rate of the gradient descent update rule.
        """
        L = self.L

        for l in range(1, L):
            self.parameters["W" + str(l)] -= (
                self.config.learning_rate * grads["dW" + str(l)]
            )
            self.parameters["b" + str(l)] -= (
                self.config.learning_rate * grads["db" + str(l)]
            )

    def train(self, X: np.ndarray, Y: np.ndarray, iterations: int) -> None:
        """
        This method trains the neural network using gradient descent optimization.
        It iteratively performs forward propagation, computes the loss, backward propagation, and updates the parameters of the network.

        Args:
            X: Input data matrix where each column represents a training example.
            Y: True label matrix where each column represents the true labels for the corresponding training example in X.
            iterations: Number of training iterations.

        Notes:
            During each iteration:
            - Forward propagation is performed to compute the output predictions (AL) and caches containing intermediate values needed for backward propagation.
            - The loss is computed based on the output predictions (AL) and true labels (Y).
            - Backward propagation is performed to compute gradients of the loss with respect to the parameters of the network.
            - The parameters of the network are updated using gradient descent with the computed gradients and the specified learning rate.

            After every 100 iterations, the current cost (loss) is printed to monitor the training progress.

        """
        for i in range(iterations):
            # Forward propagation
            AL, caches = self.forward_propagation(X)

            # Compute cost
            # cost = self.compute_loss(AL, Y)

            # Backward propagation
            grads = self.backward_propagation(AL, Y, caches)

            # Update parameters
            self.update_parameters(grads)

            # if i % 100 == 0:  # Print the cost every 100 iterations
            #     print(f"Cost after iteration {i}: {cost}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the outputs for given input data using the trained neural network.

        Args:
            X: Input data (features) where each column represents an example.

        Returns:
            Predicted labels or values for the input data.
        """
        # Perform forward propagation to get the output activations
        AL, _ = self.forward_propagation(X)

        # check for output layer/ model classification-type
        if self.layer_dims[-1] == 1:
            # For binary classification, you might use 0.5 as a threshold
            predictions = (AL > 0.5).astype(int)
        else:
            # For multiclass classification, return the index of the max probability
            predictions = np.argmax(AL, axis=0)

        return predictions
