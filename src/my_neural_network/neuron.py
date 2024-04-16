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

    def sigmoid(self, Z: np.ndarray) -> np.ndarray:
        """
        Apply the sigmoid activation function to each element in the input array.

        The sigmoid function is defined as 1 / (1 + e^(-Z)), which maps any real
        valued number into the range (0, 1); useful for binary classification.

        Args:
            Z (np.ndarray): The input array containing the linear combination of weights
                            and biases at a given layer.

        Returns:
            np.ndarray: The result of applying the sigmoid function to each element of the input array.
        """
        return 1 / (1 + np.exp(-Z))

    def softmax(self, Z: np.ndarray) -> np.ndarray:
        """
        Apply the softmax function to each set of scores in the input array.

        Softmax is used to normalize the input array into a probability distribution
        consisting of K probabilities proportional to the exponentials of the input numbers.
        Used as the activation function for the final layer of a multi-class classification nn.

        Args:
            Z (np.ndarray): The input array containing the linear combination of weights
                            and biases of the output layer.

        Returns:
            np.ndarray: The probabilities of each class after applying softmax
                        to the input array.
        """
        expZ = np.exp(
            Z - np.max(Z)
        )  # Stability improvement: shift values for numerical stability
        return expZ / expZ.sum(
            axis=0, keepdims=True
        )  # Normalize exponentials for probability distribution

    def compute_loss(self, AL: np.ndarray, Y: np.ndarray) -> float:
        """
        Computes the categorical cross-entropy loss.

        Cross-entropy -> loss function for multi-class classification problems.
        It measures the performance of a classification model whose output is a probability value between 0 and 1.

        Args:
            AL (np.ndarray): The activation layer... the probabilities predicted by the model for each class and example,
                             i.e., the output of the last layer's activation function.
            Y (np.ndarray): The true labels of the data in one-hot encoded form.

        Returns:
            float: The average cross-entropy loss across all examples in the dataset.
        """
        # Categorical cross-entropy computation
        m = Y.shape[1]  # Number of samples
        cost = (
            -np.sum(Y * np.log(AL + 1e-8)) / m
        )  # 1e-8 is added for numerical stability to avoid log(0)
        cost = np.squeeze(cost)  # Reduce the dimensionality of the cost to a scalar
        return cost

    def forward_propagation(self, X: np.ndarray) -> tuple:
        """
        Performs the forward pass through the network.
        Computes the activation at each layer by applying the linear transformation then the activation function.

        Iterates over all layers except the last one using ReLU activation.
        Applies softmax activation at the final layer to output probabilities for classification.

        Args:
            X (np.ndarray): Input data (features) where each column represents an example.

        Returns:
            tuple: A tuple containing:
                - AL (np.ndarray): The softmax probabilities of the output layer, which are the
                                   predicted probabilities for each class for each example.
                - caches (list): A list of tuples, where each tuple contains:
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

        # Iterate through each layer up to the second to last layer to apply ReLU activation
        for l in range(1, L):
            A_prev = A
            W = self.parameters["W" + str(l)]
            b = self.parameters["b" + str(l)]
            Z = np.dot(W, A_prev) + b
            A = np.maximum(0, Z)  # ReLU activation
            caches.append((A_prev, W, b, Z))

        # Final layer uses a softmax activation function
        W = self.parameters["W" + str(L)]
        b = self.parameters["b" + str(L)]
        Z = np.dot(W, A) + b
        AL = self.softmax(Z)
        caches.append((A, W, b, Z))

        return AL, caches

    def compute_loss_binary(self, AL: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute the binary cross-entropy loss.

        Binary cross-entropy is the loss function for binary classification problems.
        It measures the performance of a classification model whose output is a probability value between 0 and 1.

        Args:
            AL (np.ndarray): The activation from the last layer's activation function; the probabilities predicted by the model for the positive class.
            Y (np.ndarray): The true labels of the data as a binary vector (e.g., containing 0 for negative class and 1 for positive class).

        Returns:
            float: The average binary cross-entropy loss across all examples in the dataset.
        """
        m = Y.shape[1]  # Number of samples
        cost = -np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8)) / m
        cost = np.squeeze(cost)  # Reduce the dimensionality of the cost to a scalar
        return cost

    def forward_propagation_binary(self, X: np.ndarray) -> tuple:
        """
        Performs the forward pass through the network for binary classification.
        Computes the activation at each layer by applying the linear transformation followed by the activation function.

        Iterates over all layers except the last one using ReLU activation.
        Applies sigmoid activation at the final layer to output a probability for the binary class.

        Args:
            X (np.ndarray): Input data (features) where each column represents an example.

        Returns:
            tuple: A tuple containing:
                - AL (np.ndarray): The sigmoid probability of the output layer, which is the
                                predicted probability for the positive class for each example.
                - caches (list): A list of tuples, where each tuple contains:
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
            A = np.maximum(0, Z)  # ReLU activation
            caches.append((A_prev, W, b, Z))

        W = self.parameters["W" + str(L)]
        b = self.parameters["b" + str(L)]
        Z = np.dot(W, A) + b
        AL = self.sigmoid(Z)
        caches.append((A, W, b, Z))

        return AL, caches

    def relu_derivative(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the ReLU activation function.

        Args:
            Z (np.ndarray): The linear component of the activation function.

        Returns:
            np.ndarray: Derivative of ReLU, 1 for elements of Z > 0; otherwise, 0.
        """
        return (Z > 0).astype(int)

    def sigmoid_derivative(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the sigmoid function with respect to the input Z.

        Args:
            Z (np.ndarray): The linear component of the sigmoid function.

        Returns:
            np.ndarray: Derivative of the sigmoid function.
        """
        sig = self.sigmoid(Z)
        return sig * (1 - sig)

    def backward_propagation(self, AL: np.ndarray, Y: np.ndarray, caches: list) -> dict:
        """
        Perform the backward pass using gradient descent to compute the gradients of the loss function
        with respect to the weights and biases.

        Args:
            AL (np.ndarray): Probability vector, output of the forward propagation (L_model_forward()).
            Y (np.ndarray): True "label" vector (for example: containing 0 if non-cat, 1 if cat).
            caches (list): List of caches containing every cache of linear_activation_forward()
                           with "relu" (it's caches[l], for l in range(L-1) i.e., l = 0...L-2).

        Returns:
            grads (dict): A dictionary with the gradients.
                         grads["dA" + str(l)] = ...
                         grads["dW" + str(l)] = ...
                         grads["db" + str(l)] = ...
        """

        grads = {}
        L = self.L
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # Ensure Y has the same shape as AL

        # Initializing the backpropagation
        dAL = -(np.divide(Y, AL + 1e-8) - np.divide(1 - Y, 1 - AL + 1e-8))

        # Loop from l=L-1 to l=0
        for l in range(L - 1, 0, -1):
            current_cache = caches[l - 1]
            A_prev, W, b, Z = current_cache

            # Lth layer (SIGMOID -> LINEAR) gradients.
            if l == L - 1:
                dZ = dAL * self.sigmoid_derivative(Z)
            else:  # lth layer: (RELU -> LINEAR) gradients.
                dZ = dAL * self.relu_derivative(Z)

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

    def update_parameters(self, grads: dict, learning_rate: float) -> None:
        """
        Update the parameters using gradient descent.

        Args:
            grads (dict): Dictionary with gradients.
            learning_rate (float): Learning rate of the gradient descent update rule.
        """
        L = self.L

        for l in range(1, L):
            self.parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
            self.parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    def train(self, X, Y, iterations, learning_rate):
        for i in range(iterations):
            # Forward propagation
            AL, caches = self.forward_propagation(X)

            # Compute cost
            cost = self.compute_loss(AL, Y)

            # Backward propagation
            grads = self.backward_propagation(AL, Y, caches)

            # Update parameters
            self.update_parameters(grads, learning_rate)

            if i % 100 == 0:  # Print the cost every 100 iterations
                print(f"Cost after iteration {i}: {cost}")


if __name__ == "__main__":
    # Define the architecture
    layer_dims = [5, 4, 3, 2]  # Example architecture
    net = SimpleNeuralNetwork(layer_dims)

    # Create some sample data
    X_sample = np.random.rand(
        layer_dims[0], 100
    )  # 100 examples with features equal to size of input layer
    Y_sample = np.random.rand(
        layer_dims[-1], 100
    )  # 100 labels with features equal to size of output layer

    # Train the network
    net.train(X_sample, Y_sample, iterations=1000, learning_rate=0.0075)
