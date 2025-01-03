import numpy as np


class ActivationFunction:
    @staticmethod
    def sigmoid(Z: np.ndarray) -> np.ndarray:
        """
        Apply the sigmoid activation function to each element in the input array.

        The sigmoid function is defined as 1 / (1 + e^(-Z)), which maps any real
        valued number into the range (0, 1); useful for binary classification.

        Args:
            Z: The input array containing the linear combination of weights
                            and biases at a given layer.

        Returns:
            np.ndarray: Sigmoid function applied to each element of the input array.
        """
        Z = np.clip(Z, -500, 500)  # clip Z to manage large values
        result = 1 / (1 + np.exp(-Z))
        result = np.where(np.isnan(result), 0.0, result)  # replace NaN results with 0
        return result

    @staticmethod
    def softmax(Z: np.ndarray) -> np.ndarray:
        """
        Apply the softmax function to each set of scores in the input array.

        Softmax is used to normalize the input array into a probability distribution
        consisting of K probabilities proportional to the exponentials of the input numbers.
        Used as the activation function for the final layer of a multi-class classification nn.

        Args:
            Z: The input array containing the linear combination of weights
                            and biases of the output layer.

        Returns:
            np.ndarray: The probabilities of each class after applying softmax
                        to the input array.
        """
        expZ = np.exp(Z - np.max(Z))  # shift values for numerical stability
        return expZ / expZ.sum(
            axis=0, keepdims=True
        )  # normalize exponentials for prob distr

    @staticmethod
    def relu(Z) -> np.ndarray:
        """
        Performs ReLU activation for hidden layers.

        Args:
            Z: The input array containing the linear combination of weights
                    and biases at a given layer.
        Returns:
            np.ndarray: Activation layer
        """
        # replace inf with the maximum float
        # prevents the function from propagating infinite values further into the network
        Z = np.where(np.isinf(Z), np.finfo(np.float64).max, Z)
        result = np.maximum(0, Z)  # ReLU operation
        result = np.where(
            np.isnan(result), 0.0, result
        )  # ensure that NaN values do not propagate through the network,
        return result

    @staticmethod
    def sigmoid_derivative(Z: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the sigmoid function with respect to the input Z.

        Args:
            Z: The linear component of the sigmoid function.

        Returns:
            np.ndarray: Derivative of the sigmoid function.
        """
        sig = ActivationFunction.sigmoid(Z)
        return sig * (1 - sig)

    @staticmethod
    def relu_derivative(Z: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the ReLU activation function.

        Args:
            Z: The linear component of the activation function.

        Returns:
            np.ndarray: Derivative of ReLU, 1 for elements of Z > 0; otherwise, 0.
        """
        return (Z > 0).astype(int)
