import numpy as np
from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm

from activation_functions import ActivationFunction
from my_neural_network.constants import OptimizerType, TaskType
from my_neural_network.error_handlers import validate_input_shapes


class NeuralNetworkConfig(BaseModel):
    layer_dims: list[int]
    learning_rate: float = Field(0.001, ge=0.0)
    seed: int = Field(42, ge=0)
    mini_batch_size: int = Field(32, ge=1)
    beta1: float = Field(0.9, ge=0.0, le=1.0)  # adam hyperparameter
    beta2: float = Field(0.999, ge=0.0, le=1.0)  # adam hyperparameter
    epsilon: float = Field(1e-8, gt=0.0)  # adam hyperparameter
    optimizer: OptimizerType = OptimizerType.ADAM  # "ADAM" or "SGD",
    task: TaskType = TaskType.CLASSIFICATION  # "classification" or "regression"

    @field_validator("learning_rate", mode="before")
    @classmethod
    def adjust_learning_rate(cls, value: float, info) -> float:
        # access the optimizer from other fields
        optimizer = info.data.get(
            "optimizer", OptimizerType.ADAM
        )  # default to Adam if optimizer isn't set
        if optimizer == OptimizerType.SGD:
            return 0.01 if value is None else value  # default lr for SGD is 0.01
        return value if value is not None else 0.001  # default for Adam is 0.001

    # ensure the layer_dims list has at least two layers (input and output)
    @field_validator("layer_dims", mode="before")
    @classmethod
    def validate_layer_dims(cls, value: list[int]) -> list[int]:
        if len(value) < 2:
            raise ValueError(
                "layer_dims must have at least two elements (input and output layers)."
            )
        return value


class SimpleNeuralNetwork:
    def __init__(self, config: NeuralNetworkConfig):
        self.config = config
        self.parameters = {}
        self.L = len(config.layer_dims)  # number of layers in the network
        self.layer_dims = config.layer_dims

        # initialize parameters
        np.random.seed(self.config.seed)
        for l in range(1, self.L):
            self.parameters["W" + str(l)] = np.random.randn(
                self.layer_dims[l], self.layer_dims[l - 1]
            ) * np.sqrt(
                2 / self.layer_dims[l - 1]
            )  # without He: * 0.01
            self.parameters["b" + str(l)] = np.zeros((self.layer_dims[l], 1))

    def _initialize_adam(self):
        """
        Lazy initializes the first (m) and second (v) moment estimates for Adam.
        """
        self.m = {}
        self.v = {}
        for l in range(1, self.L):
            self.m["dW" + str(l)] = np.zeros_like(self.parameters["W" + str(l)])
            self.m["db" + str(l)] = np.zeros_like(self.parameters["b" + str(l)])
            self.v["dW" + str(l)] = np.zeros_like(self.parameters["W" + str(l)])
            self.v["db" + str(l)] = np.zeros_like(self.parameters["b" + str(l)])

    def _create_mini_batches(
        self, X, Y, mini_batch_size: int, seed=None
    ) -> list[tuple[np.ndarray]]:
        """
        Performs mini-batching of the training data.
        Returns:
            List of mini-batched data and labels
        """
        # setting up params for mini-batching
        if seed is not None:
            np.random.seed(seed)
        m = X.shape[1]  # total number of samples
        mini_batches = []

        # shuffle data with random permutation
        permutation = np.random.permutation(m)
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape(Y.shape[0], m)

        # partition into mini batches
        num_complete_minibatches = m // mini_batch_size
        for k in range(num_complete_minibatches):
            mini_batch_X = shuffled_X[
                :, k * mini_batch_size : (k + 1) * mini_batch_size
            ]
            mini_batch_Y = shuffled_Y[
                :, k * mini_batch_size : (k + 1) * mini_batch_size
            ]
            mini_batches.append((mini_batch_X, mini_batch_Y))

        # handle the end case (the last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size :]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size :]
            mini_batches.append((mini_batch_X, mini_batch_Y))

        return mini_batches

    def compute_loss(self, AL: np.ndarray, Y: np.ndarray) -> float:
        """
        Computes the binary cross-entropy loss or categorical cross-entropy loss depending on network type.

        Binary cross-entropy is the loss function for binary classification problems.
        It measures the performance of a classification model whose output is a probability value between 0 and 1.

        Cross-entropy -> loss function for multi-class classification problems.
        It measures the performance of a classification model whose output is a probability value between 0 and 1.

        Args:
            AL: The activation from the last layer's activation function; the probabilities predicted by the model for the positive class.
            Y: The true labels of the data as a binary vector (e.g., containing 0 for negative class and 1 for positive class).

        Returns:
            float: The average binary cross-entropy loss across all examples in the dataset.
        """
        m = Y.shape[1]  # number of samples

        if (task := self.config.task) == TaskType.CLASSIFICATION:
            if Y.shape[0] == 1:  # Binary cross-entropy
                # 1e-8 is added for numerical stability to avoid log(0)
                cost = (
                    -np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8)) / m
                )
            else:  # categorical cross-entropy for multi-class
                cost = -np.sum(Y * np.log(AL + 1e-8)) / m
        elif task == TaskType.REGRESSION:
            # Mean Squared Error (MSE) loss
            cost = np.sum((AL - Y) ** 2) / (2 * m)
        else:
            raise ValueError(
                "Invalid task type. Must be 'classification' or 'regression'."
            )

        return np.squeeze(cost)  # reduce the dimensionality of the cost to a scalar

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
                - AL: The activation of the output layer:
                                the sigmoid or softmax probability of the output layer, which is the
                                predicted probability for the positive class for each example.
                - caches: A list of tuples needed for backprop, where each tuple contains:
                                (A_prev, W, b, Z) for each layer. These are cached for
                                use in the backward pass:
                                - A_prev: activations from the previous layer
                                - W: weights matrix of the current layer
                                - b: bias vector of the current layer
                                - Z: linear component (W*A_prev + b) of the current layer
        """
        caches = []  # stores (A_prev, W, b, Z) for backprop later
        A = X  # set activation of the input layer to the input data

        # iterate over all hidden layers, excluding the output layer
        for l in range(1, self.L - 1):
            A_prev = A
            W = self.parameters["W" + str(l)]
            b = self.parameters["b" + str(l)]
            Z = np.dot(W, A_prev) + b
            A = ActivationFunction.relu(Z)  # ReLU activation for all hidden layers
            caches.append((A_prev, W, b, Z))

        # final activation leading to output layer
        W = self.parameters["W" + str(self.L - 1)]
        b = self.parameters["b" + str(self.L - 1)]
        Z = np.dot(W, A) + b

        # check numerical stability
        if np.isnan(Z).any() or np.isinf(Z).any():
            print("NaN or Inf detected in Z")

        if (task := self.config.task) == TaskType.CLASSIFICATION:
            # activation func for output layer
            if self.layer_dims[-1] == 1:
                AL = ActivationFunction.sigmoid(
                    Z
                )  # Binary classification (single neuron in output layer)
            else:
                AL = ActivationFunction.softmax(
                    Z
                )  # Multi-class classification (more than one neuron)
        elif task == TaskType.REGRESSION:
            AL = Z  # linear activation for regression (no activation func)
        else:
            raise ValueError(
                "Invalid task type. Must be 'classification' or 'regression'."
            )

        # `A` here is the activation from the last hidden layer.
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
        # Both classification types simplify to using the same gradient function
        # Multi-class classification- softmax activation
        # Binary classification - sigmoid activation for binary cross-entropy loss
        if (task := self.config.task) == TaskType.CLASSIFICATION:
            if (
                AL.shape[0] == 1
            ):  # Binary classification - sigmoid activation for binary cross-entropy loss
                # return -(np.divide(Y, AL + 1e-8) - np.divide(1 - Y, 1 - AL + 1e-8))
                return AL - Y
            else:  # Multi-class classification- softmax activation
                return AL - Y
        elif (
            task == TaskType.REGRESSION
        ):  # regression also simplifies the same, but retaining check anyway
            return AL - Y
        else:
            raise ValueError(
                "Invalid task type. Must be 'classification' or 'regression'."
            )

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
        grads = {}  # cache gradient params to update during gradient descent
        L = self.L  # total number of layers including the input layer
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # ensure Y has the same shape as AL

        # retrieve cache for the output layer, layer L-1
        current_cache = caches[-1]
        A_prev, W, b, Z = current_cache

        # initialize backpropagation for output layer
        # dZ = AL - Y  # for the output layer (layer L-1)
        # compute the initial gradient of the loss
        dZ = self.compute_initial_gradient(AL, Y)

        # calc gradients for the output layer
        grads["dW" + str(L - 1)] = (1 / m) * np.dot(dZ, A_prev.T)
        grads["db" + str(L - 1)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # initialize dA_prev for the previous layer
        dA_prev = np.dot(W.T, dZ)

        # backprop-loop over the hidden layers in reverse order (from L-2 to 1)
        for l in reversed(range(1, L - 1)):
            current_cache = caches[l - 1]  # cache for layer l
            A_prev, W, b, Z = current_cache

            # calc dZ for hidden layer
            dZ = dA_prev * ActivationFunction.relu_derivative(Z)

            # calc gradients
            grads["dW" + str(l)] = (1 / m) * np.dot(dZ, A_prev.T)
            grads["db" + str(l)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            # calc dA_prev for the next layer
            dA_prev = np.dot(W.T, dZ)

        return grads

    def update_parameters(self, grads: dict, t: int) -> None:
        """
        Update the parameters using gradient descent or Adam.

        Args:
            grads: Dictionary with gradients.
            t: time step for Adam.
        """
        optimizer = self.config.optimizer

        if optimizer == OptimizerType.SGD:
            learning_rate = self.config.learning_rate
            for l in range(1, self.L):
                self.parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
                self.parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
        elif optimizer == OptimizerType.ADAM:
            self._update_parameters_with_adam(grads, t)
        else:
            raise ValueError(f"Optimizer '{self.config.optimizer}' is not supported.")

    def _update_parameters_with_adam(self, grads, t):
        """
        Performs the Adam hyperparameter updates.

        Args:
            grads: weights and biases computed during backpropagation.
            t: time step (epoch number) for bias corrections.
        """
        beta1 = self.config.beta1
        beta2 = self.config.beta2
        epsilon = self.config.epsilon
        learning_rate = self.config.learning_rate

        for l in range(1, self.L):
            # moving average, m, of the gradients
            self.m["dW" + str(l)] = (
                beta1 * self.m["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
            )
            self.m["db" + str(l)] = (
                beta1 * self.m["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]
            )

            # bias-corrected first moment estimate
            m_corrected_dW = self.m["dW" + str(l)] / (1 - beta1**t)
            m_corrected_db = self.m["db" + str(l)] / (1 - beta1**t)

            # moving average, v, of the squared gradients
            self.v["dW" + str(l)] = beta2 * self.v["dW" + str(l)] + (
                1 - beta2
            ) * np.square(grads["dW" + str(l)])
            self.v["db" + str(l)] = beta2 * self.v["db" + str(l)] + (
                1 - beta2
            ) * np.square(grads["db" + str(l)])

            # bias-corrected second raw moment estimate
            v_corrected_dW = self.v["dW" + str(l)] / (1 - beta2**t)
            v_corrected_db = self.v["db" + str(l)] / (1 - beta2**t)

            # update parameters
            self.parameters["W" + str(l)] -= learning_rate * (
                m_corrected_dW / (np.sqrt(v_corrected_dW) + epsilon)
            )
            self.parameters["b" + str(l)] -= learning_rate * (
                m_corrected_db / (np.sqrt(v_corrected_db) + epsilon)
            )

    @validate_input_shapes
    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int,
        X_val: np.ndarray = None,
        Y_val: np.ndarray = None,
        patience: int = 10,
    ) -> None:
        """
        This method trains the neural network using gradient descent optimization.
        It iteratively performs forward propagation, computes the loss, backward propagation, and updates the parameters of the network.

        Args:
            X: Input data matrix where each column represents a training example.
            Y: True label matrix where each column represents the true labels for the corresponding training example in X.
            epochs: Number of training epochs.
            X_val: Validation input data (optional).
            Y_val: Validation labels (optional).
            patience: Number of epochs to wait for improvement before stopping early.

        Notes:
            During each iteration:
            - Forward propagation is performed to compute the output predictions (AL) and caches containing intermediate values needed for backward propagation.
            - The loss is computed based on the output predictions (AL) and true labels (Y).
            - Backward propagation is performed to compute gradients of the loss with respect to the parameters of the network.
            - The parameters of the network are updated using gradient descent with the computed gradients and the specified learning rate.

            - Early stopping will only work if validation data is provided.

        """
        optimizer = self.config.optimizer
        t = 0  # initial time step for Adam

        if optimizer == OptimizerType.ADAM:
            self._initialize_adam()

        best_val_loss = float("inf")
        patience_counter = 0
        with tqdm(total=epochs, desc="Training Progress") as pbar:
            for i in range(epochs):
                mini_batches = self._create_mini_batches(
                    X, Y, mini_batch_size=self.config.mini_batch_size, seed=42
                )
                epoch_cost = 0
                for mini_batch_X, mini_batch_Y in mini_batches:
                    AL, caches = self.forward_propagation(mini_batch_X)
                    cost = self.compute_loss(AL, mini_batch_Y)

                    # check for NaN or Inf in the cost, (exploding gradients)
                    if np.isnan(cost) or np.isinf(cost):
                        print(
                            f"NaN or Inf detected in cost at epoch {i + 1}. Stopping training."
                        )
                        pbar.close()
                        return

                    epoch_cost += cost
                    grads = self.backward_propagation(AL, mini_batch_Y, caches)

                    # update parameters according to optimizer method
                    if optimizer == OptimizerType.ADAM:
                        t += 1

                    self.update_parameters(grads, t)

                # Validation phase (if validation data is provided)
                if X_val is not None and Y_val is not None:
                    AL_val, _ = self.forward_propagation(X_val)
                    val_loss = self.compute_loss(AL_val, Y_val)

                    # print(
                    #     f"Epoch {i + 1}: Train Loss = {epoch_cost / len(mini_batches):.4f}, "
                    #     f"Validation Loss = {val_loss:.4f}"
                    # )

                    # Check for early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0  # Reset patience counter
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping triggered at epoch {i + 1}")
                            break

                pbar.update(1)

    @validate_input_shapes
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the outputs for given input data using the trained neural network.

        Args:
            X: Input data (features) where each column represents an example.

        Returns:
            Predicted labels or values for the input data.
        """
        # run forward propagation to get the output activations
        AL, _ = self.forward_propagation(X)

        if (task := self.config.task) == TaskType.CLASSIFICATION:
            # check for output layer/model classification-type
            if self.layer_dims[-1] == 1:
                # for binary classification, uses 0.5 threshold
                predictions = (AL > 0.5).astype(int)
            else:
                # for multiclass-classification, return index of the max probability class
                predictions = np.argmax(AL, axis=0)
        elif task == TaskType.REGRESSION:
            predictions = AL  # continuous values for regression
        else:
            raise ValueError(
                "Invalid task type. Must be 'classification' or 'regression'."
            )

        return predictions
