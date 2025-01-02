import numpy as np


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        Creates a convolutional layer.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels/filters
            kernel_size: Size of each kernel
            stride: Slider amount for the kernel.
            padding: Number of "pixels" to pad each side of input.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # init kernels and biases
        self.kernels = (
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        )
        self.biases = np.zeros((out_channels, 1))

    def _pad_input(self, X):
        """
        Adds 0-padding to the input tensor.

        Args:
            X: Input tensor with shape (batch_size, in_channels, height, width).

        Returns:
            The padded tensor.
        """
        if self.padding > 0:
            return np.pad(
                X,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
            )
        return X

    def forward(self, X):
        """
        Performs the forward pass for the convolutional layer.

        Args:
            X: Input tensor with shape (batch_size, in_channels, height, width).

        Returns:
            Output tensor after convolution.
        """
        # cache padded input for backpropagation
        self.X_padded = self._pad_input(X)
        # batch_size, _, height, width = self.X_padded.shape

        # calculate output dimensions from padded input
        # out_height = (height - self.kernel_size) // self.stride + 1
        # out_width = (width - self.kernel_size) // self.stride + 1

        # convolve
        output = np.einsum("bijk,oikl->bokl", self.X_padded, self.kernels)
        output += self.biases.reshape(
            1, -1, 1, 1
        )  # broadcast biases to match output shape

        self.output = output  # cache output for backpropagation

        return output

    def compute_initial_gradient(self, AL, Y):
        """
        Computes the initial gradient of the loss with respect to the activations.

        Args:
            AL: Activations from the current layer.
            Y: Ground truth labels.

        Returns:
            Gradient of the loss with respect to the layer output (dA).
        """
        return AL - Y

    def backward(self, dA):
        """
        Performs the backward pass to compute gradients of weights, biases, and the input.

        Args:
            dA: Gradient of the loss with respect to the output of the current layer.

        Returns:
            Tuple (dA_prev, dW, db):
                dA_prev: Gradient of the loss with respect to the input of this layer.
                dW: Gradient of the loss with respect to the weights.
                db: Gradient of the loss with respect to the biases.
        """
        # batch_size, _, out_height, out_width = dA.shape
        # _, _, height, width = self.X_padded.shape

        # init gradients for weights, biases, and input
        dA_prev = np.zeros_like(self.X_padded)  # same shape as the padded input
        dW = np.zeros_like(self.kernels)
        db = np.zeros_like(self.biases)

        # compute gradients
        # gradient of weights
        dW = np.einsum("bijk,bokl->oikl", self.X_padded, dA)

        # gradient of biases
        db = np.sum(dA, axis=(0, 2, 3), keepdims=True)

        # gradient of input (dA_prev)
        dA_prev = np.einsum("oikl,bokl->bijpq", self.kernels, dA)

        # remove padding from dA_prev if it was added during forward pass
        if self.padding > 0:
            dA_prev = dA_prev[
                :, :, self.padding : -self.padding, self.padding : -self.padding
            ]

        return dA_prev, dW, db
