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
        # apply padding
        self.X_padded = self._pad_input(X)
        self.X = X  # cache original input for backpropagation

        # calculate output dimensions from padded input
        batch_size, _, height, width = self.X_padded.shape
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1

        # allocate output tensor to store convolution results
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        # convolve
        # iterate over each sample in batch
        for b in range(batch_size):
            # for each filter (each filter makes one feature map)
            for o in range(self.out_channels):
                # traverse over the locations of the feature map
                for i in range(out_height):
                    for j in range(out_width):
                        # region of the input matrix that the kernel processes
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        # extract the input region
                        input_region = self.X_padded[b, :, h_start:h_end, w_start:w_end]
                        # element-wise multiplication and summation
                        output[b, o, i, j] = (
                            np.sum(input_region * self.kernels[o]) + self.biases[o]
                        )

        self.output = output  # cache output for backpropagation

        return output

    def backward(self, dA):
        batch_size, _, out_height, out_width = dA.shape
        _, _, height, width = self.X_padded.shape

        dA_prev = np.zeros_like(self.X_padded)
        dW = np.zeros_like(self.kernels)
        db = np.zeros_like(self.biases)

        for b in range(batch_size):
            for o in range(self.out_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        input_region = self.X_padded[b, :, h_start:h_end, w_start:w_end]

                        # gradients
                        dW[o] += input_region * dA[b, o, i, j]
                        db[o] += dA[b, o, i, j]  # db[o] += np.sum(dA[:, o, :, :])
                        dA_prev[b, :, h_start:h_end, w_start:w_end] += (
                            self.kernels[o] * dA[b, o, i, j]
                        )

        # remove padding from dA_prev
        if self.padding > 0:
            dA_prev = dA_prev[
                :, :, self.padding : -self.padding, self.padding : -self.padding
            ]

        return dA_prev, dW, db
