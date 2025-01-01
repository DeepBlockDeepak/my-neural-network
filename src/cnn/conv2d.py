import numpy as np


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
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
        Perform forward propagation for convolutional layer.

        Args:
            X: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            np.ndarray: Output tensor after convolution
        """
        batch_size, in_channels, height, width = X.shape
        assert (
            in_channels == self.in_channels
        ), "Input channels must match kernel channels."

        # calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # apply padding
        X_padded = self._pad_input(X)

        # allocate output tensor to store convolution results
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        # convolve
        for b in range(batch_size):  # iterate over each sample in batch
            for o in range(
                self.out_channels
            ):  # for each filter (each filter makes one feature map)
                for i in range(
                    out_height
                ):  # traverse over the locations of the feature map
                    for j in range(out_width):
                        # region of the input matrix that the kernel processes
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        # extract the input region
                        input_region = X_padded[b, :, h_start:h_end, w_start:w_end]

                        # element-wise multiplication and summation
                        output[b, o, i, j] = (
                            np.sum(input_region * self.kernels[o, :, :, :])
                            + self.biases[o]
                        )

        return output
