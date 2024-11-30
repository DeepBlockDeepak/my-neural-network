import numpy as np


# for use on SimpleNeuralNetwork class methods
def validate_input_shapes(func):
    def wrapper(self, X, *args, **kwargs):
        # check that X has correct number of features
        if X.shape[0] != self.layer_dims[0]:
            raise ValueError(
                f"Number of features in X ({X.shape[0]}) does not match the network's expected "
                f"input dimension ({self.layer_dims[0]})."
            )

        # FOLLOWING Section conforms for train(X,Y), and not predict(X)
        # if 'Y' is passed explicitly or as a keyword argument, validate it
        Y = kwargs.get("Y", None)
        if len(args) > 0 and isinstance(args[0], np.ndarray):
            Y = args[0]  # extract Y from positional arguments if provided

        if Y is not None:  # must avoid ambiguity for NumPy arrays' truthiness
            if X.shape[1] != Y.shape[1]:
                raise ValueError(
                    f"Number of samples in X ({X.shape[1]}) and Y ({Y.shape[1]}) must be equal."
                )
            if Y.shape[0] != self.layer_dims[-1]:
                raise ValueError(
                    f"Number of output units in Y ({Y.shape[0]}) does not match the network's "
                    f"output layer dimension ({self.layer_dims[-1]})."
                )
        return func(self, X, *args, **kwargs)

    return wrapper
