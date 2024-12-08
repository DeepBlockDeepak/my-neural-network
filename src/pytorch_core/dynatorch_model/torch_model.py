import torch.nn as nn


class DynaTorchModel(nn.Module):
    def __init__(self, layer_dims, task_type="classification"):
        super(DynaTorchModel, self).__init__()  # super().__init__()

        layers = []
        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]

            # add a linear layer
            layers.append(nn.Linear(in_dim, out_dim))

            # if not the output layer, add a ReLU activation
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())

        # handle output activation based on the task
        if task_type == "classification":
            if layer_dims[-1] == 1:
                # Binary classification
                # alternatively, omit the sigmoid and use BCEWithLogitsLoss()
                layers.append(nn.Sigmoid())
                self.loss_fn = nn.BCELoss()
            else:
                # Multi-class classification
                # using raw logits so no need for activation func
                # layers.append(nn.Softmax(dim=1))
                # self.loss_fn = nn.NLLLoss()
                self.loss_fn = nn.CrossEntropyLoss()
        elif task_type == "regression":
            # no activation for regression, or just leave it linear?
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
