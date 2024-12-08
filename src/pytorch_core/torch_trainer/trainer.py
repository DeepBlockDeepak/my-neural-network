from typing import Optional

import torch
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        device: str = None,  # 'cuda' if available, otherwise 'cpu'
        patience: Optional[int] = 10,  # set to None to disable,
        learning_rate: float = 0.001,
        scheduler_type: Optional[str] = None,
        scheduler_params: Optional[dict] = None,  # parameters for the scheduler
    ):
        # automatically use GPU if available
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(device)
        self.learning_rate = learning_rate
        self.optimizer = optimizer or torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )
        # set up the scheduler dynamically based on the type and parameters
        self.scheduler = None
        if scheduler_type:
            self.scheduler = self._create_scheduler(scheduler_type, scheduler_params)

        self.patience = patience  # early stopping params
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _create_scheduler(self, scheduler_type: str, scheduler_params: Optional[dict]):
        """
        Create a scheduler dynamically based on the specified type and parameters,
        using defaults if parameters are not provided.
        """
        if scheduler_params is None:
            scheduler_params = {}

        if scheduler_type == "ReduceLROnPlateau":
            defaults = {"mode": "min", "factor": 0.1, "patience": 10, "threshold": 1e-4}
            params = {
                **defaults,
                **scheduler_params,
            }  # Merge defaults with provided params
            return ReduceLROnPlateau(self.optimizer, **params)
        elif scheduler_type == "StepLR":
            defaults = {"step_size": 10, "gamma": 0.1}
            params = {**defaults, **scheduler_params}
            return StepLR(self.optimizer, **params)
        elif scheduler_type == "ExponentialLR":
            defaults = {"gamma": 0.9}
            params = {**defaults, **scheduler_params}
            return ExponentialLR(self.optimizer, **params)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def fit(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        X_val: torch.Tensor = None,
        Y_val: torch.Tensor = None,
    ):
        """
        Train the model on the provided training data,
        optionally validating on a validation set.
        """
        # set up the optimizer if not provided
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )

        # DataLoader for training data - helps minibatching loading
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # DataLoader for validation data if provided
        val_loader = None
        if X_val is not None and Y_val is not None:
            val_dataset = TensorDataset(X_val, Y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        progress = tqdm(range(epochs), desc="Train Prog", leave=True)
        # train
        for epoch in progress:
            self.model.train()
            running_train_loss = 0.0

            # mini-batch with training DataLoader
            for X_batch, Y_batch in train_loader:
                # move data to the selected device (CPU or GPU)
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                # zero-out the grads from previous step
                self.optimizer.zero_grad()

                # forward pass
                predictions = self.model(X_batch)
                loss = self.model.loss_fn(predictions, Y_batch)

                # backprop
                loss.backward()

                # update model weights
                self.optimizer.step()

                # accumulate the training loss
                running_train_loss += loss.item() * X_batch.size(0)

            # average training loss for this epoch
            avg_train_loss = running_train_loss / len(train_dataset)

            # evaluate the model on the validation set if provided
            avg_val_loss = None
            if val_loader is not None:
                avg_val_loss = self._evaluate(val_loader)

            # step the scheduler if one is set
            if self.scheduler is not None:
                # special handling if ReduceLROnPlateau is used
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(
                        avg_val_loss if avg_val_loss is not None else avg_train_loss
                    )
                else:
                    self.scheduler.step()

            # postfix on the progress bar to show loss
            if val_loader is not None:
                progress.set_postfix(
                    {"TL": f"{avg_train_loss:.1f}", "VL": f"{avg_val_loss:.1f}"}
                )
            else:
                progress.set_postfix({"train_loss": f"{avg_train_loss:.2f}"})

            # when early stopping is enabled and there is validation data
            if self.patience is not None and val_loader is not None:
                # did validation loss improve?
                if avg_val_loss < self.best_val_loss:
                    # if improved, update best validation loss and reset patience counter
                    self.best_val_loss = avg_val_loss
                    self.patience_counter = 0
                    # save the best model state
                    torch.save(self.model.state_dict(), "best_dynatorch_model.pth")
                else:
                    # no improvement -> increment patience counter
                    self.patience_counter += 1
                    # when patience is exceeded, trigger early stop
                    if self.patience_counter >= self.patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break

    def predict(self, X: torch.Tensor, batch_size: int = 32):
        """
        Predict using the trained model.

        Returns:
            np.ndarray: Predictions as a NumPy array.
        """
        self.model.eval()
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_preds = []

        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                preds = self.model(X_batch)
                all_preds.append(preds.cpu())

        # concatenate all predictions and return as a NumPy array
        all_preds = torch.cat(all_preds, dim=0)
        return all_preds.numpy()

    def _evaluate(self, loader: DataLoader) -> float:
        """
        Evaluate the model on validation DataLoader to compute the average loss.

        Returns:
            average loss over the entire dataset provided by `loader`.
        """
        self.model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for X_val_batch, Y_val_batch in loader:
                X_val_batch = X_val_batch.to(self.device)
                Y_val_batch = Y_val_batch.to(self.device)

                # forward pass and validation loss
                val_preds = self.model(X_val_batch)
                val_loss = self.model.loss_fn(val_preds, Y_val_batch)

                # accumulate validation loss, scaled by batch size
                running_val_loss += val_loss.item() * X_val_batch.size(0)

        # average validation loss
        avg_val_loss = running_val_loss / len(loader.dataset)

        return avg_val_loss
