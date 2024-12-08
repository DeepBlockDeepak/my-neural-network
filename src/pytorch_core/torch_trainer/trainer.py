from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        device: str = None,  # 'cuda' if available, otherwise 'cpu'
        patience: Optional[int] = 10,  # set to None to disable
    ):
        # automatically use GPU if available
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = None
        self.patience = patience  # early stopping params
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def fit(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        X_val: torch.Tensor = None,
        Y_val: torch.Tensor = None,
    ):
        """
        Train the model on the provided training data,
        optionally validating on a validation set.
        """
        # set up the optimizer if not provided
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # DataLoader for training data - helps minibatching loading
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # DataLoader for validation data if provided
        val_loader = None
        if X_val is not None and Y_val is not None:
            val_dataset = TensorDataset(X_val, Y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # train
        for epoch in range(epochs):
            self.model.train()
            running_train_loss = 0.0

            # progress bar for this epoch
            data_iter = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False
            )

            # mini-batch with training DataLoader
            for X_batch, Y_batch in data_iter:
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
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(
                        avg_val_loss if avg_val_loss is not None else avg_train_loss
                    )
                else:
                    self.scheduler.step()

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
                        print("Early stopping triggered.")
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
