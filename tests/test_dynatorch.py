import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from my_neural_network.constants import TaskType
from pytorch_core.dynatorch_model.torch_model import DynaTorchModel
from pytorch_core.torch_trainer.trainer import Trainer


def test_streeteasy_regression_torch():
    # load dataset
    apartments_df = pd.read_csv("tests/street_easy_data/streeteasy.csv")

    numerical_features = [
        "bedrooms",
        "bathrooms",
        "size_sqft",
        "min_to_subway",
        "floor",
        "building_age_yrs",
        "no_fee",
        "has_roofdeck",
        "has_washer_dryer",
        "has_doorman",
        "has_elevator",
        "has_dishwasher",
        "has_patio",
        "has_gym",
    ]
    X = apartments_df[numerical_features].to_numpy()  # shape: (m, n_features)
    y = apartments_df["rent"].to_numpy()  # shape: (m,)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # fit on training data
    X_test = scaler.transform(X_test)  # transform test data

    # convert to torch tensors
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)  # (m_train, n_features)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32)  # (m_train,)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)  # (m_test, n_features)
    y_test_torch = torch.tensor(y_test, dtype=torch.float32)  # (m_test,)

    # For regression, MSELoss works fine with (m,) for y_train and predictions.
    # If model outputs (m,1), just unsqueeze y:
    y_train_torch = y_train_torch.unsqueeze(1)  # (m_train,1)
    y_test_torch = y_test_torch.unsqueeze(1)  # (m_test,1)
    # Ensure consistent shapes between predictions and labels.

    # define model layer dims and create model
    layer_dims = [X_train.shape[1], 128, 64, 1]
    model = DynaTorchModel(layer_dims, task_type=TaskType.REGRESSION.value)
    # Note: REGRESSION = "regression"

    # create a trainer
    trainer = Trainer(
        model=model,
        optimizer=None,  # will create an Adam optimizer inside fit if None
        device=None,  # automatically chooses CPU or GPU
        patience=10,  # early stopping patience
    )

    # Fit the model
    trainer.fit(
        X_train=X_train_torch,
        Y_train=y_train_torch,
        epochs=2000,
        batch_size=32,
        learning_rate=0.001,
        X_val=X_test_torch,
        Y_val=y_test_torch,
    )

    # Predict on test set
    predictions = trainer.predict(X_test_torch, batch_size=32)

    # evaluate
    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)

    # results
    print(f"DynaTorchModel Test RMSE: {rmse:.4f}")

    # Plotting predictions vs actual
    y_test_flat = y_test.flatten()
    predictions_flat = predictions.flatten()

    plt.figure(figsize=(10, 6))
    plt.scatter(
        y_test_flat, predictions_flat, alpha=0.5, color="blue", label="Predictions"
    )

    # plotting line y = x
    max_value = max(y_test_flat.max(), predictions_flat.max())
    min_value = min(y_test_flat.min(), predictions_flat.min())
    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        color="pink",
        label="y = x",
    )

    plt.xlabel("Actual Rent")
    plt.ylabel("Predicted Rent")
    plt.title(
        f"Predicted vs Actual Rent Values (DynaTorchModel with RMSE = {rmse:.2f})"
    )
    plt.legend()

    plot_filename = "tests/street_easy_data/streeteasy_predictions_DynaTorchModel.png"
    plt.savefig(plot_filename)
    plt.close()
