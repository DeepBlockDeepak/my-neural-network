import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from my_neural_network.constants import TaskType
from pytorch_core.dynatorch_model.torch_model import DynaTorchModel
from pytorch_core.torch_trainer.trainer import Trainer
from tests.titanic_data.preprocess import preprocess_data


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

    # for regression, MSELoss works fine with (m,) for y_train and predictions.
    # If model outputs (m,1), just unsqueeze y:
    y_train_torch = y_train_torch.unsqueeze(1)  # (m_train,1)
    y_test_torch = y_test_torch.unsqueeze(1)  # (m_test,1)

    # define model and trainer
    layer_dims = [X_train.shape[1], 128, 64, 1]
    model = DynaTorchModel(layer_dims, task_type=TaskType.REGRESSION.value)
    trainer = Trainer(
        model=model,
        optimizer=None,  # will create an Adam optimizer inside fit if None
        device=None,  # automatically chooses CPU or GPU
        patience=10,  # early stopping patience
        scheduler_type="ReduceLROnPlateau",
        learning_rate=0.001,
    )

    trainer.fit(
        X_train=X_train_torch,
        Y_train=y_train_torch,
        epochs=20000,
        batch_size=512,  # try len(X_train_torch)
        X_val=X_test_torch,
        Y_val=y_test_torch,
    )

    # predict
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


def test_breast_cancer_classification_torch():
    # load data
    data = load_breast_cancer()
    X = data.data  # shape: (m, n_features), for breast cancer ~ (569, 30)
    Y = data.target  # shape: (m,), values are 0 or 1

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    # Shapes now:
    # X_train: (m_train, 30), X_test: (m_test, 30)
    # Y_train: (m_train,), Y_test: (m_test,)

    # scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # still (m_train, 30)
    X_test = scaler.transform(X_test)  # (m_test, 30)

    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    Y_train_torch = torch.tensor(Y_train, dtype=torch.float32)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    Y_test_torch = torch.tensor(Y_test, dtype=torch.float32)

    # unsqueeze Y to match (m,1) if needed since model outputs (m,1)
    Y_train_torch = Y_train_torch.unsqueeze(1)  # (m_train,1)
    Y_test_torch = Y_test_torch.unsqueeze(1)  # (m_test,1)

    # input dimension = 30 (features), output = 1 (binary classification)
    layer_dims = [X_train.shape[1], 64, 32, 1]

    model = DynaTorchModel(layer_dims, task_type=TaskType.CLASSIFICATION.value)
    trainer = Trainer(
        model=model,
        optimizer=None,  # will use default Adam inside fit if None
        device=None,  # auto choose device
        patience=10,  # early stopping patience
        scheduler_type="ReduceLROnPlateau",
    )

    # train
    trainer.fit(
        X_train=X_train_torch,
        Y_train=Y_train_torch,
        epochs=2000,
        batch_size=10,
        X_val=X_test_torch,
        Y_val=Y_test_torch,
    )

    # predict and convert to binary predictions using a threshold of 0.5
    predictions = trainer.predict(X_test_torch, batch_size=10)
    pred_classes = (predictions > 0.5).astype(int).flatten()

    accuracy = accuracy_score(Y_test, pred_classes)
    print(f"DynaTorchModel Accuracy: {accuracy:.4f}")


def test_titanic_classification_torch():
    # load
    train_data = pd.read_csv("tests/titanic_data/train.csv")

    # process data -> obtain pd.DataFrames of feature engineered train/val
    X_train_df = preprocess_data(train_data)
    y_train_df = train_data["Survived"]  # extract labels

    X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(
        X_train_df, y_train_df, test_size=0.2, random_state=0
    )

    # convert to numpy
    X_train = X_train_df.to_numpy()  # (m_train, n_features)
    X_val = X_val_df.to_numpy()  # (m_val, n_features)
    y_train = y_train_df.to_numpy()  # (m_train,)
    y_val = y_val_df.to_numpy()  # (m_val,)

    # torch tensorize
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)  # (m_train, n_features)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(
        1
    )  # (m_train,1)
    X_val_torch = torch.tensor(X_val, dtype=torch.float32)  # (m_val, n_features)
    y_val_torch = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)  # (m_val,1)

    # task_type='classification' ensures a single-sigmoid output & BCELoss
    layer_dims = [X_train.shape[1], 64, 32, 1]
    model = DynaTorchModel(
        layer_dims=layer_dims, task_type=TaskType.CLASSIFICATION.value
    )
    trainer = Trainer(
        model=model,
        optimizer=None,  # will use default Adam if None
        device=None,  # auto choose
        patience=10,  # early stopping patience
        scheduler_type="ReduceLROnPlateau",
    )

    trainer.fit(
        X_train=X_train_torch,
        Y_train=y_train_torch,
        epochs=1000,
        batch_size=32,
        X_val=X_val_torch,
        Y_val=y_val_torch,
    )

    # predict
    predictions = trainer.predict(X_val_torch, batch_size=32)
    # predictions are probabilities, convert to 0 or 1
    pred_classes = (predictions > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_val_torch, pred_classes)
    print(f"DynaTorchModel Accuracy: {accuracy:.4f}")
