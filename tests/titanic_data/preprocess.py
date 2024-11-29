import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from tests.titanic_data.features import apply_feature_engineering


def preprocess_data(train_data: pd.DataFrame):
    """
    Prepare the dataset for training by selecting relevant features,
    handling missing values, converting categorical data to numerical.

    Args:
        train_data: The training dataset.

    Returns:
        DataFrame: The processed training dataset.
    """

    # apply feature engineering
    train_data = apply_feature_engineering(train_data)

    # handling missing values in 'Embarked'
    train_data["Embarked"] = train_data["Embarked"].fillna(
        train_data["Embarked"].mode()[0]
    )

    # hand-define numeric and categorical features
    numeric_features = [
        "Pclass",
        "SibSp",
        "Parch",
        "Fare",
        "FamilySize",
        "IsAlone",  # consider dropping bool-int col from scaling
        "HasCabin",  # consider dropping bool-int col from scaling
    ]
    categorical_features = ["Sex", "Embarked", "Title", "AgeBin", "FareBin"]

    # impute missing values in numeric data
    imputer = SimpleImputer(strategy="median")
    train_numeric_data = pd.DataFrame(
        imputer.fit_transform(train_data[numeric_features]), columns=numeric_features
    )

    # # convert boolean features to integers
    # boolean_features = ["IsAlone", "HasCabin"]
    # train_numeric_data[boolean_features] = train_numeric_data[boolean_features].astype(int)

    # scale numeric data
    scaler = StandardScaler()
    train_numeric_data = pd.DataFrame(
        scaler.fit_transform(train_numeric_data), columns=numeric_features
    )

    # one-hot encode the categorical data
    categorical_data = pd.get_dummies(train_data[categorical_features])

    # convert boolean columns to integers before splitting
    bool_cols = categorical_data.select_dtypes(include="bool").columns
    categorical_data[bool_cols] = categorical_data[bool_cols].astype(int)

    # combine the numeric and categorical data
    X_train = pd.concat([train_numeric_data, categorical_data], axis=1)

    return X_train
