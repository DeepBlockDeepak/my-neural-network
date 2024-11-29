import pandas as pd
from sklearn.impute import SimpleImputer


def extract_title(data):
    """
    Extracts the title from the 'Name' column and creates a new 'Title' column.

    'the regex captures any word immediately followed by
    a period character -> the title in the names.
    """

    data["Title"] = data["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)

    return data


def normalize_titles(data):
    """
    Binning common titles and grouping the rare titles under their own category.
    """
    title_mapping = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Rare",
        "Countess": "Rare",
        "Capt": "Rare",
        "Col": "Rare",
        "Don": "Rare",
        "Dr": "Rare",
        "Major": "Rare",
        "Rev": "Rare",
        "Sir": "Rare",
        "Jonkheer": "Rare",
        "Dona": "Rare",
    }
    data["Title"] = data["Title"].replace(title_mapping)
    return data


def create_family_size(data):
    """
    New feature adds 'SibSp' and 'Parch' columns plus one for the passenger themself.
    """

    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

    return data


def create_is_alone(data):
    """
    new feature is 1 if FamilySize is 1, else 0.
    """

    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

    return data


def create_cabin_indicator(data):
    """
    New binary 'HasCabin' feature. 1 if 'Cabin' data is available (not NaN), else 0.
    """
    data["HasCabin"] = data["Cabin"].apply(lambda x: 0 if pd.isna(x) else 1)

    return data


def impute_age(data):
    """
    Avoiding many missing values in the 'AgeBin', in create_age_bins().
    """
    imputer = SimpleImputer(strategy="median")
    data["Age"] = imputer.fit_transform(data[["Age"]])
    return data


def create_age_bins(data):
    """
    Categorizes 'Age' into bins and creates a new 'AgeBin' column with the category labels.
    """

    bins = [0, 12, 20, 40, 60, 80]
    labels = ["Child", "Teen", "Adult", "MiddleAged", "Senior"]
    data["AgeBin"] = pd.cut(data["Age"], bins, labels=labels)

    return data


def create_fare_bins(data):
    """
    Categorizes 'Fare' into bins and creates a new 'FareBin' column with the category labels.
    """
    max_fare = data["Fare"].max()
    bins = [0, 7.91, 14.454, 31, max_fare + 1]
    labels = ["Low", "Below_Average", "Above_Average", "High"]
    data["FareBin"] = pd.cut(data["Fare"], bins, labels=labels, include_lowest=True)

    return data


def apply_feature_engineering(data):
    """
    Applies all the feature engineering functions defined above to the data.

    Args:
        data (DataFrame): The original DataFrame.

    Returns:
        DataFrame: The DataFrame with all the new features added.
    """

    data = extract_title(data)
    data = normalize_titles(data)
    data = create_family_size(data)
    data = create_is_alone(data)
    data = create_cabin_indicator(data)
    data = impute_age(data)
    data = create_age_bins(data)
    data = create_fare_bins(data)

    return data
