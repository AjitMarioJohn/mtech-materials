from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class AdultDataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    dataset_name: str
    n_samples: int
    n_features: int


DEFAULT_ADULT_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"


def load_adult_income_raw_dataframe(url: str = DEFAULT_ADULT_DATA_URL) -> pd.DataFrame:
    """Load raw Adult Income data from UCI URL into a pandas DataFrame."""
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    return pd.read_csv(
        StringIO(response.text),
        names=column_names,
        skipinitialspace=True,
        na_values="?",
    )


def preview_adult_preprocessing(
    n_rows: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    url: str = DEFAULT_ADULT_DATA_URL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return sample rows before and after preprocessing for quick inspection."""
    raw_df = load_adult_income_raw_dataframe(url=url)
    before_df = raw_df.head(n_rows).copy()

    data = load_adult_income_data(
        test_size=test_size,
        random_state=random_state,
        url=url,
    )
    after_df = pd.DataFrame(data.X_train[:n_rows], columns=data.feature_names)
    return before_df, after_df


def load_adult_income_data(
    test_size: float = 0.2,
    random_state: int = 42,
    url: str = DEFAULT_ADULT_DATA_URL,
) -> AdultDataset:
    """Load and preprocess the Adult Income dataset from UCI ML Repository.

    Fetches data directly from URL and applies:
    - one-hot encoding for categorical columns
    - standard scaling for numeric columns
    - stratified train-test split (configurable)
    
    Args:
        test_size: Fraction of data to use for testing (0.2 = 80/20 split)
        random_state: Random seed for reproducibility
        url: Full URL to adult.data file
        
    Returns:
        AdultDataset with preprocessed X_train, X_test, y_train, y_test
    """
    df = load_adult_income_raw_dataframe(url=url)

    # Extract target variable
    y = (df["income"].str.strip() == ">50K").astype(np.int64).to_numpy()

    # Remove target column from features
    X_df = df.drop(columns=["income"])

    # Identify numeric and categorical columns
    numeric_columns = X_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [c for c in X_df.columns if c not in numeric_columns]

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_columns),
            (
                "cat",
                Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                categorical_columns,
            ),
        ]
    )

    # Train-test split with stratification
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Fit preprocessor on training data and transform both sets
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out().tolist()

    return AdultDataset(
        X_train=X_train.astype(np.float64),
        X_test=X_test.astype(np.float64),
        y_train=y_train.astype(np.int64),
        y_test=y_test.astype(np.int64),
        feature_names=feature_names,
        dataset_name="Adult Income (UCI ML Repository)",
        n_samples=int(X_df.shape[0]),
        n_features=int(X_df.shape[1]),
    )

