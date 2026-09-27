"""
src/data_preprocessing.py
Data loading and preprocessing for Online Shoppers Intention dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def explore_data(df: pd.DataFrame) -> dict:
    """Generate a summary of the dataset for exploration."""
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "class_distribution": df["Revenue"].value_counts().to_dict(),
        "numeric_stats": df.describe().to_dict(),
    }
    return summary


def preprocess_data(df: pd.DataFrame, scale: bool = True) -> tuple:
    """
    Preprocess the dataset:
    1. Handle missing values
    2. Encode categorical variables (Month, VisitorType)
    3. Convert boolean columns to int
    4. Optionally scale numerical features

    Returns:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        encoders (dict): Fitted encoders for reproducibility
    """
    df = df.copy()

    # --- Handle Missing Values ---
    df.dropna(inplace=True)
    print(f"After dropping NAs: {df.shape[0]} rows")

    # --- Encode Categorical Variables ---
    encoders = {}

    # Month encoding (ordinal makes sense here)
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "June",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_map = {m: i + 1 for i, m in enumerate(month_order)}
    df["Month"] = df["Month"].map(month_map)
    encoders["Month"] = month_map

    # VisitorType encoding
    le_visitor = LabelEncoder()
    df["VisitorType"] = le_visitor.fit_transform(df["VisitorType"])
    encoders["VisitorType"] = le_visitor

    # Boolean to int
    df["Weekend"] = df["Weekend"].astype(int)
    df["Revenue"] = df["Revenue"].astype(int)

    # --- Separate Features & Target ---
    X = df.drop("Revenue", axis=1)
    y = df["Revenue"]

    # --- Scale ---
    scaler = None
    if scale:
        scaler = StandardScaler()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        encoders["scaler"] = scaler

    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    return X, y, encoders


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def create_modified_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a modified version of the dataset for DVC versioning (Version 2).
    Adds engineered features and removes low-variance columns.
    """
    df = df.copy()

    # Feature engineering
    df["TotalPages"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    df["TotalDuration"] = (df["Administrative_Duration"] +
                           df["Informational_Duration"] +
                           df["ProductRelated_Duration"])
    df["BounceExitRatio"] = df["BounceRates"] / (df["ExitRates"] + 1e-6)

    print(f"Modified dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"New features added: TotalPages, TotalDuration, BounceExitRatio")
    return df
