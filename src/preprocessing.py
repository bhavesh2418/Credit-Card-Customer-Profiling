"""
preprocessing.py — Clean, impute, engineer features, and scale for clustering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from src.config import (
    ID_COLUMN, MODELS_DIR,
    CLEAN_DATA_FILE, UNSCALED_DATA_FILE
)


def preprocess(df: pd.DataFrame, save: bool = True):
    """Full pipeline: drop ID → impute → engineer → scale.
    Returns: (df_scaled, df_unscaled)
    """
    df = df.copy()

    # Drop ID
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    # Impute missing with median
    n_missing = df.isnull().sum().sum()
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    print(f"Imputed   : {n_missing} missing values (median fill)")

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped   : {before - len(df)} duplicate rows")

    # ── Feature Engineering ──────────────────────────────────────────────────
    df["PURCHASES_TO_LIMIT_RATIO"]       = (df["PURCHASES"] / (df["CREDIT_LIMIT"] + 1)).round(4)
    df["CASH_ADVANCE_RATIO"]             = (df["CASH_ADVANCE"] / (df["BALANCE"] + 1)).round(4)
    df["PAYMENT_TO_MINIMUM_RATIO"]       = (df["PAYMENTS"] / (df["MINIMUM_PAYMENTS"] + 1)).round(4)
    df["MONTHLY_AVG_PURCHASE"]           = (df["PURCHASES"] / df["TENURE"]).round(2)
    df["INSTALLMENT_TO_PURCHASE_RATIO"]  = (df["INSTALLMENTS_PURCHASES"] / (df["PURCHASES"] + 1)).round(4)
    df["CASH_ADVANCE_TO_CREDIT_RATIO"]   = (df["CASH_ADVANCE"] / (df["CREDIT_LIMIT"] + 1)).round(4)
    df["BALANCE_TO_CREDIT_RATIO"]        = (df["BALANCE"] / (df["CREDIT_LIMIT"] + 1)).round(4)
    print(f"Engineered: 7 new ratio features")
    print(f"Shape     : {df.shape}")

    df_unscaled = df.copy()

    # Scale
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    if save:
        df_scaled.to_csv(CLEAN_DATA_FILE, index=False)
        df_unscaled.to_csv(UNSCALED_DATA_FILE, index=False)
        print(f"Saved     : data/processed/cc_clean_scaled.csv")
        print(f"Saved     : data/processed/cc_clean_unscaled.csv")

    return df_scaled, df_unscaled
