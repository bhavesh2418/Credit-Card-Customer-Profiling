"""
feature_selection.py — RFE and LASSO feature selection before clustering.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LassoCV
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor
from src.config import IMAGES_DIR, PALETTE, FIG_DPI


def lasso_feature_importance(df_scaled: pd.DataFrame, target_col: str = "BALANCE") -> pd.Series:
    """
    Use LassoCV to identify feature importances.
    Since clustering is unsupervised, we use BALANCE as a proxy target
    (highest-variance feature) to rank feature relevance.
    """
    X = df_scaled.drop(columns=[target_col], errors="ignore")
    y = df_scaled[target_col] if target_col in df_scaled.columns else df_scaled.iloc[:, 0]

    lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
    lasso.fit(X, y)

    importance = pd.Series(np.abs(lasso.coef_), index=X.columns).sort_values(ascending=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [PALETTE[0] if v > importance.mean() else "#BBDEFB" for v in importance.values]
    bars = ax.barh(importance.index[::-1], importance.values[::-1],
                   color=colors[::-1], edgecolor="white", alpha=0.9)
    ax.axvline(importance.mean(), color="#F44336", ls="--", lw=1.5, label=f"Mean: {importance.mean():.3f}")
    ax.set_title("LASSO Feature Importance\n(proxy target: BALANCE)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Absolute Coefficient")
    ax.legend()
    plt.tight_layout()
    _save("12_lasso_feature_importance.png")

    print(f"LASSO alpha: {lasso.alpha_:.5f}")
    print(f"Features with non-zero coeff: {(np.abs(lasso.coef_) > 0).sum()}/{len(lasso.coef_)}")
    return importance


def rfe_feature_ranking(df_scaled: pd.DataFrame, n_features: int = 15,
                        target_col: str = "BALANCE") -> pd.DataFrame:
    """
    Recursive Feature Elimination using GradientBoostingRegressor.
    Returns a DataFrame of features ranked by RFE support + ranking.
    """
    X = df_scaled.drop(columns=[target_col], errors="ignore")
    y = df_scaled[target_col] if target_col in df_scaled.columns else df_scaled.iloc[:, 0]

    estimator = GradientBoostingRegressor(n_estimators=50, random_state=42)
    rfe = RFE(estimator, n_features_to_select=n_features, step=1)
    rfe.fit(X, y)

    rfe_df = pd.DataFrame({
        "Feature":  X.columns,
        "Selected": rfe.support_,
        "Ranking":  rfe.ranking_
    }).sort_values("Ranking")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [PALETTE[2] if s else "#FFCDD2" for s in rfe_df["Selected"]]
    ax.barh(rfe_df["Feature"][::-1], rfe_df["Ranking"][::-1],
            color=colors[::-1], edgecolor="white", alpha=0.9)
    ax.axvline(1, color="#F44336", ls="--", lw=1.5, label="Selected (rank=1)")
    ax.set_title(f"RFE Feature Ranking\n(Top {n_features} selected in green)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("RFE Rank (lower = more important)")
    ax.legend()
    plt.tight_layout()
    _save("13_rfe_feature_ranking.png")

    selected = rfe_df[rfe_df["Selected"]]["Feature"].tolist()
    print(f"RFE selected {len(selected)} features: {selected}")
    return rfe_df, selected


def get_selected_features(df_scaled: pd.DataFrame, top_n_lasso: int = 15) -> list:
    """
    Combine LASSO + RFE to get a consensus feature set for clustering.
    Returns union of top LASSO features and RFE-selected features.
    """
    lasso_imp   = lasso_feature_importance(df_scaled)
    rfe_df, rfe_selected = rfe_feature_ranking(df_scaled)

    top_lasso = lasso_imp.head(top_n_lasso).index.tolist()
    consensus = list(set(top_lasso) | set(rfe_selected))

    print(f"\nConsensus feature set ({len(consensus)} features):")
    for f in sorted(consensus):
        print(f"  {f}")
    return consensus


def _save(filename: str):
    path = IMAGES_DIR / filename
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: images/{filename}")
