"""
visualize.py — All plot functions. Saves to images/ (committed to GitHub).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from src.config import IMAGES_DIR, PALETTE, FIG_DPI

sns.set_theme(style="whitegrid", font_scale=1.05)


# ── EDA Plots ──────────────────────────────────────────────────────────────────

def plot_missing_values(df: pd.DataFrame):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    pct = (missing / len(df) * 100).round(2)

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(missing.index, pct.values, color="#F44336", edgecolor="white", alpha=0.85)
    for bar, v in zip(bars, pct.values):
        ax.text(v + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{v}%", va="center", fontsize=10, fontweight="bold")
    ax.set_title("Missing Values by Feature", fontsize=13, fontweight="bold")
    ax.set_xlabel("% Missing")
    ax.set_xlim(0, pct.max() * 1.3)
    plt.tight_layout()
    _save("01_missing_values.png")


def plot_distributions(df: pd.DataFrame):
    cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT",
            "PAYMENTS", "MINIMUM_PAYMENTS"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    for ax, col, color in zip(axes, cols, PALETTE):
        ax.hist(df[col].dropna(), bins=40, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(df[col].median(), color="black", lw=1.5, ls="--",
                   label=f"Median: ${df[col].median():,.0f}")
        ax.set_title(col.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("Value ($)")
        ax.legend(fontsize=8)
    plt.suptitle("Monetary Feature Distributions", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save("02_distributions.png")


def plot_frequency_features(df: pd.DataFrame):
    cols = ["BALANCE_FREQUENCY", "PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY",
            "PURCHASES_INSTALLMENTS_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "PRC_FULL_PAYMENT"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        ax.hist(df[col].dropna(), bins=20, color="#2196F3", edgecolor="white", alpha=0.85)
        ax.set_title(col.replace("_", " ").title(), fontweight="bold", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Frequency Score (0–1)")
    plt.suptitle("Frequency Feature Distributions", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save("03_frequency_features.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    df_num = df.drop(columns=["CUST_ID"], errors="ignore")
    corr = df_num.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.4, square=True, ax=ax,
                cbar_kws={"shrink": 0.75}, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("04_correlation_heatmap.png")


# ── Feature Engineering Plots ──────────────────────────────────────────────────

def plot_engineered_features(df: pd.DataFrame):
    eng_cols = [
        "PURCHASES_TO_LIMIT_RATIO", "CASH_ADVANCE_RATIO",
        "PAYMENT_TO_MINIMUM_RATIO", "MONTHLY_AVG_PURCHASE",
        "INSTALLMENT_TO_PURCHASE_RATIO", "CASH_ADVANCE_TO_CREDIT_RATIO",
        "BALANCE_TO_CREDIT_RATIO",
    ]
    eng_cols = [c for c in eng_cols if c in df.columns]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, (ax, col) in enumerate(zip(axes, eng_cols)):
        ax.hist(df[col], bins=35, color=PALETTE[i % len(PALETTE)],
                edgecolor="white", alpha=0.85)
        ax.set_title(col.replace("_", " ").title(), fontsize=8, fontweight="bold")
        ax.set_xlabel("Value")
    for ax in axes[len(eng_cols):]:
        ax.set_visible(False)
    plt.suptitle("Engineered Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save("05_engineered_features.png")


def plot_outliers(df: pd.DataFrame):
    cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, col, color in zip(axes, cols, PALETTE):
        bp = ax.boxplot(df[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.6),
                        medianprops=dict(color="black", linewidth=2),
                        flierprops=dict(marker=".", markersize=2, alpha=0.3))
        ax.set_title(col.replace("_", " ").title(), fontsize=9, fontweight="bold")
        ax.set_ylabel("Value ($)")
    plt.suptitle("Outlier Analysis — Monetary Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("06_outliers.png")


# ── Clustering Plots ───────────────────────────────────────────────────────────

def plot_optimal_k(k_results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    from src.config import N_CLUSTERS

    axes[0].plot(k_results["k_range"], k_results["inertias"],
                 marker="o", color="#2196F3", lw=2.5, markersize=7)
    axes[0].axvline(N_CLUSTERS, color="#F44336", ls="--", lw=2, label=f"Chosen K={N_CLUSTERS}")
    axes[0].set_title("Elbow Method", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia (WCSS)")
    axes[0].legend()
    axes[0].fill_between(k_results["k_range"], k_results["inertias"], alpha=0.07, color="#2196F3")

    axes[1].plot(k_results["k_range"], k_results["silhouettes"],
                 marker="s", color="#4CAF50", lw=2.5, markersize=7)
    axes[1].axvline(N_CLUSTERS, color="#F44336", ls="--", lw=2, label=f"Chosen K={N_CLUSTERS}")
    axes[1].set_title("Silhouette Score", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()
    axes[1].fill_between(k_results["k_range"], k_results["silhouettes"], alpha=0.07, color="#4CAF50")

    plt.suptitle(f"Optimal K Selection — Chosen K = {N_CLUSTERS}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("07_optimal_k.png")


def plot_clusters_pca(df_pca: pd.DataFrame, labels: np.ndarray, explained: float,
                      title: str = "KMeans", fname: str = "08_clusters_pca_kmeans.png"):
    fig, ax = plt.subplots(figsize=(9, 6))
    for cluster in np.unique(labels):
        mask = labels == cluster
        ax.scatter(df_pca.loc[mask, "PC1"], df_pca.loc[mask, "PC2"],
                   label=f"Cluster {cluster}", alpha=0.45, s=14,
                   color=PALETTE[cluster % len(PALETTE)])
    ax.set_title(f"{title} — PCA 2D View\n({explained:.1f}% variance explained)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(markerscale=3, title="Segment")
    plt.tight_layout()
    _save(fname)


def plot_cluster_profiles(df_unscaled: pd.DataFrame, labels: np.ndarray):
    df = df_unscaled.copy()
    df["Cluster"] = labels
    key = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT",
           "PAYMENTS", "PRC_FULL_PAYMENT", "PURCHASES_TO_LIMIT_RATIO",
           "MONTHLY_AVG_PURCHASE"]
    key = [c for c in key if c in df.columns]
    profile = df.groupby("Cluster")[key].mean()
    norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    sns.heatmap(norm.T, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=axes[0], cbar_kws={"shrink": 0.8})
    axes[0].set_title("Cluster Feature Profiles (Normalised)", fontweight="bold")
    axes[0].set_xlabel("Cluster")

    profile[["BALANCE", "PURCHASES", "CASH_ADVANCE", "PAYMENTS"]].plot(
        kind="bar", ax=axes[1], color=PALETTE[:4], edgecolor="white", alpha=0.88
    )
    axes[1].set_title("Key Metrics by Segment (Mean $)", fontweight="bold")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Mean Value ($)")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(loc="upper right", fontsize=9)

    plt.suptitle("Customer Segment Profiles", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("09_cluster_profiles.png")


def plot_cluster_distribution(labels: np.ndarray):
    unique, counts = np.unique(labels, return_counts=True)
    pcts = counts / counts.sum() * 100
    segment_labels = [f"Cluster {k}\n({p:.1f}%)" for k, p in zip(unique, pcts)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bars = axes[0].bar(segment_labels, counts, color=PALETTE[:len(unique)],
                       edgecolor="white", alpha=0.88)
    for bar, n in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 30, f"{n:,}",
                     ha="center", fontsize=10, fontweight="bold")
    axes[0].set_title("Customer Count per Segment", fontweight="bold")
    axes[0].set_ylabel("Number of Customers")

    axes[1].pie(counts, labels=segment_labels, colors=PALETTE[:len(unique)],
                autopct="%1.1f%%", startangle=140,
                wedgeprops=dict(edgecolor="white", linewidth=1.5))
    axes[1].set_title("Segment Distribution (%)", fontweight="bold")

    plt.suptitle("Customer Distribution Across Segments", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("10_cluster_distribution.png")


def plot_pca_variance(explained_ratio: np.ndarray):
    cumulative = np.cumsum(explained_ratio) * 100
    n = len(explained_ratio)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, n + 1), explained_ratio * 100, color="#2196F3",
           edgecolor="white", alpha=0.85, label="Individual")
    ax.plot(range(1, n + 1), cumulative, marker="o", color="#F44336",
            lw=2, label="Cumulative")
    ax.axhline(80, color="gray", ls="--", lw=1, alpha=0.6, label="80% threshold")
    ax.set_title("PCA — Explained Variance by Component", fontweight="bold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_xticks(range(1, n + 1))
    ax.legend()
    plt.tight_layout()
    _save("11_pca_variance.png")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save(filename: str):
    path = IMAGES_DIR / filename
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: images/{filename}")
