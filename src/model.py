"""
model.py — KMeans + Hierarchical clustering, PCA, optimal K selection.
Includes: PCA-before-clustering comparison, ARI, NMI metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                              normalized_mutual_info_score)
from sklearn.decomposition import PCA
from src.config import RANDOM_STATE, K_RANGE, N_CLUSTERS, MODELS_DIR, IMAGES_DIR, FIG_DPI


def find_optimal_k(df_scaled: pd.DataFrame) -> dict:
    """Compute inertia + silhouette for K in K_RANGE."""
    inertias, silhouettes = [], []
    print(f"{'K':>3} | {'Inertia':>12} | {'Silhouette':>10}")
    print("-" * 32)
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(df_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(df_scaled, labels)
        silhouettes.append(sil)
        print(f"{k:>3} | {km.inertia_:>12,.0f} | {sil:>10.4f}")
    return {"k_range": list(K_RANGE), "inertias": inertias, "silhouettes": silhouettes}


def train_kmeans(df_scaled: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> np.ndarray:
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(df_scaled)
    sil = silhouette_score(df_scaled, labels)
    joblib.dump(km, MODELS_DIR / "kmeans.pkl")
    print(f"KMeans K={n_clusters} | Silhouette={sil:.4f} | Inertia={km.inertia_:,.0f}")
    sizes = dict(zip(*np.unique(labels, return_counts=True)))
    for k, n in sizes.items():
        print(f"  Cluster {k}: {n:,} customers ({n/len(labels)*100:.1f}%)")
    return labels


def train_hierarchical(df_scaled: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> np.ndarray:
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = hc.fit_predict(df_scaled)
    sil = silhouette_score(df_scaled, labels)
    print(f"Hierarchical K={n_clusters} | Silhouette={sil:.4f}")
    return labels


def pca_transform(df_scaled: pd.DataFrame, n_components: int = 2):
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    comps = pca.fit_transform(df_scaled)
    joblib.dump(pca, MODELS_DIR / "pca.pkl")
    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"PCA {n_components}D: {explained:.1f}% variance explained")
    df_pca = pd.DataFrame(comps, columns=[f"PC{i+1}" for i in range(n_components)])
    return df_pca, pca.explained_variance_ratio_, explained


def compare_pca_clustering(df_scaled: pd.DataFrame,
                            n_clusters: int = N_CLUSTERS) -> pd.DataFrame:
    """
    Compare KMeans on:
      1. Raw scaled data
      2. PCA (n_components retaining >90% variance)
      3. PCA (3 components)
    Reports Silhouette, Inertia, ARI, NMI vs raw labels as reference.
    """
    # ── Baseline: raw scaled ──────────────────────────────────────────────────
    km_raw    = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    raw_labels = km_raw.fit_predict(df_scaled)
    raw_sil   = silhouette_score(df_scaled, raw_labels)

    # ── PCA retaining >90% variance ───────────────────────────────────────────
    pca_full  = PCA(random_state=RANDOM_STATE)
    pca_full.fit(df_scaled)
    cumvar    = np.cumsum(pca_full.explained_variance_ratio_)
    n_90      = int(np.searchsorted(cumvar, 0.90)) + 1
    pca_n90   = PCA(n_components=n_90, random_state=RANDOM_STATE)
    data_n90  = pca_n90.fit_transform(df_scaled)
    km_n90    = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels_n90 = km_n90.fit_predict(data_n90)
    sil_n90   = silhouette_score(data_n90, labels_n90)

    # ── PCA 3 components ──────────────────────────────────────────────────────
    pca_3     = PCA(n_components=3, random_state=RANDOM_STATE)
    data_3    = pca_3.fit_transform(df_scaled)
    km_3      = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels_3  = km_3.fit_predict(data_3)
    sil_3     = silhouette_score(data_3, labels_3)

    # ── Save best PCA-3 model ─────────────────────────────────────────────────
    joblib.dump(km_3,  MODELS_DIR / "kmeans_pca3.pkl")
    joblib.dump(pca_3, MODELS_DIR / "pca3.pkl")

    results = pd.DataFrame([
        {
            "Approach":       "Raw Scaled Data",
            "Components":     df_scaled.shape[1],
            "Variance_Pct":   100.0,
            "Silhouette":     round(raw_sil, 4),
            "Inertia":        round(km_raw.inertia_, 2),
            "ARI":            "—",
            "NMI":            "—",
        },
        {
            "Approach":       f"PCA ({n_90} components, 90% var)",
            "Components":     n_90,
            "Variance_Pct":   round(cumvar[n_90 - 1] * 100, 1),
            "Silhouette":     round(sil_n90, 4),
            "Inertia":        round(km_n90.inertia_, 2),
            "ARI":            round(adjusted_rand_score(raw_labels, labels_n90), 3),
            "NMI":            round(normalized_mutual_info_score(raw_labels, labels_n90), 3),
        },
        {
            "Approach":       "PCA (3 components)",
            "Components":     3,
            "Variance_Pct":   round(pca_3.explained_variance_ratio_.sum() * 100, 1),
            "Silhouette":     round(sil_3, 4),
            "Inertia":        round(km_3.inertia_, 2),
            "ARI":            round(adjusted_rand_score(raw_labels, labels_3), 3),
            "NMI":            round(normalized_mutual_info_score(raw_labels, labels_3), 3),
        },
    ])

    print("\n-- PCA Clustering Comparison ------------------------------")
    print(results.to_string(index=False))

    # ── Comparison plot ───────────────────────────────────────────────────────
    _plot_pca_comparison(results)

    return results, labels_3


def _plot_pca_comparison(results: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    approaches = results["Approach"].tolist()
    colors     = ["#2196F3", "#4CAF50", "#F44336"]

    for ax, metric in zip(axes, ["Silhouette", "Inertia", "Variance_Pct"]):
        vals = results[metric].tolist()
        bars = ax.bar(range(len(approaches)), vals, color=colors,
                      edgecolor="white", alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02, str(v),
                    ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(approaches)))
        ax.set_xticklabels(["Raw", f"PCA\n(90% var)", "PCA\n(3 comp)"],
                           fontsize=9)
        ax.set_title(f"{metric}", fontweight="bold")
        if metric == "Silhouette":
            ax.set_ylabel("Score (higher = better)")
        elif metric == "Inertia":
            ax.set_ylabel("WCSS (lower = better)")
        else:
            ax.set_ylabel("% Variance Explained")

    plt.suptitle("KMeans Clustering: Raw vs PCA Approaches",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = IMAGES_DIR / "14_pca_clustering_comparison.png"
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: images/14_pca_clustering_comparison.png")
