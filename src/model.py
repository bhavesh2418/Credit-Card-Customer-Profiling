"""
model.py — KMeans + Hierarchical clustering, PCA, optimal K selection.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from src.config import RANDOM_STATE, K_RANGE, N_CLUSTERS, MODELS_DIR


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
