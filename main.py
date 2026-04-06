"""
main.py — End-to-end pipeline:
  Load -> Preprocess -> Find K -> KMeans -> Hierarchical -> PCA -> Visualise -> Save
Run: python main.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from src.data_loader   import load_raw_data
from src.preprocessing import preprocess
from src.model         import find_optimal_k, train_kmeans, train_hierarchical, pca_transform
from src.visualize     import (
    plot_missing_values, plot_distributions, plot_frequency_features,
    plot_correlation_heatmap, plot_engineered_features, plot_outliers,
    plot_optimal_k, plot_clusters_pca, plot_cluster_profiles,
    plot_cluster_distribution, plot_pca_variance
)
from src.config import N_CLUSTERS, REPORTS_DIR


def main():
    print("=" * 60)
    print("  Credit Card Customer Profiling & Segmentation")
    print("=" * 60)

    # 1. Load
    print("\n[1/6] Loading raw data...")
    df_raw = load_raw_data()

    # 2. EDA plots (on raw data)
    print("\n[2/6] Generating EDA plots...")
    plot_missing_values(df_raw)
    plot_distributions(df_raw)
    plot_frequency_features(df_raw)
    plot_correlation_heatmap(df_raw)

    # 3. Preprocess
    print("\n[3/6] Preprocessing & feature engineering...")
    df_scaled, df_unscaled = preprocess(df_raw, save=True)
    plot_engineered_features(df_unscaled)
    plot_outliers(df_unscaled)

    # 4. Find optimal K
    print("\n[4/6] Finding optimal K (Elbow + Silhouette)...")
    k_results = find_optimal_k(df_scaled)
    plot_optimal_k(k_results)

    # 5. KMeans
    print(f"\n[5/6] Clustering (KMeans + Hierarchical, K={N_CLUSTERS})...")
    km_labels = train_kmeans(df_scaled, n_clusters=N_CLUSTERS)
    hc_labels = train_hierarchical(df_scaled, n_clusters=N_CLUSTERS)

    # 6. PCA + plots
    print("\n[6/6] PCA & evaluation plots...")
    df_pca, ev_ratio, explained = pca_transform(df_scaled, n_components=10)
    plot_pca_variance(ev_ratio)

    df_pca2, _, exp2 = pca_transform(df_scaled, n_components=2)
    plot_clusters_pca(df_pca2, km_labels, exp2,
                      title="KMeans Clustering", fname="08_clusters_pca_kmeans.png")
    plot_clusters_pca(df_pca2, hc_labels, exp2,
                      title="Hierarchical Clustering", fname="09_clusters_pca_hierarchical.png")

    plot_cluster_profiles(df_unscaled, km_labels)
    plot_cluster_distribution(km_labels)

    # Save summary
    df_unscaled["KMeans_Cluster"]        = km_labels
    df_unscaled["Hierarchical_Cluster"]  = hc_labels
    summary = df_unscaled.groupby("KMeans_Cluster").mean().round(2)
    summary["Customer_Count"] = df_unscaled["KMeans_Cluster"].value_counts().sort_index()
    summary.to_csv(REPORTS_DIR / "cluster_summary.csv")
    print("  Saved: reports/cluster_summary.csv")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("  Images  -> images/   (committed to GitHub)")
    print("  Results -> reports/")
    print("=" * 60)


if __name__ == "__main__":
    main()
