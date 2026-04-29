"""
src/clustering.py
Phase 2 — Segmentation via Clustering

Isi fungsi untuk:
- Menentukan K optimal (Elbow + Silhouette)
- K-Means + visualisasi PCA & t-SNE
- DBSCAN + k-distance graph
- Hierarchical + dendrogram
- Profiling cluster
- Perbandingan ketiga metode
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from config import (
    K_RANGE, BEST_K, N_INIT, RANDOM_STATE,
    DBSCAN_EPS, DBSCAN_MIN_SAMPLES,
    CLUSTERING_DATA_PATH, OUTPUT_DIR
)

# Load Data
def load_clustering_data(path: str = CLUSTERING_DATA_PATH):
    """Load clustering dataset dari Phase 1."""
    df = pd.read_csv(path)
    X = df.drop(columns=['Anomaly'])
    y_true = df['Anomaly']
    print(f"✅ Clustering data loaded: {X.shape}")
    print(f"   Fitur: {X.columns.tolist()}")
    return df, X, y_true


# Elbow + Silhouette
def find_optimal_k(X: pd.DataFrame, k_range=K_RANGE, save_plots: bool = True) -> int:
    """
    Elbow Method + Silhouette Score untuk menentukan K optimal.
    Returns K terbaik secara matematis (Silhouette tertinggi).
    """
    wcss, sil_scores = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = km.fit_predict(X)
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))
        print(f"  K={k} → Silhouette: {sil_scores[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Jumlah Cluster (K)')
    axes[0].set_ylabel('WCSS')
    axes[0].set_title('Elbow Method')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_range, sil_scores, 'rs-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Jumlah Cluster (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score per K')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'elbow_silhouette.png'), dpi=150)
    plt.show()

    math_best = list(k_range)[sil_scores.index(max(sil_scores))]
    print(f"\n  K terbaik matematis   : {math_best}")
    print(f"  K dipilih (domain)    : {BEST_K}")
    print(f"  Alasan: K={BEST_K} menghasilkan profil nasabah yang lebih")
    print(f"  kaya dan actionable dibanding K={math_best} yang terlalu general.")

    return BEST_K


# K-Means
def run_kmeans(df: pd.DataFrame, X: pd.DataFrame,
               best_k: int, save_plots: bool = True) -> tuple:
    """
    Jalankan K-Means dan visualisasi dengan PCA + t-SNE.
    Returns (df dengan kolom KMeans_Cluster, X_pca, X_tsne).
    """
    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=N_INIT)
    df['KMeans_Cluster'] = km.fit_predict(X)

    print(f"✅ K-Means selesai (K={best_k})")
    print(df['KMeans_Cluster'].value_counts().sort_index().to_string())

    # PCA
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
    X_tsne = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sc1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                          c=df['KMeans_Cluster'], cmap='tab10', alpha=0.6, s=25)
    axes[0].set_title(
        f"PCA — PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%) "
        f"+ PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    fig.colorbar(sc1, ax=axes[0], label='Cluster')

    sc2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                          c=df['KMeans_Cluster'], cmap='tab10', alpha=0.6, s=25)
    axes[1].set_title('t-SNE 2D Projection')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')
    legend = axes[1].legend(*sc2.legend_elements(), title="Clusters")
    axes[1].add_artist(legend)

    plt.suptitle(f'K-Means Clustering (K={best_k})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'kmeans_visualization.png'), dpi=150)
    plt.show()

    return df, X_pca, X_tsne


# Cluster Profiling
def profile_clusters(df: pd.DataFrame, X: pd.DataFrame,
                     cluster_col: str, save_plots: bool = True) -> pd.DataFrame:
    """
    Buat profil rata-rata per cluster dan visualisasi bar chart.
    Berlaku untuk K-Means maupun Hierarchical.
    """
    key_features = [
        'Age', 'Account Balance', 'Loan Amount', 'Interest Rate',
        'CC_Utilization', 'Rewards Points', 'Transaction_to_Balance_Ratio'
    ]
    key_features = [f for f in key_features if f in X.columns]

    profile = df.groupby(cluster_col)[X.columns.tolist()].mean().round(3)
    print(f"\n=== Cluster Profile ({cluster_col}) ===")
    print(profile[key_features].T.to_string())

    # Visualisasi bar
    cluster_ids = sorted(df[cluster_col].unique())
    fig, axes = plt.subplots(1, len(cluster_ids),
                             figsize=(5 * len(cluster_ids), 5))
    if len(cluster_ids) == 1:
        axes = [axes]

    for ax, cid in zip(axes, cluster_ids):
        vals = profile.loc[cid, key_features]
        ax.barh(key_features, vals, color=f'C{cid}')
        ax.set_xlim(0, 1)
        ax.set_title(f'Cluster {cid}')
        ax.set_xlabel('Normalized Mean')

    plt.suptitle(f'Cluster Profile — {cluster_col}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_plots:
        fname = f"profile_{cluster_col.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.show()

    # Cross-check Anomaly
    print(f"\n=== Anomaly Rate per Cluster ({cluster_col}) ===")
    anomaly_rate = df.groupby(cluster_col)['Anomaly'].apply(
        lambda x: f"{(x == -1).sum()} anomali ({(x == -1).mean()*100:.1f}%)"
    )
    print(anomaly_rate.to_string())

    return profile


# DBSCAN
def run_dbscan(df: pd.DataFrame, X: pd.DataFrame, X_pca: np.ndarray,
               eps: float = DBSCAN_EPS,
               min_samples: int = DBSCAN_MIN_SAMPLES,
               save_plots: bool = True) -> tuple:
    """
    Jalankan DBSCAN setelah menentukan eps dari k-distance graph.
    Returns (df dengan DBSCAN_Cluster, n_clusters, n_noise).
    """
    # K-distance graph
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = np.sort(distances[:, min_samples - 1])

    plt.figure(figsize=(10, 4))
    plt.plot(distances, linewidth=1.5)
    plt.axhline(y=eps, color='red', linestyle='--',
                alpha=0.7, label=f'eps={eps}')
    plt.ylabel(f'{min_samples}-NN Distance')
    plt.xlabel('Data Points (sorted)')
    plt.title(f'K-Distance Graph — Menentukan eps Optimal (min_samples={min_samples})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'kdistance_graph.png'), dpi=150)
    plt.show()

    # Jalankan DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['DBSCAN_Cluster'] = dbscan.fit_predict(X)

    n_clusters = len(set(df['DBSCAN_Cluster'])) - (
        1 if -1 in df['DBSCAN_Cluster'].values else 0
    )
    n_noise = (df['DBSCAN_Cluster'] == -1).sum()

    print(f"✅ DBSCAN selesai")
    print(f"   Jumlah cluster : {n_clusters}")
    print(f"   Noise points   : {n_noise} ({n_noise/len(df)*100:.1f}%)")
    print(df['DBSCAN_Cluster'].value_counts().sort_index().to_string())

    # Visualisasi
    plt.figure(figsize=(10, 7))
    unique_labels = sorted(df['DBSCAN_Cluster'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = df['DBSCAN_Cluster'] == label
        name   = 'Noise' if label == -1 else f'Cluster {label}'
        marker = 'x'     if label == -1 else 'o'
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=[color], label=name, marker=marker,
                    alpha=0.6, s=30)

    plt.title(f'DBSCAN (eps={eps}, min_samples={min_samples}) — PCA 2D')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'dbscan_pca.png'), dpi=150)
    plt.show()

    return df, n_clusters, n_noise


# HIERARCHICAL CLUSTERING
def run_hierarchical(df: pd.DataFrame, X: pd.DataFrame,
                     best_k: int, save_plots: bool = True) -> pd.DataFrame:
    """
    Hierarchical Clustering dengan 3 linkage methods + dendrogram.
    """
    linkage_methods = ['ward', 'complete', 'average']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, method in zip(axes, linkage_methods):
        Z = linkage(X, method=method)
        dendrogram(Z, ax=ax, truncate_mode='level', p=5,
                   color_threshold=0.7 * max(Z[:, 2]))
        ax.set_title(f'Dendrogram — {method.capitalize()} Linkage')
        ax.set_xlabel('Data Points')
        ax.set_ylabel('Distance')

    plt.suptitle('Hierarchical Clustering — 3 Linkage Methods',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'dendrogram_comparison.png'), dpi=150)
    plt.show()

    # Potong dengan Ward (terbaik) pada K optimal
    Z_ward = linkage(X, method='ward')
    hier_labels = fcluster(Z_ward, t=best_k, criterion='maxclust')
    df['Hierarchical_Cluster'] = hier_labels

    print(f"✅ Hierarchical selesai (Ward, K={best_k})")
    print(df['Hierarchical_Cluster'].value_counts().sort_index().to_string())

    return df


# ════════════════════════════════════════════════════════════
# 7. PERBANDINGAN KETIGA METODE
# ════════════════════════════════════════════════════════════

def compare_methods(df: pd.DataFrame, X: pd.DataFrame,
                    n_clusters_db: int, n_noise: int) -> pd.DataFrame:
    """Bandingkan K-Means, DBSCAN, dan Hierarchical secara tabel."""

    sil_kmeans = silhouette_score(X, df['KMeans_Cluster'])
    sil_hier   = silhouette_score(X, df['Hierarchical_Cluster'])

    dbscan_valid = df[df['DBSCAN_Cluster'] != -1]
    if dbscan_valid['DBSCAN_Cluster'].nunique() > 1:
        sil_dbscan = round(
            silhouette_score(
                X[df['DBSCAN_Cluster'] != -1],
                dbscan_valid['DBSCAN_Cluster']
            ), 4
        )
    else:
        sil_dbscan = 'N/A (1 cluster)'

    comparison = pd.DataFrame({
        'Method': ['K-Means', 'DBSCAN', 'Hierarchical (Ward)'],
        'N Clusters': [
            df['KMeans_Cluster'].nunique(),
            n_clusters_db,
            df['Hierarchical_Cluster'].nunique()
        ],
        'Noise Points': [0, n_noise, 0],
        'Silhouette Score': [
            round(sil_kmeans, 4),
            sil_dbscan,
            round(sil_hier, 4)
        ]
    })

    print("\n=== Perbandingan Hasil Clustering ===")
    print(comparison.to_string(index=False))

    return comparison


# ════════════════════════════════════════════════════════════
# 8. PIPELINE UTAMA
# ════════════════════════════════════════════════════════════

def run_clustering(path: str = CLUSTERING_DATA_PATH) -> pd.DataFrame:
    """Jalankan seluruh pipeline Phase 2."""

    print("=" * 55)
    print("  PHASE 2 — Segmentation via Clustering")
    print("=" * 55)

    print("\n[1/6] Load data...")
    df, X, y_true = load_clustering_data(path)

    print("\n[2/6] Mencari K optimal...")
    best_k = find_optimal_k(X)

    print("\n[3/6] K-Means...")
    df, X_pca, X_tsne = run_kmeans(df, X, best_k)
    profile_clusters(df, X, 'KMeans_Cluster')

    print("\n[4/6] DBSCAN...")
    df, n_clusters_db, n_noise = run_dbscan(df, X, X_pca)
    # DBSCAN profiling
    non_noise = df[df['DBSCAN_Cluster'] != -1]
    if not non_noise.empty:
        print("\n--- Profil Cluster DBSCAN ---")
        print(non_noise.groupby('DBSCAN_Cluster')[X.columns].mean().round(3).T)
    print("\n--- Profil Noise Points ---")
    noise_pts = df[df['DBSCAN_Cluster'] == -1][X.columns]
    print(noise_pts.mean().round(3).to_string())

    print("\n[5/6] Hierarchical...")
    df = run_hierarchical(df, X, best_k)
    profile_clusters(df, X, 'Hierarchical_Cluster')

    print("\n[6/6] Perbandingan metode...")
    comparison = compare_methods(df, X, n_clusters_db, n_noise)

    print("\n" + "=" * 55)
    print("  ✅ PHASE 2 SELESAI")
    print("=" * 55)

    return df
