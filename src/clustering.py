"""
clustering.py — Phase 2: Segmentation via Clustering

Pada tahapan ini, kita mau menemukan kelompok-kelompok data yang
ada pada data yang terbentuk tanpa dikasih hint atau label
sehingga bersifat unsupervised learning.

Cara running:
    python src/clustering.py

Atau import dari notebook:
    from src.clustering import run_clustering
    df_clustered = run_clustering('data/dataset_clustering.csv')
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

DATA_DIR   = 'data'
OUTPUT_DIR = 'outputs/phase2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameter
RANDOM_STATE = 42
N_INIT       = 10     # n_init=10 cukup, tidak perlu 100
# BEST_K       = 3      # override domain knowledge (lihat penjelasan di find_optimal_k)
K_RANGE      = range(2, 11)
DBSCAN_EPS         = 0.5
DBSCAN_MIN_SAMPLES = 5


# ════════════════════════════════════════════════════════════
# PREPARATION
# Pada tahap ini kita mulai memuat dataset ke variable yang
# nantinya akan digunakan oleh algoritma atau metode yang dipilih.
# ════════════════════════════════════════════════════════════

def load_clustering_data(path):
    df_cluster = pd.read_csv(path)
    # Anomaly dipisah — TIDAK digunakan sebagai input clustering
    # Hanya dibuka kembali saat cross-reference di Phase 4
    X      = df_cluster.drop(columns=['Anomaly'])
    y_true = df_cluster['Anomaly']
    print(f"Shape data clustering: {X.shape}")
    print(f"Fitur: {X.columns.tolist()}")
    return df_cluster, X, y_true


# ════════════════════════════════════════════════════════════
# ELBOW METHOD & SILHOUETTE SCORE
# Pada tahap ini, elbow method dan silhouette score digunakan
# untuk mencari jumlah cluster paling optimal yang nantinya
# akan dipakai pada metode-metode yang terpilih.
# ════════════════════════════════════════════════════════════

def find_optimal_k(X, k_range=K_RANGE, save_plots=True):
    """
    Elbow Method:
    Mengukur WCSS (Within-Cluster Sum of Squares), yaitu total jarak
    setiap data point ke centroid clusternya. Semakin kecil WCSS,
    semakin "rapat" cluster yang terbentuk. Namun menambah K selalu
    menurunkan WCSS, sehingga kita mencari titik elbow yaitu titik di
    mana penurunan WCSS mulai melambat drastis. K di titik siku dianggap
    optimal karena menambah cluster setelahnya tidak memberikan manfaat
    signifikan.

    Silhouette Score:
    Mengukur seberapa baik setiap data point cocok dengan clusternya
    sendiri dibandingkan cluster tetangganya. Silhouette Score melihat
    kohesi (seberapa dekat data di dalam sebuah cluster) dan separasi
    (seberapa jauh data terpisah antar cluster). Nilainya berkisar
    antara -1 hingga 1:
    - Mendekati 1  → data point sangat cocok dengan clusternya sendiri
    - Mendekati 0  → data point berada di perbatasan antara dua cluster
    - Mendekati -1 → data point lebih cocok masuk ke cluster lain
    """
    wcss, sil_scores = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = km.fit_predict(X)
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))
        print(f"  K={k} -> Silhouette: {sil_scores[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Jumlah Cluster (K)'); axes[0].set_ylabel('WCSS')
    axes[0].set_title('Elbow Method'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_range, sil_scores, 'rs-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Jumlah Cluster (K)'); axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score per K'); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_plots: plt.savefig(os.path.join(OUTPUT_DIR,'elbow_silhouette.png'), dpi=150)
    plt.show()

    math_best = list(k_range)[sil_scores.index(max(sil_scores))]
    BEST_K = math_best
    print(f"\nK terbaik secara matematis : {math_best}")
    print(f"K yang dipilih untuk analisis : {BEST_K}")
    # Pada dataset ini, dapat dilihat bahwa K=2 adalah yang paling optimal
    # dilihat dari kedua metode. Namun K=2 hanya menghasilkan dua kelompok
    # yang terlalu general dan tidak memberikan insight yang actionable.
    # Dataset sintetis cenderung tidak memiliki natural cluster yang tajam
    # sehingga Silhouette Score rendah di semua K. Oleh karena itu, K=3
    # dipilih berdasarkan domain knowledge karena menghasilkan segmentasi
    # nasabah yang lebih kaya dan dapat diinterpretasikan secara bisnis.
    return BEST_K


# ════════════════════════════════════════════════════════════
# K-MEANS
# Pada tahap ini, kita menggunakan K-Means untuk membuat
# clustering pada dataset.
#
# K-Means adalah algoritma clustering berbasis centroid yang
# bekerja dengan cara:
# 1. Menempatkan K centroid secara acak
# 2. Mengelompokkan setiap data point ke centroid terdekat
#    (berdasarkan jarak Euclidean)
# 3. Memperbarui posisi centroid ke rata-rata semua anggota
# 4. Mengulang langkah tersebut hingga centroid konvergen
#
# Visualisasi menggunakan PCA dan t-SNE untuk mereduksi ke 2D.
# PCA = linear, cepat, menangkap variansi terbesar.
# t-SNE = non-linear, lebih baik untuk memperlihatkan struktur
# cluster yang tidak linear, namun jarak antar cluster tidak
# dapat diinterpretasikan secara absolut.
# ════════════════════════════════════════════════════════════

def run_kmeans(df_cluster, X, best_k, save_plots=True):
    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=N_INIT)
    df_cluster['KMeans_Cluster'] = km_final.fit_predict(X)

    print(f"Distribusi cluster K-Means (K={best_k}):")
    print(df_cluster['KMeans_Cluster'].value_counts().sort_index())

    # Visualisasi dengan PCA 2D
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                          c=df_cluster['KMeans_Cluster'],
                          cmap='tab10', alpha=0.6, s=30)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'K-Means Clustering (K={best_k})')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.tight_layout()
    if save_plots: plt.savefig(os.path.join(OUTPUT_DIR,'kmeans_pca.png'), dpi=150)
    plt.show()

    # Visualisasi dengan t-SNE
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
    X_tsne = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                         c=df_cluster['KMeans_Cluster'],
                         cmap='Dark2', alpha=0.8, s=30)
    ax.set_title('t-SNE 2D Projection')
    ax.set_xlabel('Dimension 1'); ax.set_ylabel('Dimension 2')
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    plt.suptitle(f'K-Means Cluster Visualization (K={best_k})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_plots: plt.savefig(os.path.join(OUTPUT_DIR,'kmeans_tsne.png'), dpi=150)
    plt.show()

    return df_cluster, X_pca, X_tsne


# ════════════════════════════════════════════════════════════
# CLUSTER PROFILING
# Pada tahap ini, kita mau menginterpretasikan cluster yang
# telah terbentuk secara matematis ke pemahaman bisnis agar
# dapat dilihat insight-nya.
# ════════════════════════════════════════════════════════════

def profile_clusters(df_cluster, X, cluster_col, save_plots=True):
    key_features = [f for f in [
        'Age','Account Balance','Loan Amount','Interest Rate',
        'CC_Utilization','Rewards Points','Transaction_to_Balance_Ratio'
    ] if f in X.columns]

    profile = df_cluster.groupby(cluster_col)[X.columns.tolist()].mean().round(3)
    print(f"\n=== Cluster Profile ({cluster_col}) ===")
    print(profile[key_features].T.to_string())

    cluster_ids = sorted(df_cluster[cluster_col].unique())
    fig, axes = plt.subplots(1, len(cluster_ids), figsize=(5*len(cluster_ids), 5))
    if len(cluster_ids) == 1: axes = [axes]

    for ax, cid in zip(axes, cluster_ids):
        vals = profile.loc[cid, key_features]
        ax.barh(key_features, vals, color=f'C{cid}')
        ax.set_xlim(0, 1)
        ax.set_title(f'Cluster {cid}')
        ax.set_xlabel('Normalized Mean Value')

    plt.suptitle(f'Cluster Profile — {cluster_col}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_plots:
        fname = f"profile_{cluster_col.lower().replace(' ','_')}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.show()

    # Cross-check dengan Anomaly (hanya untuk insight, bukan input model)
    print(f"\n=== Anomaly Rate per Cluster ({cluster_col}) ===")
    print(df_cluster.groupby(cluster_col)['Anomaly'].apply(
        lambda x: f"{(x==-1).sum()} anomali ({(x==-1).mean()*100:.1f}%)"
    ))
    return profile


# ════════════════════════════════════════════════════════════
# DBSCAN
# DBSCAN bekerja berdasarkan kepadatan data:
# - Data point yang memiliki cukup tetangga dalam radius eps
#   akan membentuk cluster
# - Data point yang terisolasi = noise point (label = -1)
#
# DBSCAN digunakan bukan untuk menggantikan K-Means, melainkan
# untuk mengidentifikasi outlier struktural yang tidak masuk
# ke pola kepadatan manapun. Noise points ini menjadi kandidat
# kuat untuk diinvestigasi di Phase 4 (Anomaly Detection).
#
# Parameter:
# - eps: radius pencarian tetangga, ditentukan dari k-distance graph
# - min_samples: minimum tetangga agar titik dianggap core point
#   (konvensi umum = 5 untuk dataset ribuan baris)
# ════════════════════════════════════════════════════════════

def run_dbscan(df_cluster, X, X_pca, eps=DBSCAN_EPS,
               min_samples=DBSCAN_MIN_SAMPLES, save_plots=True):

    # K-distance graph untuk menentukan eps optimal
    # n_neighbors harus sama dengan min_samples yang akan dipakai
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = np.sort(distances[:, min_samples - 1])

    plt.figure(figsize=(10, 4))
    plt.plot(distances, linewidth=1.5)
    plt.axhline(y=eps, color='red', linestyle='--', alpha=0.7, label=f'eps={eps}')
    plt.ylabel(f'{min_samples}-NN Distance'); plt.xlabel('Data Points (sorted)')
    plt.title(f'K-Distance Graph — Menentukan eps Optimal (min_samples={min_samples})')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    if save_plots: plt.savefig(os.path.join(OUTPUT_DIR,'kdistance_graph.png'), dpi=150)
    plt.show()

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df_cluster['DBSCAN_Cluster'] = dbscan.fit_predict(X)

    n_clusters_db = len(set(df_cluster['DBSCAN_Cluster'])) - (
        1 if -1 in df_cluster['DBSCAN_Cluster'].values else 0
    )
    n_noise = (df_cluster['DBSCAN_Cluster'] == -1).sum()

    print(f"Jumlah cluster DBSCAN : {n_clusters_db}")
    print(f"Noise points          : {n_noise} ({n_noise/len(df_cluster)*100:.1f}%)")
    print(df_cluster['DBSCAN_Cluster'].value_counts().sort_index())

    # Visualisasi DBSCAN di atas PCA projection
    plt.figure(figsize=(10, 7))
    unique_labels = sorted(df_cluster['DBSCAN_Cluster'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        mask   = df_cluster['DBSCAN_Cluster'] == label
        name   = 'Noise' if label == -1 else f'Cluster {label}'
        marker = 'x'     if label == -1 else 'o'
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=[color], label=name, marker=marker, alpha=0.6, s=30)
    plt.title(f'DBSCAN (eps={eps}, min_samples={min_samples}) — PCA 2D')
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend(); plt.tight_layout()
    if save_plots: plt.savefig(os.path.join(OUTPUT_DIR,'dbscan_pca.png'), dpi=150)
    plt.show()

    # DBSCAN Cluster Profiling
    print("=== DBSCAN Cluster Profile ===")
    non_noise = df_cluster[df_cluster['DBSCAN_Cluster'] != -1]
    if not non_noise.empty:
        print(non_noise.groupby('DBSCAN_Cluster')[X.columns.tolist()].mean().round(3).T.to_string())
    print("\n=== DBSCAN Noise Points Profile ===")
    noise_pts = df_cluster[df_cluster['DBSCAN_Cluster'] == -1][X.columns.tolist()]
    print(noise_pts.mean().round(3).to_string())
    print("\n=== Anomaly Rate — DBSCAN ===")
    print(df_cluster.groupby('DBSCAN_Cluster')['Anomaly'].apply(
        lambda x: f"{(x==-1).sum()} anomali ({(x==-1).mean()*100:.1f}%)"
    ))

    return df_cluster, n_clusters_db, n_noise


# ════════════════════════════════════════════════════════════
# HIERARCHICAL CLUSTERING
# Hierarchical Clustering membangun hierarki cluster secara
# bertahap tanpa membutuhkan K di awal.
#
# Tiga linkage method dibandingkan:
# - Ward     : meminimalkan peningkatan total WCSS — cluster seimbang
# - Complete : jarak maksimum antar cluster — sensitif outlier
# - Average  : rata-rata jarak semua pasangan — kompromi
#
# Dendrogram divisualisasikan untuk melihat struktur hierarki.
# Jumlah cluster ditentukan dengan memotong dendrogram pada
# K yang sudah ditetapkan dari Elbow + Silhouette.
# ════════════════════════════════════════════════════════════

def run_hierarchical(df_cluster, X, best_k, save_plots=True):
    linkage_methods = ['ward', 'complete', 'average']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, method in zip(axes, linkage_methods):
        Z = linkage(X, method=method)
        dendrogram(Z, ax=ax, truncate_mode='level', p=5,
                   color_threshold=0.7 * max(Z[:, 2]))
        ax.set_title(f'Dendrogram — {method.capitalize()} Linkage')
        ax.set_xlabel('Data Points'); ax.set_ylabel('Distance')

    plt.suptitle('Hierarchical Clustering — Perbandingan 3 Linkage Methods',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_plots: plt.savefig(os.path.join(OUTPUT_DIR,'dendrogram_comparison.png'), dpi=150)
    plt.show()

    # Potong dendrogram dengan Ward (terbaik) pada K optimal
    Z_ward = linkage(X, method='ward')
    hier_labels = fcluster(Z_ward, t=best_k, criterion='maxclust')
    df_cluster['Hierarchical_Cluster'] = hier_labels

    print(f"Distribusi cluster Hierarchical (Ward, K={best_k}):")
    print(df_cluster['Hierarchical_Cluster'].value_counts().sort_index())
    return df_cluster


# ════════════════════════════════════════════════════════════
# COMPARISON BETWEEN MODELS
# Membandingkan K-Means, DBSCAN, dan Hierarchical secara
# sistematis untuk menentukan metode terbaik bagi dataset ini.
# ════════════════════════════════════════════════════════════

def compare_methods(df_cluster, X, n_clusters_db, n_noise):
    sil_kmeans = silhouette_score(X, df_cluster['KMeans_Cluster'])
    sil_hier   = silhouette_score(X, df_cluster['Hierarchical_Cluster'])

    dbscan_valid = df_cluster[df_cluster['DBSCAN_Cluster'] != -1]
    if dbscan_valid['DBSCAN_Cluster'].nunique() > 1:
        sil_dbscan = round(silhouette_score(
            X[df_cluster['DBSCAN_Cluster'] != -1],
            dbscan_valid['DBSCAN_Cluster']
        ), 4)
    else:
        sil_dbscan = 'N/A (1 cluster)'

    comparison = pd.DataFrame({
        'Method'          : ['K-Means','DBSCAN','Hierarchical (Ward)'],
        'N Clusters'      : [df_cluster['KMeans_Cluster'].nunique(),
                             n_clusters_db,
                             df_cluster['Hierarchical_Cluster'].nunique()],
        'Noise Points'    : [0, n_noise, 0],
        'Silhouette Score': [round(sil_kmeans, 4), sil_dbscan, round(sil_hier, 4)]
    })
    print("\nPerbandingan Hasil Clustering\n")
    print(comparison.to_string(index=False))
    return comparison


# ════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════

def run_clustering(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, 'dataset_clustering.csv')

    print("=" * 55)
    print("  PHASE 2 — Segmentation via Clustering")
    print("=" * 55)

    print("\n[1/6] Load data...")
    df_cluster, X, y_true = load_clustering_data(path)

    print("\n[2/6] Elbow Method & Silhouette Score...")
    best_k = find_optimal_k(X)

    print(f"\n[3/6] K-Means (K={best_k})...")
    df_cluster, X_pca, X_tsne = run_kmeans(df_cluster, X, best_k)
    profile_clusters(df_cluster, X, 'KMeans_Cluster')

    print("\n[4/6] DBSCAN...")
    df_cluster, n_clusters_db, n_noise = run_dbscan(df_cluster, X, X_pca)

    print(f"\n[5/6] Hierarchical (Ward, K={best_k})...")
    df_cluster = run_hierarchical(df_cluster, X, best_k)
    profile_clusters(df_cluster, X, 'Hierarchical_Cluster')

    print("\n[6/6] Perbandingan ketiga metode...")
    compare_methods(df_cluster, X, n_clusters_db, n_noise)

    print("\n" + "=" * 55)
    print("  PHASE 2 SELESAI")
    print("=" * 55)
    return df_cluster


if __name__ == '__main__':
    run_clustering()
