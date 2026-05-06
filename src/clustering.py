"""
clustering.py — Phase 2: Segmentation via Clustering

Pada tahapan ini, kita mau menemukan kelompok-kelompok data yang
ada pada data yang terbentuk tanpa dikasih hint atau label
sehingga bersifat unsupervised learning.

Pendekatan yang digunakan:
    Data X (11 fitur) → PCA dulu → X_pca → Clustering → Interpretasi

Alasan PCA sebelum clustering:
- Mengurangi curse of dimensionality pada jarak Euclidean
- Menghilangkan noise dari komponen dengan variance rendah
- Visualisasi dan clustering menggunakan data yang sama (konsisten)
- Profiling tetap menggunakan X original agar interpretasi bisnis valid

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
from sklearn.preprocessing import StandardScaler
from scipy.stats import mstats

DATA_DIR   = 'data'
OUTPUT_DIR = 'outputs/phase2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameter
RANDOM_STATE           = 42
N_INIT                 = 10
BEST_K                 = 3      # override domain knowledge
K_RANGE                = range(2, 11)
DBSCAN_MIN_SAMPLES     = 5
PCA_VARIANCE_THRESHOLD = 0.80   # tangkap minimal 80% variance


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
    print(f"Shape data clustering : {X.shape}")
    print(f"Fitur                 : {X.columns.tolist()}")
    return df_cluster, X, y_true


# ════════════════════════════════════════════════════════════
# PCA — REDUKSI DIMENSI SEBELUM CLUSTERING
# PCA dilakukan SEBELUM clustering, bukan hanya untuk visualisasi.
# Tujuannya:
# 1. Mengatasi curse of dimensionality — pada dimensi tinggi,
#    jarak Euclidean antar data point cenderung seragam sehingga
#    K-Means kesulitan menemukan cluster yang bermakna.
# 2. Noise reduction — komponen dengan variance rendah umumnya
#    merepresentasikan noise. Dengan hanya mengambil komponen
#    yang menangkap 80%+ variance, kita hilangkan noise sebelum
#    clustering dimulai.
# 3. Konsistensi — visualisasi dan clustering menggunakan data
#    yang sama (X_pca), sehingga cluster di plot benar-benar
#    merepresentasikan cluster yang terbentuk.
#
# Catatan: Profiling cluster tetap menggunakan X original
# agar interpretasi bisnis setiap fitur tetap bermakna.
# ════════════════════════════════════════════════════════════

def apply_pca(X, threshold=PCA_VARIANCE_THRESHOLD, save_plots=True):
    """
    Tentukan n_components optimal lalu transform X ke PCA space.
    Returns (X_pca, pca_final, n_components).
    """
    # Fit semua komponen dulu untuk lihat cumulative variance
    pca_check = PCA(random_state=RANDOM_STATE)
    pca_check.fit(X)
    cumulative_variance = pca_check.explained_variance_ratio_.cumsum()

    # Plot cumulative variance
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(cumulative_variance)+1),
             cumulative_variance * 100, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=threshold*100, color='red', linestyle='--',
                label=f'{threshold*100:.0f}% threshold')
    plt.axhline(y=90, color='orange', linestyle='--', label='90% threshold')
    plt.xlabel('Jumlah Komponen PCA')
    plt.ylabel('Cumulative Variance Explained (%)')
    plt.title('PCA — Menentukan Jumlah Komponen Optimal')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'pca_variance.png'), dpi=150)
    plt.show()

    # Pilih n_components berdasarkan threshold
    n_components = next(
        i+1 for i, v in enumerate(cumulative_variance) if v >= threshold
    )

    # Transform ke PCA space
    pca_final = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca_final.fit_transform(X)

    print(f"=== Hasil PCA ===")
    print(f"Shape sebelum PCA  : {X.shape}")
    print(f"Shape setelah PCA  : {X_pca.shape}")
    print(f"Komponen dipilih   : {n_components} "
          f"(menangkap {cumulative_variance[n_components-1]*100:.1f}% variance)")
    print(f"\nVariance per komponen:")
    for i, var in enumerate(pca_final.explained_variance_ratio_):
        print(f"  PC{i+1}: {var*100:.1f}%  "
              f"(cumulative: {cumulative_variance[i]*100:.1f}%)")

    return X_pca, pca_final, n_components


# ════════════════════════════════════════════════════════════
# ELBOW METHOD & SILHOUETTE SCORE
# Dijalankan pada X_pca (bukan X original) agar hasil valid
# untuk algoritma yang juga menggunakan X_pca.
# ════════════════════════════════════════════════════════════

def find_optimal_k(X_pca, k_range=K_RANGE, save_plots=True):
    """
    Elbow Method:
    Mengukur WCSS (Within-Cluster Sum of Squares), yaitu total jarak
    setiap data point ke centroid clusternya. Semakin kecil WCSS,
    semakin rapat cluster yang terbentuk. Namun menambah K selalu
    menurunkan WCSS, sehingga kita mencari titik elbow yaitu titik di
    mana penurunan WCSS mulai melambat drastis.

    Silhouette Score:
    Mengukur seberapa baik setiap data point cocok dengan clusternya
    sendiri dibandingkan cluster tetangganya. Nilainya berkisar -1 hingga 1:
    - Mendekati 1  → data point sangat cocok dengan clusternya sendiri
    - Mendekati 0  → data point berada di perbatasan antara dua cluster
    - Mendekati -1 → data point lebih cocok masuk ke cluster lain
    """
    wcss, sil_scores = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = km.fit_predict(X_pca)   # ← X_pca
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(X_pca, labels))
        print(f"  K={k} -> Silhouette: {sil_scores[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Jumlah Cluster (K)'); axes[0].set_ylabel('WCSS')
    axes[0].set_title('Elbow Method (PCA space)'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_range, sil_scores, 'rs-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Jumlah Cluster (K)'); axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score per K (PCA space)'); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'elbow_silhouette.png'), dpi=150)
    plt.show()

    math_best = list(k_range)[sil_scores.index(max(sil_scores))]
    BEST_K = math_best
    print(f"\nK terbaik secara matematis  : {math_best}")
    print(f"K yang dipilih (domain)     : {BEST_K}")
    # Dataset sintetis cenderung tidak memiliki natural cluster yang tajam
    # sehingga Silhouette Score rendah di semua K. K=2 secara matematis
    # sering menang tapi terlalu general untuk profiling nasabah perbankan.
    # K=3 dipilih berdasarkan domain knowledge karena menghasilkan segmentasi
    # yang lebih kaya dan actionable secara bisnis.
    return BEST_K


# ════════════════════════════════════════════════════════════
# K-MEANS
# Clustering dilakukan di X_pca (bukan X original).
# Profiling dilakukan di X original agar interpretasi bisnis valid.
#
# K-Means bekerja dengan cara:
# 1. Menempatkan K centroid secara acak
# 2. Mengelompokkan setiap data point ke centroid terdekat (Euclidean)
# 3. Memperbarui posisi centroid ke rata-rata semua anggotanya
# 4. Mengulang hingga centroid konvergen
# ════════════════════════════════════════════════════════════

def run_kmeans(df_cluster, X, X_pca, pca_final, best_k, save_plots=True):
    # ── Clustering di X_pca ────────────────────────────────
    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=N_INIT)
    df_cluster['KMeans_Cluster'] = km_final.fit_predict(X_pca)   # ← X_pca

    print(f"Distribusi cluster K-Means (K={best_k}):")
    print(df_cluster['KMeans_Cluster'].value_counts().sort_index())

    # ── Visualisasi PC1 vs PC2 (dari X_pca langsung) ───────
    # Tidak perlu PCA ulang karena X_pca sudah di-reduce
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    scatter1 = axes[0].scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=df_cluster['KMeans_Cluster'], cmap='tab10', alpha=0.6, s=25
    )
    axes[0].set_title(f'K-Means (K={best_k}) — PC1 vs PC2')
    axes[0].set_xlabel(f'PC1 ({pca_final.explained_variance_ratio_[0]*100:.1f}% variance)')
    axes[0].set_ylabel(f'PC2 ({pca_final.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')

    # PC1 vs PC3 jika ada
    if X_pca.shape[1] >= 3:
        scatter2 = axes[1].scatter(
            X_pca[:, 0], X_pca[:, 2],
            c=df_cluster['KMeans_Cluster'], cmap='tab10', alpha=0.6, s=25
        )
        axes[1].set_title(f'K-Means (K={best_k}) — PC1 vs PC3')
        axes[1].set_xlabel(f'PC1 ({pca_final.explained_variance_ratio_[0]*100:.1f}% variance)')
        axes[1].set_ylabel(f'PC3 ({pca_final.explained_variance_ratio_[2]*100:.1f}% variance)')
        plt.colorbar(scatter2, ax=axes[1], label='Cluster')
    else:
        axes[1].axis('off')

    plt.suptitle(f'K-Means Clustering (K={best_k}) — PCA Space',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'kmeans_pca.png'), dpi=150)
    plt.show()

    # ── t-SNE dari X_pca ───────────────────────────────────
    # t-SNE dijalankan dari X_pca (bukan X) agar konsisten
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
    X_tsne = tsne.fit_transform(X_pca)   # ← X_pca

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=df_cluster['KMeans_Cluster'], cmap='Dark2', alpha=0.8, s=25
    )
    ax.set_title(f'K-Means (K={best_k}) — t-SNE dari PCA Space')
    ax.set_xlabel('Dimension 1'); ax.set_ylabel('Dimension 2')
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'kmeans_tsne.png'), dpi=150)
    plt.show()

    return df_cluster, X_tsne


# ════════════════════════════════════════════════════════════
# CLUSTER PROFILING
# Profiling menggunakan X original (bukan X_pca) karena nilai
# PCA tidak dapat diinterpretasikan secara bisnis. X original
# yang sudah ternormalisasi digunakan agar nilai antar fitur
# dapat dibandingkan secara langsung.
# ════════════════════════════════════════════════════════════

def profile_clusters(df_cluster, X, cluster_col, save_plots=True):
    key_features = [f for f in [
        'Age', 'Account Balance', 'Loan Amount', 'Interest Rate',
        'CC_Utilization', 'Rewards Points', 'Transaction_to_Balance_Ratio'
    ] if f in X.columns]

    # Profil rata-rata per cluster (X original)
    profile = df_cluster.groupby(cluster_col)[X.columns.tolist()].mean().round(3)
    print(f"\n=== Cluster Profile — {cluster_col} (nilai X original) ===")
    print(profile[key_features].T.to_string())

    # Visualisasi bar chart per cluster
    cluster_ids = sorted(df_cluster[cluster_col].unique())
    fig, axes = plt.subplots(1, len(cluster_ids),
                             figsize=(5 * len(cluster_ids), 5))
    if len(cluster_ids) == 1: axes = [axes]

    for ax, cid in zip(axes, cluster_ids):
        vals = profile.loc[cid, key_features]
        ax.barh(key_features, vals, color=f'C{int(cid)}')
        ax.set_xlim(0, 1)
        ax.set_title(f'Cluster {cid}')
        ax.set_xlabel('Normalized Mean Value')

    plt.suptitle(f'Cluster Profile — {cluster_col}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_plots:
        fname = f"profile_{cluster_col.lower().replace(' ','_')}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.show()

    # Cross-check Anomaly (hanya untuk insight, bukan input model)
    print(f"\n=== Anomaly Rate per Cluster — {cluster_col} ===")
    print(df_cluster.groupby(cluster_col)['Anomaly'].apply(
        lambda x: f"{(x==-1).sum()} anomali ({(x==-1).mean()*100:.1f}%)"
    ))
    return profile


# ════════════════════════════════════════════════════════════
# DBSCAN
# Dijalankan di X_pca (konsisten dengan K-Means dan Hierarchical).
# eps dicari otomatis agar tidak hardcode — iterasi eps dari
# 0.30 sampai 2.0 dan pilih yang memenuhi kriteria:
# 1. Jumlah cluster <= 3
# 2. Noise points 5%-20%
# Kalau tidak ada yang memenuhi, pilih yang mendekati 10% noise.
# ════════════════════════════════════════════════════════════

def run_dbscan(df_cluster, X, X_pca, pca_final,
               min_samples=DBSCAN_MIN_SAMPLES, save_plots=True):

    # ── K-Distance Graph (X_pca) ────────────────────────────
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_pca)
    distances, _ = nbrs.kneighbors(X_pca)
    distances = np.sort(distances[:, min_samples - 1])

    plt.figure(figsize=(10, 4))
    plt.plot(distances, linewidth=1.5)
    plt.ylabel(f'{min_samples}-NN Distance')
    plt.xlabel('Data Points (sorted)')
    plt.title(f'K-Distance Graph (min_samples={min_samples}) — PCA Space')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'kdistance_graph.png'), dpi=150)
    plt.show()

    # ── Auto-search eps terbaik ─────────────────────────────
    print("=== Iterasi Pencarian eps Optimal ===")
    eps_candidates = np.arange(0.30, 2.0, 0.01).round(2)
    results = []

    for eps_test in eps_candidates:
        db_test   = DBSCAN(eps=eps_test, min_samples=min_samples)
        labels    = db_test.fit_predict(X_pca)
        n_clust   = len(set(labels)) - (1 if -1 in labels else 0)
        n_nois    = (labels == -1).sum()
        pct_noise = n_nois / len(labels) * 100
        results.append({'eps': eps_test, 'n_clusters': n_clust,
                        'n_noise': n_nois, 'pct_noise': round(pct_noise, 1)})

    results_df = pd.DataFrame(results)

    # Kriteria ideal: cluster <= 3, noise 5%-20%
    good = results_df[
        (results_df['n_clusters'] <= 3) &
        (results_df['pct_noise'] >= 5) &
        (results_df['pct_noise'] <= 20)
    ]

    if not good.empty:
        best_row = good.iloc[(good['pct_noise'] - 10).abs().argsort()[:1]]
        best_eps = best_row['eps'].values[0]
        print(f"\nKandidat eps yang memenuhi kriteria:")
        print(good.to_string(index=False))
        print(f"\n✅ eps terpilih : {best_eps} "
              f"(clusters: {best_row['n_clusters'].values[0]}, "
              f"noise: {best_row['pct_noise'].values[0]}%)")
    else:
        best_row = results_df.iloc[(results_df['pct_noise'] - 10).abs().argsort()[:1]]
        best_eps = best_row['eps'].values[0]
        print(f"\nTidak ada eps yang memenuhi semua kriteria.")
        print(f"⚠️  Fallback eps : {best_eps} "
              f"(clusters: {best_row['n_clusters'].values[0]}, "
              f"noise: {best_row['pct_noise'].values[0]}%)")

    # K-Distance Graph dengan eps terpilih
    plt.figure(figsize=(10, 4))
    plt.plot(distances, linewidth=1.5)
    plt.axhline(y=best_eps, color='red', linestyle='--',
                alpha=0.8, label=f'eps terpilih = {best_eps}')
    plt.ylabel(f'{min_samples}-NN Distance')
    plt.xlabel('Data Points (sorted)')
    plt.title(f'K-Distance Graph — eps={best_eps} dipilih otomatis')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'kdistance_graph_final.png'), dpi=150)
    plt.show()

    # ── Jalankan DBSCAN ─────────────────────────────────────
    dbscan = DBSCAN(eps=best_eps, min_samples=min_samples)
    df_cluster['DBSCAN_Cluster'] = dbscan.fit_predict(X_pca)   # ← X_pca

    n_clusters_db = len(set(df_cluster['DBSCAN_Cluster'])) - (
        1 if -1 in df_cluster['DBSCAN_Cluster'].values else 0
    )
    n_noise = (df_cluster['DBSCAN_Cluster'] == -1).sum()

    print(f"\nJumlah cluster DBSCAN : {n_clusters_db}")
    print(f"Noise points          : {n_noise} ({n_noise/len(df_cluster)*100:.1f}%)")
    print(df_cluster['DBSCAN_Cluster'].value_counts().sort_index())

    # ── Visualisasi ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    unique_labels = sorted(df_cluster['DBSCAN_Cluster'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for ax, (pc_x, pc_y) in zip(axes, [(0, 1), (0, 2)]):
        if pc_y >= X_pca.shape[1]:
            ax.axis('off'); continue
        for label, color in zip(unique_labels, colors):
            mask   = df_cluster['DBSCAN_Cluster'] == label
            name   = 'Noise' if label == -1 else f'Cluster {label}'
            marker = 'x'     if label == -1 else 'o'
            size   = 60      if label == -1 else 25
            alpha  = 0.8     if label == -1 else 0.5
            ax.scatter(X_pca[mask, pc_x], X_pca[mask, pc_y],
                       c=[color], label=name, marker=marker,
                       alpha=alpha, s=size)
        ax.set_title(f'PC{pc_x+1} vs PC{pc_y+1}')
        ax.set_xlabel(f'PC{pc_x+1} ({pca_final.explained_variance_ratio_[pc_x]*100:.1f}%)')
        ax.set_ylabel(f'PC{pc_y+1} ({pca_final.explained_variance_ratio_[pc_y]*100:.1f}%)')
        ax.legend(fontsize=8)

    plt.suptitle(
        f'DBSCAN (eps={best_eps}, min_samples={min_samples}) — PCA Space\n'
        f'{n_clusters_db} cluster + {n_noise} noise points ({n_noise/len(df_cluster)*100:.1f}%)',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'dbscan_pca.png'), dpi=150)
    plt.show()

    # ── Profiling noise vs cluster ──────────────────────────
    # Profiling menggunakan X original
    print("\n=== DBSCAN Noise Points Profile (X original) ===")
    noise_profile = df_cluster[
        df_cluster['DBSCAN_Cluster'] == -1
    ][X.columns.tolist()].mean().round(3)
    print(noise_profile.to_string())

    print("\n=== Anomaly Rate — Noise vs Cluster ===")
    noise_rows   = df_cluster[df_cluster['DBSCAN_Cluster'] == -1]
    cluster_rows = df_cluster[df_cluster['DBSCAN_Cluster'] != -1]
    print(f"Noise points : {(noise_rows['Anomaly']==-1).sum()} "
          f"anomali dari {len(noise_rows)} "
          f"({(noise_rows['Anomaly']==-1).mean()*100:.1f}%)")
    print(f"Cluster      : {(cluster_rows['Anomaly']==-1).sum()} "
          f"anomali dari {len(cluster_rows)} "
          f"({(cluster_rows['Anomaly']==-1).mean()*100:.1f}%)")

    return df_cluster, n_clusters_db, n_noise, best_eps


# ════════════════════════════════════════════════════════════
# HIERARCHICAL CLUSTERING
# linkage() dijalankan di X_pca (bukan X original) agar
# konsisten dengan K-Means dan DBSCAN.
#
# Tiga linkage method dibandingkan:
# - Ward     : meminimalkan peningkatan total WCSS — cluster seimbang
# - Complete : jarak maksimum antar cluster — sensitif outlier
# - Average  : rata-rata jarak semua pasangan — kompromi
# ════════════════════════════════════════════════════════════

def run_hierarchical(df_cluster, X, X_pca, pca_final, best_k, save_plots=True):
    linkage_methods = ['ward', 'complete', 'average']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, method in zip(axes, linkage_methods):
        Z = linkage(X_pca, method=method)   # ← X_pca
        dendrogram(Z, ax=ax, truncate_mode='level', p=5,
                   color_threshold=0.7 * max(Z[:, 2]))
        ax.set_title(f'Dendrogram — {method.capitalize()} Linkage')
        ax.set_xlabel('Data Points'); ax.set_ylabel('Distance')

    plt.suptitle('Hierarchical Clustering — 3 Linkage Methods (PCA Space)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'dendrogram_comparison.png'), dpi=150)
    plt.show()

    # Potong dengan Ward pada K optimal
    Z_ward = linkage(X_pca, method='ward')   # ← X_pca
    hier_labels = fcluster(Z_ward, t=best_k, criterion='maxclust')
    df_cluster['Hierarchical_Cluster'] = hier_labels

    print(f"Distribusi cluster Hierarchical (Ward, K={best_k}):")
    print(df_cluster['Hierarchical_Cluster'].value_counts().sort_index())
    return df_cluster


# ════════════════════════════════════════════════════════════
# COMPARISON BETWEEN MODELS
# Silhouette Score dihitung di X_pca agar konsisten dengan
# space yang digunakan oleh masing-masing algoritma.
# ════════════════════════════════════════════════════════════

def compare_methods(df_cluster, X, X_pca, n_clusters_db, n_noise):
    # Silhouette dihitung di X_pca — konsisten dengan clustering
    sil_kmeans = silhouette_score(X_pca, df_cluster['KMeans_Cluster'])
    sil_hier   = silhouette_score(X_pca, df_cluster['Hierarchical_Cluster'])

    dbscan_valid = df_cluster[df_cluster['DBSCAN_Cluster'] != -1]
    if dbscan_valid['DBSCAN_Cluster'].nunique() > 1:
        sil_dbscan = round(silhouette_score(
            X_pca[df_cluster['DBSCAN_Cluster'] != -1],
            dbscan_valid['DBSCAN_Cluster']
        ), 4)
    else:
        sil_dbscan = 'N/A (1 cluster)'

    comparison = pd.DataFrame({
        'Method'          : ['K-Means', 'DBSCAN', 'Hierarchical (Ward)'],
        'N Clusters'      : [df_cluster['KMeans_Cluster'].nunique(),
                             n_clusters_db,
                             df_cluster['Hierarchical_Cluster'].nunique()],
        'Noise Points'    : [0, n_noise, 0],
        'Silhouette Score': [round(sil_kmeans, 4), sil_dbscan, round(sil_hier, 4)]
    })
    print("\n=== Perbandingan Hasil Clustering (dievaluasi di PCA space) ===")
    print(comparison.to_string(index=False))
    return comparison

from sklearn.preprocessing import StandardScaler

def prepare_for_pca(X, lower_pct=1, upper_pct=99):
    """
    Pipeline sebelum PCA:
    1. Winsorization — cap nilai ekstrem
    2. StandardScaler — samakan variance semua fitur ke mean=0, std=1
    
    StandardScaler di sini bukan untuk menggantikan scaling di Phase 1,
    melainkan khusus untuk mempersiapkan data agar PCA tidak bias
    terhadap fitur dengan variance tinggi.
    """
    X_prep = X.copy()

    # Step 1: Cap outliers
    print("=== Step 1: Winsorization ===")
    for col in X_prep.columns:
        lower = np.percentile(X_prep[col], lower_pct)
        upper = np.percentile(X_prep[col], upper_pct)
        n_capped = ((X_prep[col] < lower) | (X_prep[col] > upper)).sum()
        X_prep[col] = X_prep[col].clip(lower, upper)
        if n_capped > 0:
            print(f"  {col}: {n_capped} nilai di-cap [{lower:.3f}, {upper:.3f}]")

    # Step 2: StandardScaler — wajib sebelum PCA
    # Tanpa ini, fitur dengan variance besar akan mendominasi PC1
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_prep),
        columns=X_prep.columns,
        index=X_prep.index
    )

    print(f"\n=== Variance Setelah StandardScaler ===")
    print(X_scaled.var().round(4).to_string())
    print(f"\nSemua variance = 1.0 → PCA tidak bias ✅")

    return X_scaled

# ════════════════════════════════════════════════════════════
# MAIN PIPELINE
# Urutan: Load → PCA → Elbow/Sil → K-Means → DBSCAN → Hier → Compare
# Semua clustering menggunakan X_pca.
# Semua profiling menggunakan X original.
# ════════════════════════════════════════════════════════════

def run_clustering(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, 'dataset_clustering.csv')

    print("=" * 55)
    print("  PHASE 2 — Segmentation via Clustering")
    print("=" * 55)

    print("\n[1/7] Load data...")
    df_cluster, X, y_true = load_clustering_data(path)

    # Jalankan ini sebelum apply_pca untuk investigasi
    print("=== Statistik X setelah scaling ===")
    print(X.describe().round(3))
    print("\n=== Variance per fitur ===")
    print(X.var().sort_values(ascending=False).round(4))

    print("\n[2/7] PCA — reduksi dimensi sebelum clustering...")
    X_prep = prepare_for_pca(X)         
    X_pca, pca_final, n_components = apply_pca(X_prep)

    print("\n[3/7] Elbow Method & Silhouette Score (di PCA space)...")
    best_k = find_optimal_k(X_pca)

    print(f"\n[4/7] K-Means (K={best_k}, di PCA space)...")
    df_cluster, X_tsne = run_kmeans(df_cluster, X, X_pca, pca_final, best_k)
    profile_clusters(df_cluster, X, 'KMeans_Cluster')

    print("\n[5/7] DBSCAN (auto-eps, di PCA space)...")
    df_cluster, n_clusters_db, n_noise, best_eps = run_dbscan(
        df_cluster, X, X_pca, pca_final
    )

    print(f"\n[6/7] Hierarchical (Ward, K={best_k}, di PCA space)...")
    df_cluster = run_hierarchical(df_cluster, X, X_pca, pca_final, best_k)
    profile_clusters(df_cluster, X, 'Hierarchical_Cluster')

    print("\n[7/7] Perbandingan ketiga metode...")
    compare_methods(df_cluster, X, X_pca, n_clusters_db, n_noise)

    print("\n" + "=" * 55)
    print("  PHASE 2 SELESAI")
    print("=" * 55)
    return df_cluster


if __name__ == '__main__':
    run_clustering()