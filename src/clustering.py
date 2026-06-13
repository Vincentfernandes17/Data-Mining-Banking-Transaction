"""
clustering.py — Phase 2: Segmentation via Clustering

Tujuan: menemukan kelompok nasabah (unsupervised) yang bermakna secara bisnis.

KEPUTUSAN METODOLOGIS PENTING (berdasarkan temuan Phase 1)
─────────────────────────────────────────────────────────
Di Phase 1 terbukti bahwa fitur kontinu MENTAH pada dataset ini saling
INDEPENDEN dan terdistribusi mendekati uniform (skew ~0, hanya 2 dari 55
pasang yang berkorelasi, itupun pasangan turunan). Dua konsekuensi:

1. PCA TIDAK DIPAKAI sebagai reduksi sebelum clustering.
   Tanpa redundansi antar fitur, setiap komponen PCA hanya menangkap ~1/n
   variance (scree plot datar) sehingga reduksi 11→9 fitur tidak ada gunanya.
   Kita TETAP menjalankan PCA sekali — bukan untuk dipakai, tapi untuk
   MEMBUKTIKAN dan mendokumentasikan kenapa ia tidak efektif (fungsi
   dimensionality_analysis di bawah).

2. Clustering dilakukan pada 3 RASIO PERILAKU hasil feature engineering:
       - CC_Utilization              : tekanan kartu kredit (saldo/limit)
       - Transaction_to_Balance_Ratio: intensitas likuiditas (transaksi/saldo)
       - Loan_to_Balance_Ratio       : leverage utang (pinjaman/saldo)
   Rasio dari dua fitur uniform menghasilkan distribusi BERSTRUKTUR (skew 2–7),
   sehingga cluster yang terbentuk benar-benar memisahkan perilaku nasabah.
   Hasil: Silhouette naik dari ~0.07 (fitur mentah) → ~0.57 (rasio perilaku).

Pipeline: Load → Dimensionality Analysis → Winsorize+Scale → Elbow/Silhouette
          → K-Means → DBSCAN → Hierarchical → Compare → Named Profiles → Save

Cara running:
    python src/clustering.py
    from src.clustering import run_clustering
    df = run_clustering('data/dataset_clustering.csv')
"""

import os
import sys
for _stream in (sys.stdout, sys.stderr):   # konsol Windows cp1252 → paksa UTF-8
    try:
        _stream.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # backend non-interaktif → plot disimpan ke file
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats.mstats import winsorize

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'phase2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Parameter ───────────────────────────────────────────────
RANDOM_STATE   = 42
N_INIT         = 10
K_RANGE        = range(2, 11)
BEST_K         = 3       # dipilih atas dasar domain (justifikasi di find_optimal_k)
WINSOR_LIMIT   = 0.02    # cap 2% ekor atas agar cluster digerakkan perilaku, bukan 1-2 outlier
DBSCAN_MIN_SAMPLES = 10

# 3 fitur INPUT clustering (rasio perilaku). Sisanya kolom konteks/profiling.
CLUSTER_FEATURES = [
    'CC_Utilization',
    'Transaction_to_Balance_Ratio',
    'Loan_to_Balance_Ratio',
]

# Kolom konteks yang dipakai saat profiling (nilai ASLI, bukan input jarak)
PROFILE_CONTEXT = [
    'Age', 'Account Balance', 'Credit Limit', 'Credit Card Balance',
    'Loan Amount', 'Transaction Amount', 'Interest Rate', 'Rewards Points',
    'Account_Age_Years', 'Days_Since_Last_Transaction',
    'Account Balance After Transaction',
]


# ════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════
def load_clustering_data(path):
    df = pd.read_csv(path)
    feats = [c for c in CLUSTER_FEATURES if c in df.columns]
    if len(feats) < len(CLUSTER_FEATURES):
        missing = set(CLUSTER_FEATURES) - set(feats)
        raise ValueError(f"Fitur clustering hilang dari dataset: {missing}. "
                         f"Jalankan ulang Phase 1.")
    print(f"Shape dataset        : {df.shape}")
    print(f"Fitur INPUT clustering: {feats}")
    print(f"\nStatistik 3 rasio perilaku (nilai asli):")
    print(df[feats].describe().round(3).to_string())
    print(f"\nSkewness (struktur):  "
          + ", ".join(f"{c}={df[c].skew():.2f}" for c in feats))
    return df, feats


# ════════════════════════════════════════════════════════════
# DIMENSIONALITY ANALYSIS — BUKTI PCA TIDAK EFEKTIF
# Bukan reduksi sungguhan. Kita jalankan PCA pada fitur kontinu mentah
# hanya untuk MEMBUKTIKAN scree plot-nya datar (~1/n per komponen),
# yang menjelaskan kenapa reduksi 11→9 tidak berguna di dataset ini.
# ════════════════════════════════════════════════════════════
def dimensionality_analysis(df, save_plots=True):
    raw_cont = [c for c in [
        'Age', 'Account Balance', 'Transaction Amount', 'Loan Amount',
        'Interest Rate', 'Loan Term', 'Credit Limit', 'Credit Card Balance',
        'Rewards Points', 'Account_Age_Years', 'Days_Since_Last_Transaction',
    ] if c in df.columns]

    X = StandardScaler().fit_transform(df[raw_cont])
    pca = PCA(random_state=RANDOM_STATE).fit(X)
    evr = pca.explained_variance_ratio_
    cum = evr.cumsum()
    n80 = int(np.argmax(cum >= 0.80)) + 1

    print("\n=== DIMENSIONALITY ANALYSIS (kenapa PCA TIDAK dipakai) ===")
    print(f"Jumlah fitur kontinu        : {len(raw_cont)}")
    for i, (r, c) in enumerate(zip(evr, cum), 1):
        print(f"  PC{i:>2}: {r*100:5.1f}%   (cumulative {c*100:5.1f}%)")
    print(f"Komponen utk 80% variance   : {n80} dari {len(raw_cont)} "
          f"→ kompresi hanya {(1-n80/len(raw_cont))*100:.0f}%")
    print("Setiap PC menangkap ~1/n variance → fitur INDEPENDEN, tidak ada "
          "redundansi untuk dikompres.")
    print("KEPUTUSAN: PCA dibuang. Clustering pakai 3 rasio perilaku berstruktur.")

    plt.figure(figsize=(10, 4))
    ideal = np.full(len(evr), 1 / len(evr))
    plt.bar(range(1, len(evr) + 1), evr * 100, alpha=0.7,
            label='Variance aktual per PC')
    plt.plot(range(1, len(evr) + 1), ideal * 100, 'r--o',
             label=f'Garis "independen total" (1/n = {100/len(evr):.1f}%)')
    plt.xlabel('Komponen PCA'); plt.ylabel('Variance Explained (%)')
    plt.title('Scree Plot DATAR = Fitur Independen → PCA Tidak Efektif')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'pca_why_not_used.png'), dpi=150)
    plt.close()


# ════════════════════════════════════════════════════════════
# PREPARE — WINSORIZE + STANDARD SCALE (di ruang fitur rasio)
# Winsorization meng-cap ekor atas (2%) agar cluster digerakkan oleh
# perbedaan perilaku yang nyata, bukan segelintir outlier ekstrem.
# Outlier ekstrem TIDAK dihapus — disimpan untuk diinvestigasi di Phase 4.
# ════════════════════════════════════════════════════════════
def prepare_features(df, feats, winsor_limit=WINSOR_LIMIT):
    Xw = df[feats].copy()
    print(f"\n=== Winsorization (cap {winsor_limit*100:.0f}% ekor atas) ===")
    for c in feats:
        upper = np.percentile(Xw[c], 100 * (1 - winsor_limit))
        n_cap = (Xw[c] > upper).sum()
        Xw[c] = winsorize(Xw[c], limits=[0, winsor_limit])
        print(f"  {c:32s}: {n_cap} nilai di-cap pada {upper:.3f}")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(Xw),
                            columns=feats, index=df.index)
    print("Scaling selesai (mean≈0, std≈1 per fitur).")
    return X_scaled


# ════════════════════════════════════════════════════════════
# ELBOW METHOD & SILHOUETTE SCORE
# ════════════════════════════════════════════════════════════
def find_optimal_k(X_scaled, k_range=K_RANGE, save_plots=True):
    wcss, sil = [], []
    print("\n=== Elbow Method & Silhouette Score ===")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = km.fit_predict(X_scaled)
        wcss.append(km.inertia_)
        sil.append(silhouette_score(X_scaled, labels))
        print(f"  K={k:>2}: WCSS={km.inertia_:10.1f}  Silhouette={sil[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(list(k_range), wcss, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Jumlah Cluster (K)'); axes[0].set_ylabel('WCSS')
    axes[0].set_title('Elbow Method'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(list(k_range), sil, 'rs-', linewidth=2, markersize=8)
    axes[1].axvline(BEST_K, color='green', linestyle='--', alpha=0.7,
                    label=f'K dipilih = {BEST_K}')
    axes[1].set_xlabel('Jumlah Cluster (K)'); axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score per K'); axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'elbow_silhouette.png'), dpi=150)
    plt.close()

    math_best = list(k_range)[int(np.argmax(sil))]
    print(f"\nK terbaik (silhouette murni) : {math_best} (sil={max(sil):.3f})")
    print(f"K yang dipilih (domain)      : {BEST_K} (sil={sil[list(k_range).index(BEST_K)]:.3f})")
    print("Justifikasi: K=2 hanya memisahkan segmen over-limit vs sisanya. "
          "K=3 tetap memiliki silhouette tinggi TAPI memunculkan segmen ketiga "
          "(liquidity-stressed/high-leverage) yang lebih actionable secara bisnis.")
    return BEST_K, sil


# ════════════════════════════════════════════════════════════
# SEGMENT NAMING — data-driven, tidak hardcode ke id cluster
# ════════════════════════════════════════════════════════════
def name_segments(df, feats, cluster_col):
    """Beri nama tiap cluster berdasarkan rasio mana yang paling menonjol
    dibanding median global. Label cluster K-Means itu acak, jadi penamaan
    harus berbasis profil, bukan nomor cluster."""
    ref = df[feats].median()
    names = {}
    for cid, g in df.groupby(cluster_col):
        med = g[feats].median()
        cc, txn, loan = (med.get('CC_Utilization', 0),
                         med.get('Transaction_to_Balance_Ratio', 0),
                         med.get('Loan_to_Balance_Ratio', 0))
        tags = []
        if cc >= 1.0:
            tags.append('Over-Limit Credit')
        elif cc >= 0.7:
            tags.append('High Credit-Util')
        if txn >= 2 * ref['Transaction_to_Balance_Ratio']:
            tags.append('High Txn-Intensity')
        if loan >= 2 * ref['Loan_to_Balance_Ratio']:
            tags.append('High Leverage')
        if not tags:
            names[cid] = 'Mainstream / Balanced'
        elif {'High Txn-Intensity', 'High Leverage'} & set(tags):
            names[cid] = 'Liquidity-Stressed / High-Leverage'
        elif 'Over-Limit Credit' in tags or 'High Credit-Util' in tags:
            names[cid] = 'Credit-Stressed / Over-Limit'
        else:
            names[cid] = ' + '.join(tags)
    return names


# ════════════════════════════════════════════════════════════
# K-MEANS
# ════════════════════════════════════════════════════════════
def run_kmeans(df, X_scaled, feats, best_k, save_plots=True):
    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=N_INIT)
    df['KMeans_Cluster'] = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, df['KMeans_Cluster'])

    print(f"\n=== K-Means (K={best_k}) — Silhouette={sil:.4f} ===")
    print(df['KMeans_Cluster'].value_counts().sort_index().to_string())

    # ── Scatter pada pasangan fitur rasio (standardized, interpretable) ──
    pairs = [(0, 1), (1, 2)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (i, j) in zip(axes, pairs):
        sc = ax.scatter(X_scaled.iloc[:, i], X_scaled.iloc[:, j],
                        c=df['KMeans_Cluster'], cmap='tab10', alpha=0.6, s=20)
        ax.set_xlabel(f'{feats[i]} (z)'); ax.set_ylabel(f'{feats[j]} (z)')
        ax.set_title(f'{feats[i]} vs {feats[j]}')
        plt.colorbar(sc, ax=ax, label='Cluster')
    plt.suptitle(f'K-Means (K={best_k}) — Ruang Fitur Rasio Perilaku',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'kmeans_scatter.png'), dpi=150)
    plt.close()

    # ── t-SNE 2D (embedding untuk visualisasi) ──
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30,
                init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_scaled)
    plt.figure(figsize=(9, 6))
    sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['KMeans_Cluster'],
                     cmap='Dark2', alpha=0.8, s=20)
    plt.title(f'K-Means (K={best_k}) — Proyeksi t-SNE')
    plt.xlabel('Dim 1'); plt.ylabel('Dim 2')
    plt.legend(*sc.legend_elements(), title='Cluster')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'kmeans_tsne.png'), dpi=150)
    plt.close()
    return df, sil


# ════════════════════════════════════════════════════════════
# CLUSTER PROFILING — nilai ASLI + NAMA segmen
# ════════════════════════════════════════════════════════════
def profile_clusters(df, feats, cluster_col, save_plots=True):
    names = name_segments(df, feats, cluster_col)
    ctx = [c for c in PROFILE_CONTEXT if c in df.columns]

    # Median lebih robust daripada mean untuk fitur skewed
    prof = df.groupby(cluster_col)[feats + ctx].median().round(2)
    prof['n'] = df.groupby(cluster_col).size()
    prof['pct'] = (prof['n'] / len(df) * 100).round(1)
    prof['Segment'] = [names[c] for c in prof.index]

    print(f"\n{'='*60}")
    print(f"  CLUSTER PROFILE — {cluster_col} (median, nilai asli)")
    print(f"{'='*60}")
    show = feats + ['Account Balance', 'Credit Limit', 'Loan Amount',
                    'Age', 'Rewards Points', 'n', 'pct']
    show = [c for c in show if c in prof.columns]
    disp = prof[show].copy()
    disp.insert(0, 'Segment', prof['Segment'])
    print(disp.T.to_string())

    # Cross-check Anomaly (insight, bukan input model)
    if 'Anomaly' in df.columns:
        print(f"\nAnomaly rate per {cluster_col}:")
        for cid, g in df.groupby(cluster_col):
            print(f"  Cluster {cid} ({names[cid]:38s}): "
                  f"{(g['Anomaly']==-1).mean()*100:4.1f}% "
                  f"({(g['Anomaly']==-1).sum()} dari {len(g)})")

    # Bar chart: median 3 rasio per cluster (skala log karena rentang lebar)
    cids = sorted(df[cluster_col].unique())
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(cids)), 5))
    x = np.arange(len(cids)); w = 0.25
    for k, f in enumerate(feats):
        ax.bar(x + k * w, [prof.loc[c, f] for c in cids], w, label=f)
    ax.set_yscale('log')
    ax.set_xticks(x + w)
    ax.set_xticklabels([f"C{c}\n{names[c]}" for c in cids], fontsize=8)
    ax.set_ylabel('Median rasio (skala log)')
    ax.set_title(f'Profil Rasio Perilaku per Segmen — {cluster_col}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_plots:
        fname = f"profile_{cluster_col.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.close()
    return prof, names


# ════════════════════════════════════════════════════════════
# DBSCAN — density-based, otomatis mengisolasi outlier sebagai NOISE.
# Noise points = nasabah dengan perilaku ekstrem → diumpankan ke Phase 4.
# eps dicari otomatis lewat knee k-distance + target noise 2–15%.
# ════════════════════════════════════════════════════════════
def run_dbscan(df, X_scaled, feats, min_samples=DBSCAN_MIN_SAMPLES, save_plots=True):
    Xv = X_scaled.values
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(Xv)
    dist, _ = nbrs.kneighbors(Xv)
    kdist = np.sort(dist[:, -1])

    # Auto-search eps: pilih yang menghasilkan noise paling dekat ke ~5%
    print("\n=== DBSCAN — pencarian eps otomatis ===")
    candidates = np.round(np.arange(0.20, 2.01, 0.05), 2)
    rows = []
    for eps in candidates:
        lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xv)
        nc = len(set(lab)) - (1 if -1 in lab else 0)
        nn = int((lab == -1).sum())
        rows.append((eps, nc, nn, nn / len(lab) * 100))
    res = pd.DataFrame(rows, columns=['eps', 'n_clusters', 'n_noise', 'pct_noise'])

    good = res[(res['pct_noise'] >= 2) & (res['pct_noise'] <= 15) & (res['n_clusters'] >= 1)]
    pick = good if not good.empty else res
    best_eps = float(pick.iloc[(pick['pct_noise'] - 5).abs().argsort().iloc[0]]['eps'])

    db = DBSCAN(eps=best_eps, min_samples=min_samples)
    df['DBSCAN_Cluster'] = db.fit_predict(Xv)
    n_clusters = len(set(df['DBSCAN_Cluster'])) - (1 if -1 in df['DBSCAN_Cluster'].values else 0)
    n_noise = int((df['DBSCAN_Cluster'] == -1).sum())
    print(res[(res['eps'] * 100 % 25 == 0)].to_string(index=False))
    print(f"\neps terpilih={best_eps}  → {n_clusters} cluster + "
          f"{n_noise} noise ({n_noise/len(df)*100:.1f}%)")

    # k-distance plot dengan eps terpilih
    plt.figure(figsize=(10, 4))
    plt.plot(kdist, linewidth=1.5)
    plt.axhline(best_eps, color='red', linestyle='--', label=f'eps={best_eps}')
    plt.xlabel('Data points (sorted)'); plt.ylabel(f'{min_samples}-NN distance')
    plt.title('K-Distance Graph — penentuan eps DBSCAN')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'dbscan_kdistance.png'), dpi=150)
    plt.close()

    # Scatter noise vs cluster
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (i, j) in zip(axes, [(0, 1), (1, 2)]):
        for lab in sorted(df['DBSCAN_Cluster'].unique()):
            m = df['DBSCAN_Cluster'] == lab
            is_noise = (lab == -1)
            ax.scatter(X_scaled.iloc[m.values, i], X_scaled.iloc[m.values, j],
                       label='Noise' if is_noise else f'Cluster {lab}',
                       marker='x' if is_noise else 'o',
                       s=50 if is_noise else 18,
                       alpha=0.9 if is_noise else 0.5,
                       c='red' if is_noise else None)
        ax.set_xlabel(f'{feats[i]} (z)'); ax.set_ylabel(f'{feats[j]} (z)')
        ax.legend(fontsize=8)
    plt.suptitle(f'DBSCAN (eps={best_eps}) — Noise = perilaku ekstrem '
                 f'({n_noise} titik, {n_noise/len(df)*100:.1f}%)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'dbscan_scatter.png'), dpi=150)
    plt.close()

    # Profil noise (nilai asli)
    if n_noise > 0:
        print("\n=== Profil DBSCAN Noise (median, nilai asli) ===")
        noise = df[df['DBSCAN_Cluster'] == -1]
        print(noise[feats].median().round(3).to_string())
        if 'Anomaly' in df.columns:
            print(f"Anomaly rate di noise   : {(noise['Anomaly']==-1).mean()*100:.1f}%")
            print(f"Anomaly rate di non-noise: "
                  f"{(df[df['DBSCAN_Cluster']!=-1]['Anomaly']==-1).mean()*100:.1f}%")
    return df, n_clusters, n_noise, best_eps


# ════════════════════════════════════════════════════════════
# HIERARCHICAL — 3 linkage dibandingkan, potong Ward pada K.
# Pada n=5000 dendrogram penuh terlalu berat → sampel 1000 baris.
# ════════════════════════════════════════════════════════════
def run_hierarchical(df, X_scaled, best_k, save_plots=True):
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_scaled), size=min(1000, len(X_scaled)), replace=False)
    Xs = X_scaled.values[idx]

    methods = ['ward', 'complete', 'average']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, m in zip(axes, methods):
        Z = linkage(Xs, method=m)
        dendrogram(Z, ax=ax, truncate_mode='level', p=5,
                   color_threshold=0.7 * max(Z[:, 2]))
        ax.set_title(f'Dendrogram — {m.capitalize()} (sampel 1000)')
        ax.set_xlabel('Data points'); ax.set_ylabel('Distance')
    plt.suptitle('Hierarchical Clustering — 3 Linkage Methods',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'dendrogram_comparison.png'), dpi=150)
    plt.close()

    # Ward pada seluruh data untuk label final
    df['Hierarchical_Cluster'] = AgglomerativeClustering(
        n_clusters=best_k, linkage='ward').fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, df['Hierarchical_Cluster'])
    print(f"\n=== Hierarchical (Ward, K={best_k}) — Silhouette={sil:.4f} ===")
    print(df['Hierarchical_Cluster'].value_counts().sort_index().to_string())
    print("Catatan linkage: Ward menghasilkan cluster paling seimbang; "
          "Complete/Average lebih sensitif outlier (cabang panjang terpisah).")
    return df, sil


# ════════════════════════════════════════════════════════════
# COMPARISON — silhouette dihitung di ruang yang SAMA (X_scaled)
# ════════════════════════════════════════════════════════════
def compare_methods(df, X_scaled, sil_km, sil_hier, n_clusters_db, n_noise):
    mask = df['DBSCAN_Cluster'] != -1
    if df.loc[mask, 'DBSCAN_Cluster'].nunique() > 1:
        sil_db = round(silhouette_score(X_scaled[mask.values],
                                        df.loc[mask, 'DBSCAN_Cluster']), 4)
    else:
        sil_db = 'N/A (1 cluster inti + noise)'

    comp = pd.DataFrame({
        'Method':           ['K-Means', 'DBSCAN', 'Hierarchical (Ward)'],
        'N Clusters':       [df['KMeans_Cluster'].nunique(), n_clusters_db,
                             df['Hierarchical_Cluster'].nunique()],
        'Noise Points':     [0, n_noise, 0],
        'Silhouette':       [round(sil_km, 4), sil_db, round(sil_hier, 4)],
    })
    print(f"\n{'='*60}")
    print("  PERBANDINGAN METODE (silhouette di ruang fitur yang sama)")
    print(f"{'='*60}")
    print(comp.to_string(index=False))
    print("\nKesimpulan: K-Means & Hierarchical (Ward) memberi segmen seimbang & "
          "interpretable. DBSCAN unggul untuk MEMISAHKAN outlier perilaku "
          "(noise) → dipakai sebagai sinyal awal Phase 4.")
    return comp


# ════════════════════════════════════════════════════════════
# SAVE — dataset berlabel cluster untuk Phase 4
# ════════════════════════════════════════════════════════════
def save_clustered(df, names):
    df = df.copy()
    df['KMeans_Segment'] = df['KMeans_Cluster'].map(names)
    out = os.path.join(DATA_DIR, 'dataset_clustered.csv')
    df.to_csv(out, index=False)
    print(f"\n✅ Dataset berlabel cluster tersimpan → {out}")
    print(f"   Kolom label: KMeans_Cluster, KMeans_Segment, DBSCAN_Cluster, "
          f"Hierarchical_Cluster")
    return out


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def run_clustering(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, 'dataset_clustering.csv')

    print("=" * 55)
    print("  PHASE 2 — Segmentation via Clustering")
    print("=" * 55)

    print("\n[1/7] Load data...")
    df, feats = load_clustering_data(path)

    print("\n[2/7] Dimensionality analysis (bukti PCA tidak dipakai)...")
    dimensionality_analysis(df)

    print("\n[3/7] Winsorize + scale fitur rasio...")
    X_scaled = prepare_features(df, feats)

    print("\n[4/7] Elbow & Silhouette...")
    best_k, _ = find_optimal_k(X_scaled)

    print(f"\n[5/7] K-Means (K={best_k})...")
    df, sil_km = run_kmeans(df, X_scaled, feats, best_k)
    _, names = profile_clusters(df, feats, 'KMeans_Cluster')

    print("\n[6/7] DBSCAN + Hierarchical...")
    df, n_clusters_db, n_noise, _ = run_dbscan(df, X_scaled, feats)
    df, sil_hier = run_hierarchical(df, X_scaled, best_k)
    profile_clusters(df, feats, 'Hierarchical_Cluster')

    print("\n[7/7] Perbandingan & simpan...")
    compare_methods(df, X_scaled, sil_km, sil_hier, n_clusters_db, n_noise)
    save_clustered(df, names)

    print("\n" + "=" * 55)
    print("  PHASE 2 SELESAI")
    print("=" * 55)
    return df


if __name__ == '__main__':
    run_clustering()
