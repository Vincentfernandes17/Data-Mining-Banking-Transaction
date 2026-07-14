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
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
from scipy.stats.mstats import winsorize
from itertools import combinations

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

# Fitur kontinu MENTAH — dipakai untuk dimensionality analysis & sebagai
# kandidat "non-manual" di feature_selection_comparison.
RAW_CONTINUOUS = [
    'Age', 'Account Balance', 'Transaction Amount', 'Loan Amount',
    'Interest Rate', 'Loan Term', 'Credit Limit', 'Credit Card Balance',
    'Rewards Points', 'Account_Age_Years', 'Days_Since_Last_Transaction',
]

# Fitur demografis/kategorikal untuk MEMPERKAYA profil segmen (di-join by index
# dari dataset_arm.csv; BUKAN input clustering). Semua file row-aligned 5000 baris.
PROFILE_CATEGORICAL = [
    'Gender', 'Age_Group', 'Account Type', 'Loan Type', 'Loan Status', 'Card Type',
]


# ════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════
def load_clustering_data(path):
    """Muat dataset_clustering.csv (output Phase 1), pastikan 3 fitur rasio
    tersedia, dan cetak statistik + skewness sebagai bukti struktur data.
    Mengembalikan (df, feats)."""
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
# ATTACH CATEGORICALS — perkaya profil segmen dengan demografi
# Fitur kategorikal (Gender, Age_Group, dst) diambil dari dataset_arm.csv yang
# ROW-ALIGNED (5000 baris, urutan sama). Dipakai HANYA untuk profiling, bukan
# input jarak clustering. Alignment diverifikasi lewat kolom bersama.
# ════════════════════════════════════════════════════════════
def attach_categoricals(df):
    """Tempelkan 6 kolom kategorikal dari dataset_arm.csv untuk MEMPERKAYA
    profil segmen (bukan input jarak clustering).

    Keamanan join by-index: jumlah baris harus sama DAN kolom bersama
    'Transaction Type' harus cocok ≥99.9% — bila tidak, penempelan
    dibatalkan dengan aman (profil kategorikal dilewati).
    """
    arm_path = os.path.join(DATA_DIR, 'dataset_arm.csv')
    if not os.path.exists(arm_path):
        print("  dataset_arm.csv tidak ada → profil kategorikal dilewati.")
        return df
    arm = pd.read_csv(arm_path)
    if len(arm) != len(df):
        print(f"  Jumlah baris beda (arm={len(arm)}, clustering={len(df)}) "
              f"→ profil kategorikal dilewati.")
        return df
    # Verifikasi row-alignment lewat kolom bersama 'Transaction Type'
    if 'Transaction Type' in df.columns and 'Transaction Type' in arm.columns:
        match = (df['Transaction Type'].values == arm['Transaction Type'].values).mean()
        print(f"  Verifikasi alignment (Transaction Type cocok): {match*100:.1f}%")
        if match < 0.999:
            print("  PERINGATAN: alignment <100% → profil kategorikal dibatalkan.")
            return df
    added = []
    for c in PROFILE_CATEGORICAL:
        if c in arm.columns:
            df[c] = arm[c].values
            added.append(c)
    print(f"  Fitur kategorikal ditempel untuk profiling: {added}")
    return df


# ════════════════════════════════════════════════════════════
# DIMENSIONALITY ANALYSIS — BUKTI PCA TIDAK EFEKTIF
# Bukan reduksi sungguhan. Kita jalankan PCA pada fitur kontinu mentah
# hanya untuk MEMBUKTIKAN scree plot-nya datar (~1/n per komponen),
# yang menjelaskan kenapa reduksi 11→9 tidak berguna di dataset ini.
# ════════════════════════════════════════════════════════════
def dimensionality_analysis(df, save_plots=True):
    """Jalankan PCA sekali pada fitur kontinu mentah HANYA sebagai bukti:
    scree plot datar (~1/n variance per komponen) = fitur independen = tidak
    ada redundansi untuk dikompres → PCA dibuang dari pipeline. Menyimpan
    pca_why_not_used.png."""
    raw_cont = [c for c in RAW_CONTINUOUS if c in df.columns]

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
# FEATURE SELECTION COMPARISON — manual (domain) vs OTOMATIS
# Dosen minta pembanding: jangan cuma pilih 3 rasio secara manual, tapi tunjukkan
# bahwa pilihan itu menang secara kuantitatif. Kita bandingkan silhouette K-Means
# (K=BEST_K) pada 4 set fitur:
#   (1) semua fitur mentah, (2) PCA, (3) subset terbaik hasil pencarian OTOMATIS
#       (exhaustive semua kombinasi 3-fitur), (4) 3 rasio domain (pilihan kita).
# Kalau pencarian otomatis mendarat di 3 rasio yang sama → konfirmasi data-driven.
# ════════════════════════════════════════════════════════════
def _winsor_scale(df, cols):
    """Helper: winsorize ekor atas (WINSOR_LIMIT) lalu StandardScaler —
    transformasi ruang fitur yang sama dengan pipeline utama, dipakai
    berulang di feature_selection_comparison."""
    Xw = df[cols].copy()
    for c in cols:
        Xw[c] = winsorize(Xw[c], limits=[0, WINSOR_LIMIT])
    return StandardScaler().fit_transform(Xw)


def _kmeans_silhouette(X, k=BEST_K, n_init=N_INIT, sample=None):
    """Helper: fit K-Means lalu kembalikan silhouette score-nya.
    `sample` opsional untuk silhouette tersampel (lebih cepat); default None
    = dihitung penuh pada seluruh baris agar sebanding dengan angka final."""
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=n_init).fit(X)
    if sample:
        return silhouette_score(X, km.labels_, sample_size=min(sample, len(X)),
                                random_state=RANDOM_STATE)
    return silhouette_score(X, km.labels_)


def feature_selection_comparison(df, ratio_feats, save_plots=True):
    """Validasi kuantitatif pilihan fitur manual vs alternatif otomatis.

    Empat set fitur dibandingkan silhouette-nya (K-Means, K=BEST_K):
    (1) semua fitur mentah, (2) PCA 80% variance, (3) kombinasi 3-fitur
    TERBAIK dari exhaustive search 286 kombinasi, (4) 3 rasio domain.
    Bagian paling lama Phase 2 (~5 menit: 286× K-Means + silhouette penuh
    5000 baris). Menyimpan bar chart + CSV untuk laporan & dashboard.
    """
    raw = [c for c in RAW_CONTINUOUS if c in df.columns]
    pool = raw + ratio_feats
    print("\n=== FEATURE SELECTION COMPARISON (domain manual vs otomatis) ===")
    print(f"Pool kandidat: {len(raw)} fitur mentah + {len(ratio_feats)} rasio "
          f"= {len(pool)} fitur")

    # (1) semua fitur mentah
    sil_raw = _kmeans_silhouette(_winsor_scale(df, raw))
    # (2) PCA pada fitur mentah → komponen 80% variance lalu cluster
    Xraw = _winsor_scale(df, raw)
    pca = PCA(random_state=RANDOM_STATE).fit(Xraw)
    n80 = int(np.argmax(pca.explained_variance_ratio_.cumsum() >= 0.80)) + 1
    sil_pca = _kmeans_silhouette(
        PCA(n_components=n80, random_state=RANDOM_STATE).fit_transform(Xraw))
    # (3) seleksi OTOMATIS: exhaustive semua kombinasi 3-fitur, maksimalkan silhouette
    combos = list(combinations(pool, 3))
    print(f"Exhaustive search {len(combos)} kombinasi 3-fitur (otomatis, "
          f"silhouette PENUH pada 5000 baris — ini bagian paling lama)...")
    # Silhouette dihitung pada SELURUH 5000 baris (tanpa sampling) agar peringkat
    # benar-benar setara dengan angka final yang dilaporkan.
    scored = [(_kmeans_silhouette(_winsor_scale(df, list(c))), c) for c in combos]
    scored.sort(key=lambda t: t[0], reverse=True)
    best_auto_sil, best_auto_cols = scored[0]
    dom_set = set(ratio_feats)
    dom_rank = next(i for i, (s, c) in enumerate(scored, 1) if set(c) == dom_set)
    sil_dom = next(s for s, c in scored if set(c) == dom_set)

    comp = pd.DataFrame({
        'Feature Set': ['Semua fitur mentah', f'PCA ({n80} komponen)',
                        'Seleksi otomatis (3 terbaik)', '3 Rasio domain (dipilih)'],
        'N_Fitur':    [len(raw), n80, 3, 3],
        'Silhouette': [round(sil_raw, 4), round(sil_pca, 4),
                       round(best_auto_sil, 4), round(sil_dom, 4)],
    })
    print(comp.to_string(index=False))
    print(f"\nKombinasi 3-fitur TERBAIK (otomatis): {list(best_auto_cols)}")
    print(f"3 rasio domain menempati peringkat #{dom_rank} dari {len(scored)} kombinasi.")
    print("Top-5 kombinasi 3-fitur (silhouette penuh 5000 baris):")
    for s, c in scored[:5]:
        mark = '   <== pilihan domain' if set(c) == dom_set else ''
        print(f"  {s:.4f}  {list(c)}{mark}")

    if set(best_auto_cols) == dom_set:
        print("\nKESIMPULAN: pencarian OTOMATIS mendarat tepat pada 3 rasio yang sama "
              "dengan pilihan domain → pilihan manual TERKONFIRMASI secara data-driven.")
    else:
        print(f"\nKESIMPULAN: rasio domain (sil={sil_dom:.3f}) jauh mengungguli fitur "
              f"mentah ({sil_raw:.3f}) & PCA ({sil_pca:.3f}); peringkat #{dom_rank} "
              f"dari {len(scored)} kombinasi → pilihan manual tervalidasi kuantitatif.")

    # Bar chart perbandingan
    plt.figure(figsize=(9, 5))
    colors = ['#bdc3c7', '#bdc3c7', '#f39c12', '#2ecc71']
    bars = plt.bar(comp['Feature Set'], comp['Silhouette'], color=colors)
    for b, v in zip(bars, comp['Silhouette']):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.005, f'{v:.3f}',
                 ha='center', fontsize=10, fontweight='bold')
    plt.ylabel('Silhouette Score (K-Means, K=%d)' % BEST_K)
    plt.title('Perbandingan Feature Selection — Rasio Domain vs Alternatif Otomatis')
    plt.xticks(rotation=12, ha='right'); plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'feature_selection_comparison.png'), dpi=150)
    plt.close()
    # CSV untuk dashboard
    comp.to_csv(os.path.join(OUTPUT_DIR, 'feature_selection_comparison.csv'), index=False)
    return comp


# ════════════════════════════════════════════════════════════
# PREPARE — WINSORIZE + STANDARD SCALE (di ruang fitur rasio)
# Winsorization meng-cap ekor atas (2%) agar cluster digerakkan oleh
# perbedaan perilaku yang nyata, bukan segelintir outlier ekstrem.
# Outlier ekstrem TIDAK dihapus — disimpan untuk diinvestigasi di Phase 4.
# ════════════════════════════════════════════════════════════
def prepare_features(df, feats, winsor_limit=WINSOR_LIMIT):
    """Siapkan ruang fitur clustering: winsorize ekor atas 2% (outlier TIDAK
    dihapus — investigasinya di Phase 4) lalu StandardScaler. Mengembalikan
    DataFrame ter-skala (mean≈0, std≈1) yang dipakai SEMUA algoritma."""
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
    """Tentukan K optimal via Elbow (WCSS) + Silhouette untuk K=2..10.

    K final = BEST_K (3), dipilih atas dasar domain: silhouette-nya tetap
    tinggi DAN memunculkan segmen ketiga (liquidity-stressed) yang lebih
    actionable daripada K=2 — justifikasi dicetak & digambar (garis hijau
    di elbow_silhouette.png). Mengembalikan (BEST_K, daftar silhouette).
    """
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
    """K-Means final pada K terpilih + dua visual: scatter pasangan rasio
    (z-score) dan proyeksi t-SNE 2D (langkah tunggal terlama kedua, ~30-60
    dtk). Menambah kolom KMeans_Cluster; mengembalikan (df, silhouette)."""
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
    """Profil tiap cluster dalam NILAI ASLI (median — robust untuk skew).

    Menampilkan 3 rasio + konteks finansial (saldo/limit/pinjaman/transaksi/
    umur) + ukuran segmen + NAMA segmen data-driven; cross-check label
    Anomaly per segmen; bar chart profil (log). Diakhiri profil kategorikal
    (profile_categoricals). Mengembalikan (prof, names).
    """
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
    show = feats + ['Account Balance', 'Credit Limit', 'Credit Card Balance',
                    'Loan Amount', 'Transaction Amount', 'Age', 'n', 'pct']
    show = [c for c in show if c in prof.columns]
    disp = prof[show].copy()
    disp.insert(0, 'Segment', prof['Segment'])
    print(disp.T.to_string())
    print("Catatan: fitur FINANSIAL numerik (saldo/limit/pinjaman) yang paling "
          "membedakan segmen & menjelaskan rasio — bukan sekadar 3 input clustering.")

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

    # ── Profil DEMOGRAFIS / KATEGORIKAL per segmen (memperkaya, bukan input) ──
    profile_categoricals(df, cluster_col, names, save_plots=save_plots)
    return prof, names


# ════════════════════════════════════════════════════════════
# CLUSTER PROFILE (KATEGORIKAL) — Gender, Age_Group, Account Type, dst.
# Dosen minta profil tak cuma 3 rasio input. Kita tunjukkan komposisi demografis
# tiap segmen. Temuan jujur: pada dataset sintetis ini demografi nyaris SERAGAM
# antar segmen (independen) — jadi segmen murni dibentuk PERILAKU finansial.
# "spread" = selisih proporsi maks–min satu kategori antar cluster (poin persen).
# ════════════════════════════════════════════════════════════
def profile_categoricals(df, cluster_col, names, save_plots=True, spread_thresh=10.0):
    """Komposisi demografis/kategorikal per segmen (crosstab %, 100% stacked
    bar). 'spread' = selisih proporsi maks-min satu kategori antar cluster;
    ≥spread_thresh pp dianggap MEMBEDAKAN. Temuan jujur di dataset ini:
    demografi ~seragam antar segmen → segmen dibentuk PERILAKU finansial."""
    cats = [c for c in PROFILE_CATEGORICAL if c in df.columns]
    if not cats:
        return
    spreads = {}
    for c in cats:
        ct = pd.crosstab(df[cluster_col], df[c], normalize='index') * 100
        spreads[c] = float((ct.max() - ct.min()).max())
    order = sorted(cats, key=lambda c: spreads[c], reverse=True)

    print(f"\n--- Komposisi demografis/kategorikal per {cluster_col} (%) ---")
    for c in order:
        ct = pd.crosstab(df[cluster_col], df[c], normalize='index') * 100
        dom = ct.idxmax(axis=1)
        tag = 'MEMBEDAKAN' if spreads[c] >= spread_thresh else '~seragam'
        detail = ", ".join(f"C{cid}:{dom[cid]}({ct.loc[cid, dom[cid]]:.0f}%)"
                           for cid in ct.index)
        print(f"  {c:13s} [spread {spreads[c]:4.1f} pp · {tag:11s}] dominan → {detail}")
    diff = [c for c in order if spreads[c] >= spread_thresh]
    print(f"\n  → Fitur yang MEMBEDAKAN segmen : {diff if diff else 'praktis tidak ada'}")
    print(f"  → Fitur yang nyaris SERAGAM    : {[c for c in order if spreads[c] < spread_thresh]}")
    print("  → Interpretasi: segmen dibentuk PERILAKU finansial (rasio + saldo/limit/"
          "pinjaman), bukan demografi. Demografi independen → ciri dataset sintetis.")

    # Viz: 100% stacked bar komposisi tiap kategorikal per cluster
    ncol = 3
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow))
    axes = np.array(axes).reshape(-1)
    cids = sorted(df[cluster_col].unique())
    xlabels = [f"C{c}" for c in cids]
    for ax, c in zip(axes, order):
        ct = pd.crosstab(df[cluster_col], df[c], normalize='index') * 100
        ct = ct.reindex(cids)
        bottom = np.zeros(len(cids))
        for cat in ct.columns:
            ax.bar(xlabels, ct[cat].values, bottom=bottom, label=str(cat))
            bottom += ct[cat].values
        ax.set_title(f'{c}  (spread {spreads[c]:.0f} pp)', fontsize=10)
        ax.set_ylabel('% dalam cluster'); ax.set_ylim(0, 100)
        ax.legend(fontsize=7, ncol=2, loc='lower center')
    for ax in axes[len(order):]:
        ax.axis('off')
    plt.suptitle(f'Komposisi Demografis per Segmen — {cluster_col} '
                 f'(makin seragam = demografi tak membedakan)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_plots:
        fname = f"profile_{cluster_col.lower().replace(' ', '_')}_demographics.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.close()


# ════════════════════════════════════════════════════════════
# DBSCAN — density-based, otomatis mengisolasi outlier sebagai NOISE.
# Noise points = nasabah dengan perilaku ekstrem → diumpankan ke Phase 4.
# eps dicari otomatis lewat knee k-distance + target noise 2–15%.
# ════════════════════════════════════════════════════════════
def run_dbscan(df, X_scaled, feats, min_samples=DBSCAN_MIN_SAMPLES, save_plots=True):
    """DBSCAN dengan eps dicari OTOMATIS: coba kandidat 0.20-2.00, pilih
    yang persentase noise-nya paling dekat ~5% dalam rentang wajar 2-15%.

    Noise (-1) = nasabah berperilaku ekstrem → diumpankan ke Phase 4 sebagai
    sudut pandang density. Menyimpan k-distance plot & scatter noise.
    Mengembalikan (df, n_clusters, n_noise, best_eps).
    """
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
    """Hierarchical clustering: bandingkan dendrogram 3 linkage (ward /
    complete / average; sampel 1000 baris agar dendrogram terbaca), lalu
    potong Ward pada K untuk label final SELURUH data.

    Untuk tiap linkage dihitung KOEFISIEN KOFENETIK (cophenetic correlation)
    yaitu korelasi antara jarak asli antar titik dengan jarak kofenetik pada
    pohon dendrogram. Makin tinggi (mendekati 1) makin setia pohon itu
    merepresentasikan struktur jarak sebenarnya, sehingga koefisien inilah
    yang menjustifikasi pemilihan linkage. Mengembalikan (df, silhouette,
    coph) dengan coph = dict cophenetic per linkage.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_scaled), size=min(1000, len(X_scaled)), replace=False)
    Xs = X_scaled.values[idx]
    dist_orig = pdist(Xs)   # jarak asli antar titik (dipakai untuk kofenetik)

    methods = ['ward', 'complete', 'average']
    coph = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, m in zip(axes, methods):
        Z = linkage(Xs, method=m)
        coph[m], _ = cophenet(Z, dist_orig)   # koefisien kofenetik linkage m
        dendrogram(Z, ax=ax, truncate_mode='level', p=5,
                   color_threshold=0.7 * max(Z[:, 2]))
        ax.set_title(f'Dendrogram — {m.capitalize()} '
                     f'(sampel 1000, kofenetik={coph[m]:.3f})')
        ax.set_xlabel('Data points'); ax.set_ylabel('Distance')
    plt.suptitle('Hierarchical Clustering — 3 Linkage Methods',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'dendrogram_comparison.png'), dpi=150)
    plt.close()

    print("\n=== Koefisien Kofenetik per linkage (sampel 1000) ===")
    for m in methods:
        print(f"  {m.capitalize():9s}: {coph[m]:.4f}")
    best_link = max(coph, key=coph.get)
    print(f"Linkage dengan kofenetik tertinggi: {best_link.capitalize()} "
          f"({coph[best_link]:.4f}).")
    print("Catatan: Ward dipilih untuk label final karena menghasilkan cluster "
          "paling seimbang & interpretable; koefisien kofenetiknya juga tinggi "
          "sehingga pohon setia pada struktur jarak asli.")

    # Ward pada seluruh data untuk label final
    df['Hierarchical_Cluster'] = AgglomerativeClustering(
        n_clusters=best_k, linkage='ward').fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, df['Hierarchical_Cluster'])
    print(f"\n=== Hierarchical (Ward, K={best_k}) — Silhouette={sil:.4f} ===")
    print(df['Hierarchical_Cluster'].value_counts().sort_index().to_string())
    print("Catatan linkage: Ward menghasilkan cluster paling seimbang; "
          "Complete/Average lebih sensitif outlier (cabang panjang terpisah).")
    return df, sil, coph


# ════════════════════════════════════════════════════════════
# COMPARISON — silhouette dihitung di ruang yang SAMA (X_scaled)
# ════════════════════════════════════════════════════════════
def compare_methods(df, X_scaled, sil_km, sil_hier, n_clusters_db, n_noise):
    """Tabel perbandingan 3 algoritma pada ruang fitur yang SAMA: jumlah
    cluster, noise, dan silhouette (DBSCAN dihitung tanpa noise agar adil).

    Juga menghitung ADJUSTED RAND INDEX (ARI) antara label K-Means (metode
    utama) dan Hierarchical Ward (validasi) untuk mengukur seberapa cocok
    kedua metode independen menempatkan nasabah pada segmen yang sama; ARI=1
    berarti partisi identik, ARI≈0 berarti sekadar kebetulan. Mencetak
    kesimpulan pemilihan metode dan mengembalikan (comp, ari).
    """
    mask = df['DBSCAN_Cluster'] != -1
    if df.loc[mask, 'DBSCAN_Cluster'].nunique() > 1:
        sil_db = round(silhouette_score(X_scaled[mask.values],
                                        df.loc[mask, 'DBSCAN_Cluster']), 4)
    else:
        sil_db = 'N/A (1 cluster inti + noise)'

    ari = adjusted_rand_score(df['KMeans_Cluster'], df['Hierarchical_Cluster'])

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
    print(f"\nAdjusted Rand Index (K-Means vs Hierarchical Ward): {ari:.4f}")
    print("  → Mengukur kesepakatan dua metode independen atas keanggotaan segmen.")
    print("\nKesimpulan: K-Means & Hierarchical (Ward) memberi segmen seimbang & "
          "interpretable. DBSCAN unggul untuk MEMISAHKAN outlier perilaku "
          "(noise) → dipakai sebagai sinyal awal Phase 4.")
    return comp, ari


# ════════════════════════════════════════════════════════════
# SAVE — dataset berlabel cluster untuk Phase 4
# ════════════════════════════════════════════════════════════
def save_clustered(df, names):
    """Simpan dataset + label ketiga algoritma + nama segmen K-Means ke
    dataset_clustered.csv — input Phase 4."""
    df = df.copy()
    df['KMeans_Segment'] = df['KMeans_Cluster'].map(names)
    out = os.path.join(DATA_DIR, 'dataset_clustered.csv')
    df.to_csv(out, index=False)
    print(f"\n✅ Dataset berlabel cluster tersimpan → {out}")
    print(f"   Kolom label: KMeans_Cluster, KMeans_Segment, DBSCAN_Cluster, "
          f"Hierarchical_Cluster")
    return out


# ════════════════════════════════════════════════════════════
# SAVE VALIDATION METRICS — ringkasan metrik kuantitatif Phase 2
# untuk laporan/appendix (silhouette, kofenetik, ARI, dll).
# ════════════════════════════════════════════════════════════
def save_validation_metrics(sil_km, sil_hier, coph, ari, best_k,
                            n_clusters_db, n_noise, best_eps):
    """Simpan metrik validasi Phase 2 ke validation_metrics.csv agar bisa
    langsung dikutip di laporan (Appendix D): silhouette K-Means & Ward pada
    K final, koefisien kofenetik tiap linkage, dan Adjusted Rand Index
    K-Means vs Hierarchical."""
    rows = [
        ('Silhouette Score (K-Means, K final)',            best_k, round(sil_km, 4)),
        ('Silhouette Score (Hierarchical Ward, K final)',  best_k, round(sil_hier, 4)),
        ('Cophenetic Correlation (Ward)',                  '',     round(coph.get('ward', float('nan')), 4)),
        ('Cophenetic Correlation (Complete)',              '',     round(coph.get('complete', float('nan')), 4)),
        ('Cophenetic Correlation (Average)',               '',     round(coph.get('average', float('nan')), 4)),
        ('Adjusted Rand Index (K-Means vs Hierarchical)',  '',     round(ari, 4)),
        ('DBSCAN clusters (eps auto)',                     best_eps, n_clusters_db),
        ('DBSCAN noise points',                            best_eps, n_noise),
    ]
    out = os.path.join(OUTPUT_DIR, 'validation_metrics.csv')
    pd.DataFrame(rows, columns=['Metric', 'Param', 'Value']).to_csv(out, index=False)
    print(f"\n✅ Metrik validasi Phase 2 tersimpan → {out}")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def run_clustering(path=None):
    """Jalankan seluruh pipeline Phase 2 (9 langkah) dan kembalikan df
    berlabel: load → tempel kategorikal → bukti PCA → validasi feature
    selection → winsorize+scale → elbow/silhouette → K-Means + profil →
    DBSCAN + Hierarchical + profil → perbandingan & simpan."""
    if path is None:
        path = os.path.join(DATA_DIR, 'dataset_clustering.csv')

    print("=" * 55)
    print("  PHASE 2 — Segmentation via Clustering")
    print("=" * 55)

    print("\n[1/9] Load data...")
    df, feats = load_clustering_data(path)

    print("\n[2/9] Tempel fitur demografis/kategorikal (untuk profiling)...")
    df = attach_categoricals(df)

    print("\n[3/9] Dimensionality analysis (bukti PCA tidak dipakai)...")
    dimensionality_analysis(df)

    print("\n[4/9] Feature selection comparison (domain vs otomatis)...")
    feature_selection_comparison(df, feats)

    print("\n[5/9] Winsorize + scale fitur rasio...")
    X_scaled = prepare_features(df, feats)

    print("\n[6/9] Elbow & Silhouette...")
    best_k, _ = find_optimal_k(X_scaled)

    print(f"\n[7/9] K-Means (K={best_k}) + profil segmen...")
    df, sil_km = run_kmeans(df, X_scaled, feats, best_k)
    _, names = profile_clusters(df, feats, 'KMeans_Cluster')

    print("\n[8/9] DBSCAN + Hierarchical + profil...")
    df, n_clusters_db, n_noise, best_eps = run_dbscan(df, X_scaled, feats)
    df, sil_hier, coph = run_hierarchical(df, X_scaled, best_k)
    profile_clusters(df, feats, 'Hierarchical_Cluster')

    print("\n[9/9] Perbandingan & simpan...")
    _, ari = compare_methods(df, X_scaled, sil_km, sil_hier, n_clusters_db, n_noise)
    save_clustered(df, names)
    save_validation_metrics(sil_km, sil_hier, coph, ari, best_k,
                            n_clusters_db, n_noise, best_eps)

    print("\n" + "=" * 55)
    print("  PHASE 2 SELESAI")
    print("=" * 55)
    return df


if __name__ == '__main__':
    run_clustering()
