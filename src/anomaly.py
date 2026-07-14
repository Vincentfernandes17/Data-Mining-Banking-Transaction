"""
anomaly.py — Phase 4: Anomaly & Outlier Detection

Tujuan: menemukan record yang menyimpang jauh dari mayoritas, lalu menentukan
apakah tiap penyimpangan adalah (a) masalah kualitas data, (b) kasus langka
tapi valid, atau (c) sinyal risiko yang layak dieskalasi.

Metode (sesuai rubrik + revisi dosen — pakai varian MODIFIED/robust):
  1. IQR MODIFIED (skew-adjusted via medcouple)
     Pagar IQR klasik (Q1-1.5·IQR, Q3+1.5·IQR) mengasumsikan distribusi
     simetris. Ketiga rasio perilaku kita right-skew berat (skew 2-7),
     sehingga pagar atas klasik terlalu ketat → ekor kanan yang wajar ikut
     ter-flag. Adjusted boxplot (Hubert & Vandervieren, 2008) menggeser
     pagar mengikuti arah/besarnya skew yang diukur MEDCOUPLE (statistik
     skewness robust, Brys–Hubert–Struyf 2004).
  2. Z-SCORE MODIFIED (median/MAD, Iglewicz & Hoaglin 1993)
     Z-score klasik memakai mean & std — dua momen yang justru TERDISTORSI
     oleh outlier yang sedang dicari. Modified z-score memakai median & MAD:
         M = 0.6745 · (x - median) / MAD,   flag bila |M| > 3.5
  3. ISOLATION FOREST + THRESHOLD GAP
     Alih-alih mengasumsikan contamination=5% (angka arbitrer), threshold
     ditarik dari data: skor anomali diurutkan menurun lalu dicari GAP
     (lompatan) terbesar antar skor — titik pemisah alami antara populasi
     normal dan anomali struktural.

  Versi STANDAR ketiganya TETAP dihitung sebagai pembanding sistematis
  (rubrik: "results compared systematically") — lihat
  compare_standard_vs_modified(). Flag RESMI yang dipakai downstream
  (konsensus, klasifikasi, dashboard) adalah versi modified.

Cross-reference:
  - Noise points DBSCAN (Phase 2)  → outlier struktural dari sudut density
  - Segmen K-Means (Phase 2)       → di segmen mana anomali menumpuk
  - Label 'Anomaly' bawaan dataset → HANYA untuk validasi akhir, BUKAN target

Input  : data/dataset_clustered.csv (output Phase 2; punya label cluster)
Output : outputs/phase4/anomaly_report.csv + beberapa plot

Cara running:
    python src/anomaly.py
    from src.anomaly import run_anomaly
    report = run_anomaly()
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'phase4')
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# ── Parameter metode STANDAR (dipertahankan sebagai PEMBANDING) ──
IQR_K     = 1.5    # pagar klasik: Q1 - 1.5·IQR, Q3 + 1.5·IQR
Z_THRESH  = 3.0    # z-score klasik: |z| > 3
IF_CONTAM = 0.05   # asumsi manual ~5% anomali struktural

# ── Parameter metode MODIFIED (dipakai sebagai flag RESMI) ──
MODZ_THRESH  = 3.5    # modified z-score: |M| > 3.5 (Iglewicz & Hoaglin, 1993)
GAP_MIN_FRAC = 0.005  # jendela pencarian gap IsoForest: minimal 0.5% teratas...
GAP_MAX_FRAC = 0.15   # ...maksimal 15% teratas (hindari cut degeneratif di ekstrem)

# Fitur yang diuji (rasio perilaku — sama dengan ruang clustering)
ANOMALY_FEATURES = [
    'CC_Utilization',
    'Transaction_to_Balance_Ratio',
    'Loan_to_Balance_Ratio',
]


# ════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════
def load_data(path=None):
    """Muat dataset berlabel cluster (output Phase 2) dan beri record_id.

    Parameter `path` opsional; default data/dataset_clustered.csv.
    Mengembalikan DataFrame 5000 baris dengan kolom rasio + konteks +
    label cluster. Error yang jelas bila Phase 2 belum dijalankan.
    """
    if path is None:
        path = os.path.join(DATA_DIR, 'dataset_clustered.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} tidak ditemukan. Jalankan Phase 2 dulu "
            f"(python main.py --phase 2) agar label cluster tersedia.")
    df = pd.read_csv(path).reset_index(drop=True)
    df['record_id'] = df.index
    print(f"Dataset loaded : {df.shape[0]:,} baris")
    has_cluster = 'DBSCAN_Cluster' in df.columns
    print(f"Label cluster Phase 2 tersedia : {has_cluster}")
    return df


# ════════════════════════════════════════════════════════════
# MEDCOUPLE — statistik skewness robust untuk adjusted boxplot
# ════════════════════════════════════════════════════════════
def _medcouple(x):
    """Hitung medcouple (MC) — ukuran skewness robust (Brys et al., 2004).

    MC = median dari kernel h(xi, xj) untuk semua pasangan xi ≥ median
    dan xj ≤ median:
        h = ((xi - med) - (med - xj)) / (xi - xj)
    MC > 0 = right-skew, MC < 0 = left-skew, MC = 0 = simetris.

    Implementasi vektorisasi numpy O(n²) — untuk 5000 baris (~6 juta
    pasangan) selesai < 1 detik. Pasangan degenerate xi = xj = median
    (pembagian 0/0 → NaN) diabaikan lewat nanmedian; pada fitur rasio
    kontinu jumlah tie di median praktis nol sehingga tidak berpengaruh.
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    xp = x[x >= med]           # sisi kanan (termasuk median)
    xm = x[x <= med]           # sisi kiri  (termasuk median)
    a = xp[:, None] - med      # jarak xi ke median  (≥ 0)
    b = med - xm[None, :]      # jarak median ke xj  (≥ 0)
    denom = xp[:, None] - xm[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        h = (a - b) / denom
    return float(np.nanmedian(h))


def adjusted_iqr_bounds(s, k=IQR_K):
    """Pagar adjusted boxplot (Hubert & Vandervieren, 2008).

    Pagar klasik dikoreksi faktor eksponensial dari medcouple sehingga
    mengikuti bentuk distribusi:
        MC ≥ 0 : [Q1 - k·e^(-4·MC)·IQR ,  Q3 + k·e^(+3·MC)·IQR]
        MC < 0 : [Q1 - k·e^(-3·MC)·IQR ,  Q3 + k·e^(+4·MC)·IQR]
    Untuk data right-skew (MC>0): pagar atas MELEBAR (ekor kanan wajar tak
    ikut ter-flag), pagar bawah MENYEMPIT. Bila MC = 0 rumus kembali persis
    ke pagar klasik. Mengembalikan (lo, hi, mc).
    """
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    mc = _medcouple(s.values)
    if mc >= 0:
        lo = q1 - k * np.exp(-4 * mc) * iqr
        hi = q3 + k * np.exp( 3 * mc) * iqr
    else:
        lo = q1 - k * np.exp(-3 * mc) * iqr
        hi = q3 + k * np.exp( 4 * mc) * iqr
    return lo, hi, mc


# ════════════════════════════════════════════════════════════
# METODE 1 — IQR (standar 1.5·IQR  vs  MODIFIED skew-adjusted)
# ════════════════════════════════════════════════════════════
def detect_iqr(df, cols):
    """Deteksi outlier univariat berbasis kuartil, dua varian sekaligus.

    Untuk tiap fitur dihitung (a) pagar STANDAR Q1±1.5·IQR sebagai
    pembanding dan (b) pagar MODIFIED skew-adjusted (medcouple) sebagai
    flag resmi. Sebuah record ter-flag bila keluar pagar pada ≥1 fitur.
    Mengembalikan (flags_standar, flags_modified) — dua Series boolean.
    """
    flags_std = pd.DataFrame(index=df.index)
    flags_mod = pd.DataFrame(index=df.index)
    print("\n=== Metode 1: IQR — standar (1.5·IQR) vs MODIFIED (skew-adjusted/medcouple) ===")
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lo_s, hi_s = q1 - IQR_K * iqr, q3 + IQR_K * iqr
        lo_m, hi_m, mc = adjusted_iqr_bounds(df[c])
        flags_std[c] = (df[c] < lo_s) | (df[c] > hi_s)
        flags_mod[c] = (df[c] < lo_m) | (df[c] > hi_m)
        print(f"  {c:32s}: MC={mc:+.3f}")
        print(f"    standar  [{lo_s:10.2f}, {hi_s:10.2f}] → {flags_std[c].sum():4d} outlier")
        print(f"    modified [{lo_m:10.2f}, {hi_m:10.2f}] → {flags_mod[c].sum():4d} outlier")
    comb_std = flags_std.any(axis=1)
    comb_mod = flags_mod.any(axis=1)
    print(f"  Total ter-flag IQR standar  (≥1 fitur): {comb_std.sum()} ({comb_std.mean()*100:.1f}%)")
    print(f"  Total ter-flag IQR MODIFIED (≥1 fitur): {comb_mod.sum()} ({comb_mod.mean()*100:.1f}%)")
    print("  Interpretasi: MC>0 (right-skew) → pagar atas modified melebar, sehingga "
          "ekor kanan yang wajar tidak lagi salah-flag; yang tersisa benar-benar ekstrem.")
    return comb_std, comb_mod


# ════════════════════════════════════════════════════════════
# METODE 2 — Z-SCORE (klasik mean/std  vs  MODIFIED median/MAD)
# ════════════════════════════════════════════════════════════
def detect_zscore(df, cols, thresh=Z_THRESH, mod_thresh=MODZ_THRESH):
    """Deteksi outlier univariat berbasis simpangan, dua varian sekaligus.

    (a) Z-score KLASIK: z = (x-mean)/std, flag |z| > 3 — pembanding.
    (b) Z-score MODIFIED (Iglewicz & Hoaglin): M = 0.6745·(x-median)/MAD,
        flag |M| > 3.5 — flag resmi. Median & MAD tidak terdistorsi outlier,
        sehingga lebih jujur pada data skewed. Bila MAD = 0 (fitur nyaris
        konstan) dipakai fallback mean absolute deviation: M = (x-med)/(1.253314·MeanAD).
    Mengembalikan (flags_klasik, flags_modified).
    """
    flags_std = pd.DataFrame(index=df.index)
    flags_mod = pd.DataFrame(index=df.index)
    print(f"\n=== Metode 2: Z-Score — klasik (|z|>{thresh}) vs MODIFIED (|M|>{mod_thresh}, median/MAD) ===")
    for c in cols:
        s = df[c]
        z = (s - s.mean()) / s.std()
        flags_std[c] = z.abs() > thresh

        med = s.median()
        mad = (s - med).abs().median()
        if mad > 0:
            m = 0.6745 * (s - med) / mad
        else:                       # fallback Iglewicz-Hoaglin bila MAD = 0
            meanad = (s - med).abs().mean()
            m = (s - med) / (1.253314 * meanad)
        flags_mod[c] = m.abs() > mod_thresh
        print(f"  {c:32s}: klasik {flags_std[c].sum():4d} | modified {flags_mod[c].sum():4d} outlier")
    comb_std = flags_std.any(axis=1)
    comb_mod = flags_mod.any(axis=1)
    print(f"  Total ter-flag Z klasik   (≥1 fitur): {comb_std.sum()} ({comb_std.mean()*100:.1f}%)")
    print(f"  Total ter-flag Z MODIFIED (≥1 fitur): {comb_mod.sum()} ({comb_mod.mean()*100:.1f}%)")
    print("  Interpretasi: pada data right-skew, std KLASIK ikut membengkak oleh ekor "
          "sehingga banyak penyimpangan lolos; MAD robust → modified z lebih sensitif "
          "dan konsisten terhadap penyimpangan nyata.")
    return comb_std, comb_mod


# ════════════════════════════════════════════════════════════
# METODE 3 — ISOLATION FOREST (contamination 5% vs THRESHOLD GAP)
# ════════════════════════════════════════════════════════════
def detect_isolation_forest(df, cols, save_plots=True):
    """Deteksi anomali MULTIVARIAT/struktural dengan Isolation Forest.

    Model di-fit sekali; skor anomali = -score_samples (makin tinggi makin
    anomali). Dua cara memotong skor menjadi flag:
    (a) STANDAR : ambil 5% teratas (ekuivalen contamination=0.05) — pembanding.
    (b) GAP     : skor diurutkan menurun, lalu dicari lompatan (gap) TERBESAR
        antar skor berurutan di jendela [GAP_MIN_FRAC, GAP_MAX_FRAC]·n;
        threshold = titik tengah gap → contamination efektif ditentukan DATA,
        bukan asumsi. Jendela mencegah cut degeneratif (1-2 titik terekstrem
        atau di tengah bulk populasi normal).
    Mengembalikan (flags_standar, flags_gap, skor, info_gap).
    """
    print("\n=== Metode 3: Isolation Forest — contamination 5% (manual) vs GAP (dari data) ===")
    iso = IsolationForest(n_estimators=200, random_state=RANDOM_STATE)
    iso.fit(df[cols])
    score = pd.Series(-iso.score_samples(df[cols]), index=df.index)

    # (a) threshold STANDAR: kuantil 95% skor ≡ contamination=0.05
    thr_std = score.quantile(1 - IF_CONTAM)
    flags_std = score >= thr_std

    # (b) threshold GAP: lompatan terbesar antar skor terurut
    s_sorted = np.sort(score.values)[::-1]           # descending
    n = len(s_sorted)
    lo_i = max(1, int(GAP_MIN_FRAC * n))
    hi_i = max(lo_i + 1, int(GAP_MAX_FRAC * n))
    gaps = s_sorted[lo_i - 1:hi_i - 1] - s_sorted[lo_i:hi_i]   # gap[i] = s[i-1]-s[i]
    if len(gaps) == 0 or gaps.max() <= 0:            # degenerate → fallback standar
        print("  PERINGATAN: tidak ada gap berarti di jendela → fallback threshold 5%.")
        cut_idx, thr_gap = int(IF_CONTAM * n), thr_std
    else:
        cut_idx = lo_i + int(np.argmax(gaps))        # jumlah record di atas cut
        thr_gap = (s_sorted[cut_idx - 1] + s_sorted[cut_idx]) / 2
    flags_gap = score >= thr_gap

    med_gap = float(np.median(gaps)) if len(gaps) else float('nan')
    big_gap = float(gaps.max()) if len(gaps) else float('nan')
    print(f"  Skor: min={score.min():.4f}  max={score.max():.4f}")
    print(f"  Jendela pencarian gap : peringkat {lo_i}–{hi_i} "
          f"({GAP_MIN_FRAC*100:.1f}%–{GAP_MAX_FRAC*100:.0f}% teratas)")
    print(f"  Gap terbesar = {big_gap:.5f} pada peringkat {cut_idx} "
          f"(~{big_gap/med_gap:.0f}× median gap {med_gap:.5f} di jendela)")
    print(f"  → threshold GAP = {thr_gap:.4f} → {flags_gap.sum()} anomali "
          f"(contamination efektif {flags_gap.mean()*100:.2f}%)")
    print(f"  Pembanding contamination 5% manual → {flags_std.sum()} anomali")
    print("  Interpretasi: gap besar antar skor = pemisah alami; jumlah anomali kini "
          "ditentukan struktur data, bukan asumsi 5%.")

    _plot_isoforest_gap(s_sorted, cut_idx, thr_gap, int(flags_std.sum()), save_plots)
    info = {'thr_gap': thr_gap, 'thr_std': thr_std, 'cut_idx': cut_idx,
            'contam_eff': flags_gap.mean()}
    return flags_std, flags_gap, score, info


def _plot_isoforest_gap(s_sorted, cut_idx, thr_gap, n_std, save_plots=True):
    """Plot kurva skor anomali terurut + posisi cut GAP vs cut 5% manual.

    Visual pendukung agar pilihan threshold gap bisa diperiksa: sumbu-x =
    peringkat record (fokus 20% teratas), garis merah = cut gap, garis
    abu-abu = cut contamination 5% sebagai pembanding.
    """
    n = len(s_sorted)
    n_show = min(n, max(int(n * 0.20), cut_idx * 2))
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_show + 1), s_sorted[:n_show], lw=1.5,
             label='Skor anomali (urut menurun)')
    plt.axhline(thr_gap, color='red', lw=0.8, ls='-', alpha=0.4)
    plt.axvline(cut_idx, color='red', ls='--',
                label=f'Cut GAP terbesar → {cut_idx} anomali ({cut_idx/n*100:.1f}%)')
    plt.axvline(n_std, color='gray', ls=':',
                label=f'Cut contamination 5% manual → {n_std} anomali')
    plt.xlabel('Peringkat record (menurut skor)')
    plt.ylabel('Anomaly score (makin tinggi makin anomali)')
    plt.title('Isolation Forest — Threshold via GAP Terbesar antar Skor')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'isoforest_gap.png'), dpi=150)
    plt.close()


# ════════════════════════════════════════════════════════════
# PERBANDINGAN SISTEMATIS: STANDAR vs MODIFIED (revisi dosen)
# ════════════════════════════════════════════════════════════
def compare_standard_vs_modified(df, save_plots=True):
    """Bandingkan hasil metode standar vs modified secara sistematis.

    Untuk tiap metode dihitung jumlah & persentase flag versi standar vs
    modified, overlap keduanya, dan Jaccard (irisan/gabungan). Hasil
    disimpan ke standard_vs_modified.csv + grouped bar chart .png supaya
    bisa ditampilkan di laporan/dashboard.
    """
    pairs = [('IQR',       'flag_iqr_std',    'flag_iqr',    'skew-adjusted (medcouple)'),
             ('Z-Score',   'flag_zscore_std', 'flag_zscore', 'modified z (median/MAD)'),
             ('IsoForest', 'flag_if_std',     'flag_if',     'threshold gap (dari data)')]
    rows = []
    for name, c_std, c_mod, desc in pairs:
        s, m = df[c_std], df[c_mod]
        inter, union = int((s & m).sum()), int((s | m).sum())
        rows.append({'Metode': name, 'Varian modified': desc,
                     'Standar': int(s.sum()), 'Standar_%': round(s.mean() * 100, 1),
                     'Modified': int(m.sum()), 'Modified_%': round(m.mean() * 100, 1),
                     'Overlap': inter,
                     'Jaccard': round(inter / union, 3) if union else np.nan})
    comp = pd.DataFrame(rows)

    print(f"\n{'='*60}")
    print("  PERBANDINGAN SISTEMATIS: STANDAR vs MODIFIED")
    print(f"{'='*60}")
    print(comp.to_string(index=False))
    print("\nKesimpulan perbandingan:")
    print("  - IQR standar mengasumsikan simetri → over-flag ekor kanan fitur skewed;")
    print("    versi medcouple menyesuaikan pagar dengan bentuk distribusi.")
    print("  - Z klasik memakai mean/std yang terdistorsi outlier; median/MAD robust.")
    print("  - IsoForest 5% adalah ASUMSI; gap menurunkan threshold dari struktur data.")
    print("  → Downstream (konsensus/klasifikasi) memakai versi MODIFIED.")

    # Grouped bar chart standar vs modified per metode
    x = np.arange(len(comp)); w = 0.38
    plt.figure(figsize=(9, 5))
    plt.bar(x - w / 2, comp['Standar'], w, label='Standar', color='#bdc3c7')
    plt.bar(x + w / 2, comp['Modified'], w, label='Modified', color='#e67e22')
    for xi, (vs, vm) in zip(x, zip(comp['Standar'], comp['Modified'])):
        plt.text(xi - w / 2, vs + 5, str(vs), ha='center', fontsize=9)
        plt.text(xi + w / 2, vm + 5, str(vm), ha='center', fontsize=9, fontweight='bold')
    plt.xticks(x, [f"{r['Metode']}\n({r['Varian modified']})" for _, r in comp.iterrows()],
               fontsize=9)
    plt.ylabel('Jumlah record ter-flag')
    plt.title('Metode Standar vs Modified — Jumlah Anomali Ter-flag')
    plt.legend(); plt.grid(axis='y', alpha=0.3); plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'standard_vs_modified.png'), dpi=150)
    plt.close()
    comp.to_csv(os.path.join(OUTPUT_DIR, 'standard_vs_modified.csv'), index=False)
    return comp


# ════════════════════════════════════════════════════════════
# SYSTEMATIC COMPARISON antar 3 metode (versi MODIFIED/resmi)
# ════════════════════════════════════════════════════════════
def compare_methods(df, save_plots=True):
    """Bandingkan hasil 3 metode resmi (modified) & hitung konsensus.

    Menambahkan kolom n_methods (0-3 = berapa metode yang menandai record)
    lalu menggambar heatmap overlap antar metode. Konsensus kuat = ≥2 metode
    setuju — dipakai sebagai definisi anomali utama di seluruh downstream.
    """
    print(f"\n{'='*60}")
    print("  PERBANDINGAN SISTEMATIS 3 METODE (versi modified)")
    print(f"{'='*60}")
    summary = pd.DataFrame({
        'IQR (adjusted)':    [df['flag_iqr'].sum()],
        'Z-Score (modified)':[df['flag_zscore'].sum()],
        'IsoForest (gap)':   [df['flag_if'].sum()],
    }, index=['n_flagged'])
    print(summary.to_string())

    # Konsensus: berapa metode yang sepakat per record
    df['n_methods'] = (df['flag_iqr'].astype(int)
                       + df['flag_zscore'].astype(int)
                       + df['flag_if'].astype(int))
    print("\nDistribusi konsensus (jumlah metode yang setuju):")
    print(df['n_methods'].value_counts().sort_index().to_string())
    print(f"  Anomali konsensus kuat (≥2 metode): {(df['n_methods']>=2).sum()}")
    print(f"  Anomali oleh ketiga metode (=3)   : {(df['n_methods']==3).sum()}")

    # Heatmap overlap antar metode
    methods = ['flag_iqr', 'flag_zscore', 'flag_if']
    labels  = ['IQR (adj)', 'Z-Score (mod)', 'IsoForest (gap)']
    overlap = np.zeros((3, 3), dtype=int)
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            overlap[i, j] = int((df[a] & df[b]).sum())
    plt.figure(figsize=(6, 5))
    sns.heatmap(overlap, annot=True, fmt='d', cmap='Reds',
                xticklabels=labels, yticklabels=labels)
    plt.title('Overlap Anomali antar Metode Modified\n(diagonal = total per metode)')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'method_overlap.png'), dpi=150)
    plt.close()
    return df


# ════════════════════════════════════════════════════════════
# CROSS-REFERENCE dengan Phase 2 (DBSCAN noise & segmen K-Means)
# ════════════════════════════════════════════════════════════
def cross_reference(df, save_plots=True):
    """Silangkan anomali konsensus dengan hasil clustering Phase 2.

    Dua sudut pandang: (1) DBSCAN noise = outlier density-based — bila
    record disepakati metode statistik DAN density, itu outlier paling
    meyakinkan; (2) distribusi anomali per segmen K-Means menunjukkan di
    profil nasabah mana risiko menumpuk.
    """
    print(f"\n{'='*60}")
    print("  CROSS-REFERENCE dengan Phase 2")
    print(f"{'='*60}")

    consensus = df['n_methods'] >= 2

    if 'DBSCAN_Cluster' in df.columns:
        df['dbscan_noise'] = df['DBSCAN_Cluster'] == -1
        ct = pd.crosstab(consensus, df['dbscan_noise'],
                         rownames=['Anomali ≥2 metode'], colnames=['DBSCAN noise'])
        print("\nAnomali statistik vs DBSCAN noise:")
        print(ct.to_string())
        both = (consensus & df['dbscan_noise']).sum()
        print(f"  Disepakati statistik DAN density (DBSCAN): {both} record "
              f"→ outlier paling meyakinkan (dua sudut pandang berbeda setuju).")

    if 'KMeans_Segment' in df.columns:
        print("\nDi segmen mana anomali konsensus menumpuk:")
        seg = df[consensus]['KMeans_Segment'].value_counts()
        for name, cnt in seg.items():
            base = (df['KMeans_Segment'] == name).sum()
            print(f"  {name:38s}: {cnt:4d} anomali dari {base} ({cnt/base*100:.1f}%)")

    return df


# ════════════════════════════════════════════════════════════
# KLASIFIKASI tiap anomali: data error / rare-valid / risk signal
# Aturan transparan berbasis bukti yang bisa diperiksa.
# ════════════════════════════════════════════════════════════
def classify_anomalies(df):
    """Klasifikasi tiap record ter-flag (≥1 metode) ke tiga tipe anomali.

    Aturan berurutan (rule-based, transparan):
    (a) RISK SIGNAL       — nilai ekstrem pada dimensi berisiko (CC≥2× limit,
        transaksi ≥10× saldo, pinjaman ≥100× saldo) DAN dikuatkan ≥2 metode.
    (b) DATA ERROR/QUALITY — masalah internal terverifikasi (saldo akhir
        negatif, saldo tidak konsisten dgn tipe & jumlah transaksi).
    (c) RARE BUT VALID    — menyimpang tapi masih masuk akal.
    Mengembalikan DataFrame subset anomali + kolom classification & evidence.
    """
    print(f"\n{'='*60}")
    print("  KLASIFIKASI ANOMALI (data error / rare-valid / risk signal)")
    print(f"{'='*60}")

    # Cek konsistensi balance (butuh Transaction Type & Amount)
    has_txn = {'Transaction Type', 'Transaction Amount',
               'Account Balance', 'Account Balance After Transaction'} <= set(df.columns)

    def balance_inconsistent(row):
        """True bila saldo-akhir tidak cocok dengan tipe & jumlah transaksi."""
        if not has_txn:
            return False
        diff = row['Account Balance After Transaction'] - row['Account Balance']
        t, amt = row['Transaction Type'], row['Transaction Amount']
        if t == 'Deposit':
            return abs(diff - amt) > 1
        if t == 'Withdrawal':
            return abs(diff + amt) > 1
        return False   # Transfer: arah tak diketahui → tidak dinilai

    # Catatan kualitas data AGREGAT: inkonsistensi saldo bersifat sistemik
    # (artefak cara dataset sintetis dibuat), jadi BUKAN error per-record yang
    # boleh menimpa sinyal risiko. Kita laporkan prevalensinya, lalu hanya
    # memakainya untuk mengklasifikasi record yang BUKAN risk signal.
    if has_txn:
        inc = df.apply(balance_inconsistent, axis=1)
        print(f"\nCatatan kualitas data (agregat): {inc.sum()} record "
              f"({inc.mean()*100:.1f}%) punya saldo akhir tidak konsisten dengan "
              f"tipe/jumlah transaksi — kemungkinan artefak sistemik dataset "
              f"sintetis, bukan error individual.")

    def classify(row):
        """Terapkan aturan (a)-(c) pada satu record; kembalikan (tipe, bukti)."""
        cc   = row.get('CC_Utilization', 0)
        txn  = row.get('Transaction_to_Balance_Ratio', 0)
        loan = row.get('Loan_to_Balance_Ratio', 0)
        risk = []
        if cc >= 2.0:
            risk.append(f'saldo kartu {cc:.1f}× limit (jauh over-limit)')
        if txn >= 10:
            risk.append(f'transaksi {txn:.0f}× saldo (tekanan likuiditas)')
        if loan >= 100:
            risk.append(f'pinjaman {loan:.0f}× saldo (leverage ekstrem)')

        # (a) RISK SIGNAL — ekstrem pada dimensi berisiko + dikuatkan ≥2 metode.
        #     Diprioritaskan: ini yang paling penting untuk dieskalasi bank.
        if risk and row['n_methods'] >= 2:
            return 'Risk Signal', '; '.join(risk)

        # (b) DATA ERROR / QUALITY — masalah internal yang bisa diverifikasi,
        #     untuk record yang BUKAN risk signal kuat.
        if 'Account Balance After Transaction' in row and \
                row['Account Balance After Transaction'] < 0:
            return 'Data Error / Quality', 'saldo akhir negatif (overdraft) — perlu verifikasi'
        if has_txn and balance_inconsistent(row):
            return 'Data Error / Quality', \
                'saldo akhir tidak konsisten dgn tipe & jumlah transaksi'

        # (c) RARE BUT VALID — menyimpang tapi masih masuk akal
        if risk:
            return 'Rare but Valid', '; '.join(risk) + ' (moderat / 1 metode)'
        return 'Rare but Valid', 'menyimpang pada ≥1 fitur tapi dalam rentang wajar'

    anomalies = df[df['n_methods'] >= 1].copy()
    cls = anomalies.apply(classify, axis=1, result_type='expand')
    anomalies['classification'] = cls[0]
    anomalies['evidence'] = cls[1]

    print("\nDistribusi klasifikasi (semua record ter-flag ≥1 metode):")
    print(anomalies['classification'].value_counts().to_string())

    print("\nKlasifikasi untuk anomali KONSENSUS (≥2 metode):")
    strong = anomalies[anomalies['n_methods'] >= 2]
    print(strong['classification'].value_counts().to_string())
    return anomalies


# ════════════════════════════════════════════════════════════
# VALIDASI dengan label 'Anomaly' bawaan (HANYA setelah mining)
# ════════════════════════════════════════════════════════════
def validate_against_label(df, anomalies):
    """Ukur keselarasan temuan dengan label 'Anomaly' bawaan dataset.

    Label ini TIDAK pernah dipakai saat deteksi (aturan proyek) — hanya
    sebagai validasi akhir. Keselarasan rendah adalah hasil yang WAJAR:
    label bawaan terbukti tak terdeteksi secara univariat (MI ≈ 0, Phase 1),
    sedangkan anomali kita berbasis perilaku finansial nyata.
    """
    if 'Anomaly' not in df.columns:
        return
    print(f"\n{'='*60}")
    print("  VALIDASI dengan label 'Anomaly' bawaan (sesudah mining)")
    print(f"{'='*60}")
    print("Catatan: label ini TIDAK dipakai saat deteksi — hanya untuk mengukur "
          "apakah anomali yang kita temukan selaras dengan label.")
    label_anom = df['Anomaly'] == -1
    consensus = df['n_methods'] >= 2
    ct = pd.crosstab(consensus, label_anom,
                     rownames=['Anomali ≥2 metode'], colnames=['Label = -1'])
    print(ct.to_string())
    base_rate = label_anom.mean() * 100
    flagged_rate = df[consensus]['Anomaly'].eq(-1).mean() * 100 if consensus.sum() else 0
    print(f"\nBase rate label -1 keseluruhan : {base_rate:.1f}%")
    print(f"Base rate label -1 di anomali  : {flagged_rate:.1f}%")

    # Precision/Recall/F1 memakai label -1 sebagai kelas positif "ground truth"
    # dan konsensus (≥2 metode) sebagai prediksi. Ini metrik yang diminta rubrik,
    # sekaligus bukti kuantitatif bahwa label sintetis TIDAK selaras dgn perilaku.
    tp = int((consensus & label_anom).sum())
    fp = int((consensus & ~label_anom).sum())
    fn = int((~consensus & label_anom).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    lift_vs_base = (flagged_rate / base_rate) if base_rate else float('nan')
    print(f"\nPrecision (konsensus vs label) : {precision:.3f}  ({tp}/{tp+fp})")
    print(f"Recall    (konsensus vs label) : {recall:.3f}  ({tp}/{tp+fn})")
    print(f"F1-score                        : {f1:.3f}")
    print(f"Lift keselarasan vs base rate  : {lift_vs_base:.2f}× "
          f"(rate label di anomali / base rate)")
    print("Interpretasi: label 'Anomaly' dataset ini ~uniform & tidak terdeteksi "
          "secara univariat (lihat MI di Phase 1), jadi keselarasan rendah adalah "
          "WAJAR. Anomali kita berbasis PERILAKU finansial, bukan label sintetis.")

    # Simpan ringkasan validasi label untuk laporan (Appendix D)
    out = os.path.join(OUTPUT_DIR, 'label_validation.csv')
    pd.DataFrame([
        ('Base rate label -1 (populasi)',       round(base_rate, 2)),
        ('Base rate label -1 (di anomali ≥2)',  round(flagged_rate, 2)),
        ('Precision (konsensus vs label)',      round(precision, 4)),
        ('Recall (konsensus vs label)',         round(recall, 4)),
        ('F1-score',                            round(f1, 4)),
        ('TP', tp), ('FP', fp), ('FN', fn),
    ], columns=['Metric', 'Value']).to_csv(out, index=False)
    print(f"✅ Ringkasan validasi label tersimpan → {out}")


# ════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════
def plot_anomalies(df, save_plots=True):
    """Scatter rasio perilaku (skala symlog) dengan anomali konsensus disorot.

    Dua panel: CC_Utilization vs Transaction/Balance, dan Transaction/Balance
    vs Loan/Balance. Titik merah X = record yang disepakati ≥2 metode.
    """
    feats = [c for c in ANOMALY_FEATURES if c in df.columns]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    consensus = df['n_methods'] >= 2
    for ax, (i, j) in zip(axes, [(0, 1), (1, 2)]):
        ax.scatter(df[feats[i]], df[feats[j]], s=12, alpha=0.3,
                   c='lightgray', label='Normal')
        ax.scatter(df[consensus][feats[i]], df[consensus][feats[j]], s=40,
                   alpha=0.8, c='red', marker='x', label='Anomali ≥2 metode')
        ax.set_xlabel(feats[i]); ax.set_ylabel(feats[j])
        ax.set_xscale('symlog'); ax.set_yscale('symlog')
        ax.legend(fontsize=8)
    plt.suptitle('Anomali Konsensus pada Ruang Rasio Perilaku (skala symlog)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'anomaly_scatter.png'), dpi=150)
    plt.close()


# ════════════════════════════════════════════════════════════
# EXPORT report
# ════════════════════════════════════════════════════════════
def export_report(anomalies):
    """Simpan tabel anomali (record ter-flag ≥1 metode) ke anomaly_report.csv.

    Kolom mencakup klasifikasi + bukti, flag per metode (modified & standar
    untuk transparansi), skor IsoForest, cross-ref cluster, dan nilai fitur
    pendukung. Diurutkan dari konsensus tertinggi.
    """
    cols = [c for c in [
        'record_id', 'classification', 'evidence', 'n_methods',
        'flag_iqr', 'flag_zscore', 'flag_if', 'if_score',
        'flag_iqr_std', 'flag_zscore_std', 'flag_if_std',
        'dbscan_noise', 'KMeans_Segment',
        'CC_Utilization', 'Transaction_to_Balance_Ratio', 'Loan_to_Balance_Ratio',
        'Account Balance', 'Credit Limit', 'Credit Card Balance',
        'Transaction Type', 'Transaction Amount',
        'Account Balance After Transaction', 'Anomaly',
    ] if c in anomalies.columns]

    report = anomalies.sort_values(['n_methods', 'if_score'],
                                   ascending=False)[cols].round(3)
    out = os.path.join(OUTPUT_DIR, 'anomaly_report.csv')
    report.to_csv(out, index=False)
    print(f"\n✅ Anomaly report tersimpan → {out}  ({len(report)} record)")

    print("\n── Contoh 10 anomali teratas (konsensus tertinggi) ──")
    show = [c for c in ['record_id', 'classification', 'n_methods',
                        'CC_Utilization', 'Transaction_to_Balance_Ratio',
                        'Loan_to_Balance_Ratio', 'KMeans_Segment'] if c in report.columns]
    print(report[show].head(10).to_string(index=False))
    return report


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def run_anomaly(path=None):
    """Jalankan seluruh pipeline Phase 4 dan kembalikan anomaly report.

    Urutan: load → 3 detektor (masing-masing standar + modified) →
    perbandingan standar-vs-modified → konsensus 3 metode modified →
    cross-reference Phase 2 → klasifikasi → validasi label → plot →
    export report + dataset_final.csv untuk dashboard.
    """
    print("=" * 55)
    print("  PHASE 4 — Anomaly & Outlier Detection")
    print("=" * 55)

    df = load_data(path)
    feats = [c for c in ANOMALY_FEATURES if c in df.columns]

    # 3 metode — tiap metode menghasilkan flag STANDAR (pembanding) dan
    # MODIFIED (resmi, dipakai downstream: kolom tanpa akhiran _std)
    df['flag_iqr_std'],    df['flag_iqr']    = detect_iqr(df, feats)
    df['flag_zscore_std'], df['flag_zscore'] = detect_zscore(df, feats)
    df['flag_if_std'], df['flag_if'], df['if_score'], _ = \
        detect_isolation_forest(df, feats)

    # Perbandingan standar vs modified (revisi dosen) + antar 3 metode
    compare_standard_vs_modified(df)
    df = compare_methods(df)
    df = cross_reference(df)
    anomalies = classify_anomalies(df)
    validate_against_label(df, anomalies)

    # bawa kolom hasil ke anomalies untuk export
    for c in ['dbscan_noise', 'if_score', 'n_methods',
              'flag_iqr', 'flag_zscore', 'flag_if',
              'flag_iqr_std', 'flag_zscore_std', 'flag_if_std']:
        if c in df.columns and c not in anomalies.columns:
            anomalies[c] = df.loc[anomalies.index, c]

    plot_anomalies(df)
    report = export_report(anomalies)

    # Simpan dataset GABUNGAN lengkap (semua 5000 baris) untuk dashboard Phase 5:
    # rasio + konteks + label cluster + kolom anomali. 'classification' = 'Normal'
    # untuk baris yang tidak ter-flag.
    df_final = df.copy()
    df_final['classification'] = 'Normal'
    df_final.loc[anomalies.index, 'classification'] = anomalies['classification']
    df_final['is_anomaly'] = df_final['n_methods'] >= 1
    df_final['is_consensus_anomaly'] = df_final['n_methods'] >= 2
    out_final = os.path.join(DATA_DIR, 'dataset_final.csv')
    df_final.to_csv(out_final, index=False)
    print(f"✅ Dataset final (untuk dashboard) tersimpan → {out_final}  "
          f"({df_final.shape[0]} baris × {df_final.shape[1]} kolom)")

    print("\n" + "=" * 55)
    print("  PHASE 4 SELESAI")
    print("=" * 55)
    return report


if __name__ == '__main__':
    run_anomaly()
