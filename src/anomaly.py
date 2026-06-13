"""
anomaly.py — Phase 4: Anomaly & Outlier Detection

Tujuan: menemukan record yang menyimpang jauh dari mayoritas, lalu menentukan
apakah tiap penyimpangan adalah (a) masalah kualitas data, (b) kasus langka
tapi valid, atau (c) sinyal risiko yang layak dieskalasi.

Metode (sesuai rubrik):
  1. IQR            — outlier univariat berbasis kuartil
  2. Z-Score        — outlier univariat berbasis simpangan baku (|z| > 3)
  3. Isolation Forest — outlier MULTIVARIAT/struktural (ensemble of trees)

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

RANDOM_STATE  = 42
Z_THRESH      = 3.0
IF_CONTAM     = 0.05   # asumsi ~5% anomali struktural

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
# METODE 1 — IQR
# ════════════════════════════════════════════════════════════
def detect_iqr(df, cols):
    flags = pd.DataFrame(index=df.index)
    print("\n=== Metode 1: IQR (Q1-1.5·IQR, Q3+1.5·IQR) ===")
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flags[c] = (df[c] < lo) | (df[c] > hi)
        print(f"  {c:32s}: batas [{lo:.2f}, {hi:.2f}] → {flags[c].sum()} outlier")
    combined = flags.any(axis=1)
    print(f"  Total record ter-flag IQR (≥1 fitur): {combined.sum()} "
          f"({combined.mean()*100:.1f}%)")
    return combined


# ════════════════════════════════════════════════════════════
# METODE 2 — Z-SCORE
# ════════════════════════════════════════════════════════════
def detect_zscore(df, cols, thresh=Z_THRESH):
    flags = pd.DataFrame(index=df.index)
    print(f"\n=== Metode 2: Z-Score (|z| > {thresh}) ===")
    for c in cols:
        z = (df[c] - df[c].mean()) / df[c].std()
        flags[c] = z.abs() > thresh
        print(f"  {c:32s}: {flags[c].sum()} outlier")
    combined = flags.any(axis=1)
    print(f"  Total record ter-flag Z-Score (≥1 fitur): {combined.sum()} "
          f"({combined.mean()*100:.1f}%)")
    return combined


# ════════════════════════════════════════════════════════════
# METODE 3 — ISOLATION FOREST (multivariat)
# ════════════════════════════════════════════════════════════
def detect_isolation_forest(df, cols, contamination=IF_CONTAM):
    print(f"\n=== Metode 3: Isolation Forest (contamination={contamination}) ===")
    iso = IsolationForest(contamination=contamination, n_estimators=200,
                          random_state=RANDOM_STATE)
    pred = iso.fit_predict(df[cols])           # -1 = anomali, 1 = normal
    score = -iso.score_samples(df[cols])       # makin tinggi makin anomali
    flags = pd.Series(pred == -1, index=df.index)
    print(f"  Total record ter-flag Isolation Forest: {flags.sum()} "
          f"({flags.mean()*100:.1f}%)")
    return flags, pd.Series(score, index=df.index)


# ════════════════════════════════════════════════════════════
# SYSTEMATIC COMPARISON antar metode
# ════════════════════════════════════════════════════════════
def compare_methods(df, save_plots=True):
    print(f"\n{'='*60}")
    print("  PERBANDINGAN SISTEMATIS 3 METODE")
    print(f"{'='*60}")
    summary = pd.DataFrame({
        'IQR':      [df['flag_iqr'].sum()],
        'Z-Score':  [df['flag_zscore'].sum()],
        'IsoForest':[df['flag_if'].sum()],
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
    labels  = ['IQR', 'Z-Score', 'IsoForest']
    overlap = np.zeros((3, 3), dtype=int)
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            overlap[i, j] = int((df[a] & df[b]).sum())
    plt.figure(figsize=(6, 5))
    sns.heatmap(overlap, annot=True, fmt='d', cmap='Reds',
                xticklabels=labels, yticklabels=labels)
    plt.title('Overlap Anomali antar Metode\n(diagonal = total per metode)')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'method_overlap.png'), dpi=150)
    plt.close()
    return df


# ════════════════════════════════════════════════════════════
# CROSS-REFERENCE dengan Phase 2 (DBSCAN noise & segmen K-Means)
# ════════════════════════════════════════════════════════════
def cross_reference(df, save_plots=True):
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
    print(f"\n{'='*60}")
    print("  KLASIFIKASI ANOMALI (data error / rare-valid / risk signal)")
    print(f"{'='*60}")

    # Cek konsistensi balance (butuh Transaction Type & Amount)
    has_txn = {'Transaction Type', 'Transaction Amount',
               'Account Balance', 'Account Balance After Transaction'} <= set(df.columns)

    def balance_inconsistent(row):
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
    print("Interpretasi: label 'Anomaly' dataset ini ~uniform & tidak terdeteksi "
          "secara univariat (lihat MI di Phase 1), jadi keselarasan rendah adalah "
          "WAJAR. Anomali kita berbasis PERILAKU finansial, bukan label sintetis.")


# ════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════
def plot_anomalies(df, save_plots=True):
    feats = [c for c in ANOMALY_FEATURES if c in df.columns]
    # Scatter: anomali konsensus disorot
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
    cols = [c for c in [
        'record_id', 'classification', 'evidence', 'n_methods',
        'flag_iqr', 'flag_zscore', 'flag_if', 'if_score',
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
    print("=" * 55)
    print("  PHASE 4 — Anomaly & Outlier Detection")
    print("=" * 55)

    df = load_data(path)
    feats = [c for c in ANOMALY_FEATURES if c in df.columns]

    # 3 metode
    df['flag_iqr']    = detect_iqr(df, feats)
    df['flag_zscore'] = detect_zscore(df, feats)
    df['flag_if'], df['if_score'] = detect_isolation_forest(df, feats)

    # Perbandingan + cross-reference + klasifikasi + validasi
    df = compare_methods(df)
    df = cross_reference(df)
    anomalies = classify_anomalies(df)
    validate_against_label(df, anomalies)

    # bawa kolom hasil ke anomalies untuk export
    for c in ['dbscan_noise', 'if_score', 'n_methods',
              'flag_iqr', 'flag_zscore', 'flag_if']:
        if c in df.columns and c not in anomalies.columns:
            anomalies[c] = df.loc[anomalies.index, c]

    plot_anomalies(df)
    report = export_report(anomalies)

    print("\n" + "=" * 55)
    print("  PHASE 4 SELESAI")
    print("=" * 55)
    return report


if __name__ == '__main__':
    run_anomaly()
