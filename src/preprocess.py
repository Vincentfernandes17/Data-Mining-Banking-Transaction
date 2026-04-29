"""
src/preprocess.py
Phase 1 — Data Understanding & Preprocessing Pipeline

Semua fungsi preprocessing ada di sini.
Dipanggil dari main.py atau langsung dari notebook.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif

from config import (
    PII_COLS, ID_COLS, DATE_COLS, NUMERIC_COLS, NOMINAL_COLS,
    CLUSTERING_COLS, ARM_COLS,
    CLUSTERING_DATA_PATH, ARM_DATA_PATH, OUTPUT_DIR
)


# ════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV dataset."""
    df = pd.read_csv(path)
    print(f"✅ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ════════════════════════════════════════════════════════════
# 2. DATA VALIDATION
# ════════════════════════════════════════════════════════════

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validasi logika bisnis:
    - Balance consistency (Deposit/Withdrawal)
    - Age range
    - CC over limit
    - Interest Rate range

    Returns df tanpa kolom helper (kolom helper di-drop di sini).
    """
    df = df.copy()

    # --- Balance consistency ---
    def check_balance(row):
        diff = row['Account Balance After Transaction'] - row['Account Balance']
        t = row['Transaction Type']
        amt = row['Transaction Amount']
        if t == 'Deposit':
            return abs(diff - amt)
        elif t == 'Withdrawal':
            return abs(diff + amt)
        return None  # Transfer dikecualikan

    df['_balance_check'] = df.apply(check_balance, axis=1)
    n_inconsistent = (df['_balance_check'] > 1).sum()

    # Cross-check dengan Anomaly label
    df['_is_inconsistent'] = df['_balance_check'] > 1
    cross = pd.crosstab(df['_is_inconsistent'], df['Anomaly'])
    anomaly_in_inconsistent = cross.loc[True, -1] if (True in cross.index and -1 in cross.columns) else 0
    pct = anomaly_in_inconsistent / n_inconsistent * 100 if n_inconsistent > 0 else 0

    print(f"\n── Balance Consistency ──")
    print(f"  Inkonsisten : {n_inconsistent:,} baris ({n_inconsistent/len(df)*100:.1f}%)")
    print(f"  Yang Anomaly=-1 : {anomaly_in_inconsistent} ({pct:.1f}%)")
    print(f"  → Keputusan: TIDAK dihapus (inkonsistensi tidak berkorelasi dengan anomali)")

    # --- Age ---
    invalid_age = df[df['Age'] < 17]
    print(f"\n── Age Validation ──")
    print(f"  Usia < 17 tahun : {len(invalid_age)} baris → {'Aman ✅' if len(invalid_age)==0 else 'Perlu ditangani'}")

    # --- CC Over Limit ---
    cc_over = df[df['Credit Card Balance'] > df['Credit Limit']]
    print(f"\n── CC Over Limit ──")
    print(f"  Nasabah over limit : {len(cc_over):,} ({len(cc_over)/len(df)*100:.1f}%)")
    print(f"  → Dipertahankan, akan jadi sinyal di Phase 4")

    # --- Interest Rate ---
    invalid_rate = df[(df['Interest Rate'] < 0) | (df['Interest Rate'] > 100)]
    print(f"\n── Interest Rate ──")
    print(f"  Rate tidak valid : {len(invalid_rate)} baris → {'Aman ✅' if len(invalid_rate)==0 else 'Perlu ditangani'}")

    # Bersihkan kolom helper
    df.drop(columns=['_balance_check', '_is_inconsistent'], inplace=True)

    return df


# ════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ekstrak fitur baru dari kolom tanggal dan buat rasio finansial.
    """
    df = df.copy()

    # Parse tanggal
    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Reference date = tanggal max di dataset
    reference_date = df[DATE_COLS].max().max()
    print(f"Reference date: {reference_date.date()}")

    # Fitur temporal
    df['Account_Age_Years'] = (
        (reference_date - df['Date Of Account Opening']).dt.days / 365
    ).round(1)

    df['Days_Since_Last_Transaction'] = (
        (reference_date - df['Last Transaction Date']).dt.days
    )

    # Rasio finansial
    df['CC_Utilization'] = df['Credit Card Balance'] / df['Credit Limit']
    df['Transaction_to_Balance_Ratio'] = (
        df['Transaction Amount'] / df['Account Balance'].replace(0, np.nan)
    )

    print("✅ Feature engineering selesai")
    print(f"   Fitur baru: Account_Age_Years, Days_Since_Last_Transaction,")
    print(f"               CC_Utilization, Transaction_to_Balance_Ratio")

    return df


# ════════════════════════════════════════════════════════════
# 4. CLEANING — DROP KOLOM
# ════════════════════════════════════════════════════════════

def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop PII, surrogate keys, dan kolom tanggal mentah.
    Branch ID dipertahankan karena merepresentasikan entitas cabang.
    """
    cols_to_drop = PII_COLS + ID_COLS + DATE_COLS
    # Hanya drop yang benar-benar ada di dataframe
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    df_clean = df.drop(columns=cols_to_drop)

    print(f"✅ Drop kolom selesai")
    print(f"   Sebelum : {df.shape[1]} kolom")
    print(f"   Sesudah : {df_clean.shape[1]} kolom")
    print(f"   Di-drop : {len(cols_to_drop)} kolom (PII + ID + tanggal mentah)")

    return df_clean


# ════════════════════════════════════════════════════════════
# 5. BINNING
# ════════════════════════════════════════════════════════════

def _safe_qcut(series: pd.Series, labels: list) -> pd.Series:
    """qcut dengan rank method='first' untuk menghindari duplicate edges."""
    ranked = series.rank(method='first')
    try:
        return pd.qcut(ranked, q=len(labels), labels=labels)
    except ValueError:
        n_bins = min(len(labels), ranked.nunique())
        if n_bins < 2:
            return pd.Series([labels[0]] * len(series), index=series.index)
        return pd.qcut(ranked, q=n_bins, labels=labels[:n_bins])


def bin_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Diskretisasi kolom numerik menjadi kategori bermakna.
    Wajib untuk ARM (Phase 3) karena Apriori hanya menerima data kategorikal.
    """
    df = df.copy()

    df['Age_Group'] = _safe_qcut(
        df['Age'], ['Young', 'Adult', 'Middle-aged', 'Senior']
    )
    df['Balance_Bucket'] = _safe_qcut(
        df['Account Balance'], ['Low', 'Lower-Mid', 'Upper-Mid', 'High']
    )
    df['Transaction_Size'] = _safe_qcut(
        df['Transaction Amount'], ['Small', 'Medium', 'Large', 'Very Large']
    )
    df['Loan_Size'] = _safe_qcut(
        df['Loan Amount'], ['Small', 'Medium', 'Large', 'Very Large']
    )
    df['Rate_Category'] = _safe_qcut(
        df['Interest Rate'], ['Low', 'Moderate', 'High']
    )
    df['CC_Utilization_Category'] = pd.cut(
        df['CC_Utilization'],
        bins=[-np.inf, 0.09, 0.29, 0.49, 0.70, np.inf],
        labels=['Excellent', 'Good', 'Moderate', 'High', 'Very High'],
        include_lowest=True
    )

    print("✅ Binning selesai")
    for col in ['Age_Group', 'Balance_Bucket', 'Transaction_Size',
                'Loan_Size', 'Rate_Category', 'CC_Utilization_Category']:
        print(f"   {col}: {df[col].value_counts().to_dict()}")

    return df


# ════════════════════════════════════════════════════════════
# 6. ENCODING
# ════════════════════════════════════════════════════════════

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label Encoding untuk variabel biner/ordinal.
    One-Hot Encoding untuk variabel nominal.
    """
    df = df.copy()
    le = LabelEncoder()

    # Label encoding
    for col in ['Loan Status', 'Account Type', 'Resolution Status']:
        if col in df.columns:
            df[f'{col}_Encoded'] = le.fit_transform(df[col])

    # One-Hot Encoding
    nominal = [c for c in NOMINAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=nominal, drop_first=False)

    print(f"✅ Encoding selesai → shape: {df.shape}")
    return df


# ════════════════════════════════════════════════════════════
# 7. NORMALISASI
# ════════════════════════════════════════════════════════════

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Min-Max Scaling untuk kolom numerik."""
    df = df.copy()
    numeric = [c for c in NUMERIC_COLS if c in df.columns]

    scaler = MinMaxScaler()
    df[numeric] = scaler.fit_transform(df[numeric])

    print(f"✅ Normalisasi selesai → {len(numeric)} kolom di-scale ke [0, 1]")
    return df


# ════════════════════════════════════════════════════════════
# 8. FEATURE SELECTION
# ════════════════════════════════════════════════════════════

def feature_selection(df: pd.DataFrame, save_plots: bool = True) -> None:
    """
    Correlation Matrix + Mutual Information.
    Hasil divisualisasikan dan disimpan ke outputs/.
    """
    numeric = [c for c in NUMERIC_COLS if c in df.columns]

    # --- Correlation Matrix ---
    corr = df[numeric].corr()
    plt.figure(figsize=(13, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, square=True, linewidths=0.5)
    plt.title('Correlation Matrix — After Cleaning', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'), dpi=150)
    plt.show()

    # Pasangan highly correlated
    print("=== Pasangan Highly Correlated (|r| > 0.85) ===")
    found = False
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > 0.85:
                print(f"  {corr.columns[i]} ↔ {corr.columns[j]}: {r:.3f}")
                found = True
    if not found:
        print("  Tidak ada pasangan dengan |r| > 0.85 ✅")

    # --- Mutual Information ---
    if 'Anomaly' in df.columns:
        target   = df['Anomaly']
        features = df[numeric]
        mi_scores = mutual_info_classif(features, target, random_state=42)
        mi_df = pd.DataFrame({
            'Feature': numeric,
            'Mutual_Information': mi_scores
        }).sort_values('Mutual_Information', ascending=False)

        plt.figure(figsize=(10, 7))
        colors = ['#2ecc71' if x > 0.01 else '#e74c3c'
                  for x in mi_df['Mutual_Information']]
        plt.barh(mi_df['Feature'], mi_df['Mutual_Information'], color=colors)
        plt.axvline(x=0.01, color='red', linestyle='--', alpha=0.7,
                    label='Threshold (0.01)')
        plt.title('Mutual Information Score\n(Entropy-based Feature Selection)',
                  fontsize=12, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(OUTPUT_DIR, 'mutual_information.png'), dpi=150)
        plt.show()

        print("\n=== Mutual Information Score ===")
        print(mi_df.to_string(index=False))


# ════════════════════════════════════════════════════════════
# 9. SAVE DATASETS
# ════════════════════════════════════════════════════════════

def save_datasets(df_encoded: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """
    Simpan dua dataset output:
    - dataset_clustering.csv  → untuk Phase 2
    - dataset_arm.csv         → untuk Phase 3
    """
    # Clustering dataset
    clust_cols = [c for c in CLUSTERING_COLS if c in df_encoded.columns]
    data_for_clustering = df_encoded[clust_cols].copy()
    data_for_clustering['Anomaly'] = df_encoded['Anomaly'].values
    data_for_clustering.to_csv(CLUSTERING_DATA_PATH, index=False)

    # ARM dataset
    arm_cols = [c for c in ARM_COLS if c in df_clean.columns]
    data_for_arm = df_clean[arm_cols]
    data_for_arm.to_csv(ARM_DATA_PATH, index=False)

    print("✅ Dataset tersimpan!")
    print(f"   Clustering : {data_for_clustering.shape} → {CLUSTERING_DATA_PATH}")
    print(f"   ARM        : {data_for_arm.shape}        → {ARM_DATA_PATH}")


# ════════════════════════════════════════════════════════════
# 10. PIPELINE UTAMA
# ════════════════════════════════════════════════════════════

def run_preprocessing(raw_path: str) -> tuple:
    """
    Jalankan seluruh pipeline Phase 1 secara berurutan.

    Returns:
        (df_encoded, df_clean) — dua versi dataset yang sudah bersih
    """
    print("=" * 55)
    print("  PHASE 1 — Data Understanding & Preprocessing")
    print("=" * 55)

    df = load_data(raw_path)

    print("\n[1/7] Validasi data...")
    df = validate_data(df)

    print("\n[2/7] Feature engineering...")
    df = engineer_features(df)

    print("\n[3/7] Drop kolom tidak relevan...")
    df_clean = drop_irrelevant_columns(df)

    print("\n[4/7] Binning...")
    df_clean = bin_features(df_clean)

    print("\n[5/7] Encoding...")
    df_encoded = encode_features(df_clean)

    print("\n[6/7] Normalisasi...")
    df_encoded = normalize_features(df_encoded)

    print("\n[7/7] Feature selection & save datasets...")
    feature_selection(df_encoded)
    save_datasets(df_encoded, df_clean)

    print("\n" + "=" * 55)
    print("  ✅ PHASE 1 SELESAI")
    print("=" * 55)

    return df_encoded, df_clean
