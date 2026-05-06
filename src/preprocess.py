"""
preprocess.py — Phase 1: Data Understanding & Preprocessing

Tahap ini kita melakukan import berbagai libraries yang diperlukan
dari phase 1 sampai phase 4, lalu menjalankan seluruh pipeline
preprocessing sesuai notebook PROJECT_Data_Mining.

Cara running:
    python src/preprocess.py
    
Atau import fungsinya dari notebook:
    from src.preprocess import run_preprocessing
    df_encoded, df_clean = run_preprocessing('data/Comprehensive_Banking_Database.csv')
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif

DATA_DIR   = 'data'
OUTPUT_DIR = 'outputs'
RAW_PATH   = os.path.join(DATA_DIR, 'Comprehensive_Banking_Database.csv')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


PHASE1_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'phase1'
)
os.makedirs(PHASE1_OUTPUT_DIR, exist_ok=True)


# LOAD DATASET
# Tahap ini kita me-load dataset yang akan dipakai dan kita
# kerjakan atau kita "data mine" dari datanya.

def load_data(path):
    data_set = pd.read_csv(path)
    print(f"Dataset loaded: {data_set.shape[0]:,} rows x {data_set.shape[1]} columns")
    return data_set


# EXPLORATORY DATA ANALYSIS
# Tahapan ini bertujuan untuk memahami struktur data, distribusi
# variabel utama, pola kategori, hubungan antar fitur numerik,
# serta indikasi awal anomali. Hasil EDA ini akan menjadi dasar
# untuk keputusan feature engineering, binning, dan feature
# selection pada tahap selanjutnya.

def run_eda(data_set, save_plots=True):
    print("\n-- Head --"); print(data_set.head())
    print("\n-- Tail --"); print(data_set.tail())
    print("\n-- Info --"); data_set.info()
    print("\n-- Describe --"); print(data_set.describe())

    # Visualizing Numerical Features Distribution
    # Kita tidak perlu melakukan visualisasi terhadap beberapa fitur
    # numerical karena sudah merupakan data dummy.
    num_features = [c for c in [
        'Age','Account Balance','Transaction Amount',
        'Account Balance After Transaction','Loan Amount',
        'Interest Rate','Loan Term','Credit Limit',
        'Credit Card Balance','Minimum Payment Due','Rewards Points'
    ] if c in data_set.columns]

    # Histogram
    fig, axes = plt.subplots(len(num_features), 1, figsize=(10, 4*len(num_features)))
    if len(num_features) == 1: axes = [axes]
    for ax, col in zip(axes, num_features):
        sns.histplot(data_set[col], kde=True, bins=30, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col); ax.set_ylabel("Count")
    plt.tight_layout()
    if save_plots: plt.savefig(os.path.join(PHASE1_OUTPUT_DIR,'eda_numerical_dist.png'), dpi=150)
    plt.show()

    # Box Plot
    fig, axes = plt.subplots(len(num_features), 1, figsize=(10, 4 * len(num_features)))
    if len(num_features) == 1:
        axes = [axes]

    for ax, col in zip(axes, num_features):
        sns.boxplot(x=data_set[col], ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    plt.tight_layout()
    if save_plots: plt.savefig(os.path.join(PHASE1_OUTPUT_DIR,'eda_numerical_boxplot.png'), dpi=150)
    plt.show()

    # Visualizing Categorical Feature Distribution
    # Kita tidak perlu melakukan visualisasi ke seluruh tipe category
    # karena ada yang berupa PII.
    cat_features = [c for c in [
        'Gender','Account Type','Transaction Type','Loan Type',
        'Loan Status','Card Type','Feedback Type','Resolution Status'
    ] if c in data_set.columns]

    fig, axes = plt.subplots(len(cat_features), 1, figsize=(10, 4*len(cat_features)))
    if len(cat_features) == 1: axes = [axes]
    for ax, col in zip(axes, cat_features):
        sns.countplot(data=data_set, x=col, order=data_set[col].value_counts().index, ax=ax)
        ax.set_title(f"Count Plot of {col}")
        ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    if save_plots: plt.savefig(os.path.join(PHASE1_OUTPUT_DIR,'eda_categorical_dist.png'), dpi=150)
    plt.show()

    # Scatter Plot Financial Feature
    # Scatter plot dipakai untuk melihat hubungan perilaku finansial antar
    # fitur dan menangkap pola yang tidak selalu terlihat dari ringkasan statistik.
    # Hasil dari visualisasi tersebut tidak memberikan informasi yang berguna
    # karena tidak terlihat relasinya.
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=data_set, x='Account Balance', y='Transaction Amount',
                    hue='Transaction Type', alpha=0.7)
    plt.title("Account Balance vs Transaction Amount"); plt.tight_layout()
    if save_plots: plt.savefig(os.path.join(PHASE1_OUTPUT_DIR,'scatter_balance_transaction.png'), dpi=150)
    plt.show()

    plt.figure(figsize=(10,6))
    sns.scatterplot(data=data_set, x='Credit Limit', y='Credit Card Balance',
                    hue='Loan Status', alpha=0.7)
    plt.title("Credit Limit vs Credit Card Balance"); plt.tight_layout()
    if save_plots: plt.savefig(os.path.join(PHASE1_OUTPUT_DIR,'scatter_credit.png'), dpi=150)
    plt.show()

    # Checking Null Row
    # Tidak terdapat row yang datanya tidak lengkap atau null.
    print("\n-- Null Check --"); print(data_set.isnull().sum())

    # Checking Duplicated Row
    # Tidak terdapat row yang datanya duplikasi dari row lain.
    total_duplicates = data_set.duplicated().sum()
    print(f"\n-- Duplicate Check --\nTotal baris duplikat: {total_duplicates}")

    # Checking Raw Correlation
    # Dapat dilihat bahwa korelasi sempurna antara Credit Card Balance dengan
    # Minimum Payment Due berarti kita nantinya harus mengeliminasi salah satu
    # fitur agar tidak ada multikolinearitas.
    corr_cols = [c for c in [
        'Age','Account Balance','Transaction Amount',
        'Account Balance After Transaction','Loan Amount',
        'Interest Rate','Loan Term','Credit Limit',
        'Credit Card Balance','Minimum Payment Due','Rewards Points','Anomaly'
    ] if c in data_set.columns]

    plt.figure(figsize=(10,7))
    sns.heatmap(data_set[corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation Matrix - Raw Data (Before Cleaning)'); plt.tight_layout()
    if save_plots: plt.savefig(os.path.join(PHASE1_OUTPUT_DIR,'correlation_raw.png'), dpi=150)
    plt.show()

    # Checking Anomaly Ratio
    # Terlihat bahwa dataset ini memiliki anomaly -1 yang lebih sedikit.
    print("\n-- Anomaly Ratio --"); print(data_set['Anomaly'].value_counts())


# DATA VALIDATION

def validate_data(data_set):
    data_set = data_set.copy()

    # Checking Balance Consistency
    print("\n-- Transaction Type --"); print(data_set['Transaction Type'].value_counts())

    def check_balance_logic(row):
        diff = row['Account Balance After Transaction'] - row['Account Balance']
        t_type, amount = row['Transaction Type'], row['Transaction Amount']
        if t_type == 'Deposit': return abs(diff - amount)
        elif t_type == 'Withdrawal': return abs(diff + amount)
        return None  # Transfer dikecualikan — arah tidak dapat ditentukan dari data

    data_set['balance_check'] = data_set.apply(check_balance_logic, axis=1)
    print("=== Inkonsistensi per Transaction Type ===")
    print(data_set.groupby('Transaction Type')['balance_check'].agg(
        Total='count',
        Inkonsisten=lambda x: (x > 1).sum(),
        Pct_Inkonsisten=lambda x: f"{(x>1).mean()*100:.1f}%"
    ))

    # Cross-check: apakah baris "inkonsisten" = baris yang Anomaly = -1 ?
    data_set['is_inconsistent'] = data_set['balance_check'] > 1
    cross_check = pd.crosstab(data_set['is_inconsistent'], data_set['Anomaly'],
                              rownames=['Inkonsisten'], colnames=['Anomaly Label'])
    print(cross_check)
    ir = data_set[data_set['is_inconsistent']]
    print(f"\nDari {len(ir)} baris inkonsisten:")
    print(f"  Yang Anomaly=-1 : {(ir['Anomaly']==-1).sum()}")
    print(f"  Yang Anomaly= 1 : {(ir['Anomaly']==1).sum()}")
    # Ditemukan 1.703 baris (34%) inkonsisten. Setelah cross-check:
    # - 1.703 inkonsisten → hanya 97 yang Anomaly=-1 (5.7%)
    # - 3.297 konsisten → 203 yang Anomaly=-1 (6.2%)
    # Inkonsistensi tidak berkorelasi dengan anomali → TIDAK dihapus.
    data_set.drop(columns=['balance_check','is_inconsistent'], inplace=True)

    # Checking Age
    # Tidak ditemukan nilai usia < 17 tahun. Range 18-69, aman. Tidak ada yang dihapus.
    print("\n-- Age Validation --"); print(data_set['Age'].describe())
    print(f"Baris Age < 17: {len(data_set[data_set['Age'] < 17])}")

    # Checking Overlimit CC
    # 855 nasabah (17.1%) overlimit. Bukan error — bisa karena bunga/denda.
    # Dipertahankan, akan menjadi sinyal di Phase 4.
    cc_over = data_set[data_set['Credit Card Balance'] > data_set['Credit Limit']]
    print(f"\n-- CC Over Limit --\nNasabah overlimit: {len(cc_over)}")

    # Checking Interest Rate
    # Tidak ada Interest Rate di luar rentang 1-10%. Aman.
    print("\n-- Interest Rate --"); print(data_set['Interest Rate'].describe())
    invalid_rate = data_set[(data_set['Interest Rate'] < 0) | (data_set['Interest Rate'] > 100)]
    print(f"Interest Rate tidak valid: {len(invalid_rate)}")

    return data_set


# FEATURE ENGINEERING
# Pada tahap ini kita akan melakukan modifikasi terhadap fitur
# yang ada, seperti merubah, membuat yang baru, atau
# mengeliminasi yang tidak diperlukan.
def engineer_features(data_set):
    data_set = data_set.copy()

    date_cols = [
        'Date Of Account Opening','Last Transaction Date',
        'Transaction Date','Approval/Rejection Date',
        'Payment Due Date','Last Credit Card Payment Date',
        'Feedback Date','Resolution Date'
    ]
    for col in date_cols:
        data_set[col] = pd.to_datetime(data_set[col], errors='coerce')

    print("=== Validasi Parse Tanggal ===")
    for col in date_cols:
        n = data_set[col].isnull().sum()
        print(f"  {col}: {'Aman' if n==0 else f'WARN {n} gagal'}")

    # Fitur tanggal tidak dapat dipakai secara langsung.
    # Reference date diambil dari tanggal maksimum di seluruh dataset
    # agar konsisten dengan rentang waktu data itu sendiri.
    reference_date = data_set[date_cols].max().max()
    print(f"\nReference date: {reference_date.date()}")

    # Account_Age_Years: mengukur senioritas/loyalitas nasabah
    # Days_Since_Last_Transaction: mengukur tingkat keaktifan transaksi
    data_set['Account_Age_Years'] = (
        (reference_date - data_set['Date Of Account Opening']).dt.days / 365
    ).round(1)
    data_set['Days_Since_Last_Transaction'] = (
        (reference_date - data_set['Last Transaction Date']).dt.days
    )
    print("\n=== Fitur Temporal Baru ===")
    print(data_set[['Account_Age_Years','Days_Since_Last_Transaction']].describe().round(2))

    # CC_Utilization: rasio penggunaan kartu kredit vs limit
    #   (> 1.0 berarti overlimit)
    # Transaction_to_Balance_Ratio: besar transaksi relatif terhadap saldo
    data_set['CC_Utilization'] = (
        data_set['Credit Card Balance'] / data_set['Credit Limit']
    )
    data_set['Transaction_to_Balance_Ratio'] = (
        data_set['Transaction Amount'] / data_set['Account Balance'].replace(0, np.nan)
    )
    print("\n=== Rasio Finansial Baru ===")
    print(data_set[['CC_Utilization','Transaction_to_Balance_Ratio']].describe().round(2))

    return data_set, date_cols


# EARLY FEATURE SELECTION (DROP KOLOM)
# Pada tahap ini, dipilih fitur yang akan dipakai pada fase
# selanjutnya. Dikatakan "early" karena sudah pasti tidak
# digunakan dari awal untuk meningkatkan efisiensi.
#
# Justifikasi drop:
# - PII (First Name, Last Name, Address, Email, Contact Number):
#   tidak informatif untuk mining, risiko privasi data.
# - Surrogate Key (Customer ID, TransactionID, Loan ID, CardID,
#   Feedback ID): penomoran otomatis tanpa makna semantik.
#   Branch ID DIPERTAHANKAN karena merepresentasikan lokasi cabang.
# - Date Data: sudah diekstrak ke Account_Age_Years dan
#   Days_Since_Last_Transaction. Menyimpan keduanya = redundan.

def drop_irrelevant_columns(data_set, date_cols):
    cols_to_drop = [c for c in [
        'Customer ID','First Name','Last Name','Address','Email',
        'Contact Number','TransactionID','Loan ID','CardID','Feedback ID',
        *date_cols
    ] if c in data_set.columns]

    data_clean = data_set.drop(columns=cols_to_drop)
    print(f"\nKolom tersisa: {data_clean.shape[1]}")
    print(data_clean.columns.tolist())
    return data_clean


# BINNING
# Pada tahap ini kita mengubah data kontinu ke data kategorikal
# yang diperlukan pada fase-fase selanjutnya (terutama ARM/Apriori).
# Menggunakan quantile-based binning (qcut) agar setiap bin memiliki
# jumlah data yang relatif seimbang.

def _safe_qcut(series, labels):
    ranked = series.rank(method='first')
    try:
        return pd.qcut(ranked, q=len(labels), labels=labels, precision=2)
    except ValueError:
        n_bins = min(len(labels), ranked.nunique())
        if n_bins < 2:
            return pd.Series([labels[0]] * len(series), index=series.index)
        return pd.qcut(ranked, q=n_bins, labels=labels[:n_bins], precision=2)


def bin_features(data_clean):
    data_clean = data_clean.copy()

    data_clean['Age_Group']      = _safe_qcut(data_clean['Age'],
                                              ['Young','Adult','Middle-aged','Senior'])
    data_clean['Balance_Bucket'] = _safe_qcut(data_clean['Account Balance'],
                                              ['Low','Lower-Mid','Upper-Mid','High'])
    data_clean['Transaction_Size'] = _safe_qcut(data_clean['Transaction Amount'],
                                                ['Small','Medium','Large','Very Large'])
    data_clean['Loan_Size']      = _safe_qcut(data_clean['Loan Amount'],
                                              ['Small','Medium','Large','Very Large'])
    data_clean['Rate_Category']  = _safe_qcut(data_clean['Interest Rate'],
                                              ['Low','Moderate','High'])
    # CC_Utilization menggunakan domain-based cut karena ada threshold
    # industri yang bermakna (30%, 70%, 100%)
    data_clean['CC_Utilization_Category'] = pd.cut(
        data_clean['CC_Utilization'],
        bins=[-np.inf, 0.09, 0.29, 0.49, 0.70, np.inf],
        labels=['Excellent','Good','Moderate','High','Very High'],
        include_lowest=True
    )

    print("Binning selesai!")
    for col in ['Age_Group','Balance_Bucket','CC_Utilization_Category']:
        print(f"\n{col}:"); print(data_clean[col].value_counts(dropna=False))

    return data_clean


# ENCODING DATA
# Pada tahap ini mengubah input data menjadi input numerik.
# Kita memilih Label Encoding dan One-Hot Encoding karena ada
# data yang memiliki lebih dari dua kategori. One-Hot Encoding
# digunakan untuk menghindari bias pada model yang menganggap
# 2 > 1 (asumsi ordinalitas yang tidak tepat).

def encode_features(data_clean):
    data_clean = data_clean.copy()
    le = LabelEncoder()

    # Label encoding untuk variabel biner/ordinal
    data_clean['Loan_Status_Encoded']       = le.fit_transform(data_clean['Loan Status'])
    data_clean['Account_Type_Encoded']      = le.fit_transform(data_clean['Account Type'])
    data_clean['Resolution_Status_Encoded'] = le.fit_transform(data_clean['Resolution Status'])
    print("Loan Status classes:", le.classes_)

    # One-Hot Encoding untuk variabel nominal tanpa urutan
    nominal_cols = ['Gender','Account Type','Transaction Type',
                    'Loan Type','Card Type','Feedback Type','Resolution Status']
    data_encoded = pd.get_dummies(data_clean, columns=nominal_cols, drop_first=False)
    print(f"Shape setelah encoding: {data_encoded.shape}")
    return data_encoded


# SCALING AND NORMALIZING
# Pada tahap ini kita melakukan scaling dan normalisasi terhadap
# data numerikal. Kita memilih MinMaxScaler agar data berada pada
# range [0,1] untuk menghindari bias — berbeda dengan StandardScaler
# yang tidak menjamin batas atas/bawah tertentu.

def normalize_features(data_encoded):
    data_encoded = data_encoded.copy()
    numeric_cols = [c for c in [
        'Age','Account Balance','Transaction Amount',
        'Account Balance After Transaction','Loan Amount',
        'Interest Rate','Loan Term','Credit Limit',
        'Credit Card Balance','Minimum Payment Due',
        'Rewards Points','Account_Age_Years',
        'Days_Since_Last_Transaction','CC_Utilization',
        'Transaction_to_Balance_Ratio'
    ] if c in data_encoded.columns]

    scaler = MinMaxScaler()
    data_encoded[numeric_cols] = scaler.fit_transform(data_encoded[numeric_cols])
    print("Normalisasi selesai!")
    print(data_encoded[numeric_cols].describe().round(3))
    return data_encoded


# FINAL FEATURE SELECTION
# Pada tahap ini kita mulai memilih fitur-fitur yang akan dipakai
# oleh model kita menggunakan dua metode:
# 1. Correlation Matrix — deteksi multikolinearitas
# 2. Mutual Information — ukur informativeness berbasis entropi

def feature_selection(data_encoded, save_plots=True):
    corr_cols = [c for c in [
        'Age','Account Balance','Transaction Amount',
        'Account Balance After Transaction','Loan Amount',
        'Interest Rate','Loan Term','Credit Limit',
        'Credit Card Balance','Minimum Payment Due',
        'Rewards Points','Account_Age_Years',
        'Days_Since_Last_Transaction','CC_Utilization',
        'Transaction_to_Balance_Ratio'
    ] if c in data_encoded.columns]

    # Correlation-Matrix Method
    # Menggunakan correlation-matrix untuk melihat hubungan antar feature
    # dan mendeteksi multikolinearitas. Dari hasil korelasi, ditemukan
    # korelasi sempurna antara Credit Card Balance dan Minimum Payment Due
    # karena minimum payment dihitung sebagai persentase tetap dari saldo.
    corr_matrix = data_encoded[corr_cols].corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
    plt.title('Correlation Matrix - Fitur Numerik'); plt.tight_layout()
    if save_plots: plt.savefig(os.path.join(PHASE1_OUTPUT_DIR,'correlation_matrix.png'), dpi=150)
    plt.show()

    print("Pasangan fitur highly correlated (>0.85):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i,j]) > 0.85:
                print(f"  {corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_matrix.iloc[i,j]:.3f}")

    # Mutual Information Method
    # Mengukur seberapa besar pengaruh yang diberikan sebuah fitur untuk
    # membedakan record normal vs anomali. MI = 0 berarti tidak ada pengaruh.
    # Hasil: sebagian besar fitur mendekati MI = 0, yang menandakan anomali
    # tidak dapat dideteksi secara univariat — baru terlihat saat fitur
    # dikombinasikan (multivariat, dilakukan di Phase 4).
    target   = data_encoded['Anomaly']
    features = data_encoded[corr_cols]
    mi_scores = mutual_info_classif(features, target, random_state=42)
    mi_df = pd.DataFrame({'Feature': corr_cols, 'Mutual_Information': mi_scores}
                         ).sort_values('Mutual_Information', ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(data=mi_df, x='Mutual_Information', y='Feature', palette='viridis')
    plt.title('Mutual Information Score per Fitur\n(Entropy-based Feature Selection)')
    plt.xlabel('Mutual Information Score'); plt.tight_layout()
    if save_plots: plt.savefig(os.path.join(PHASE1_OUTPUT_DIR,'mutual_information.png'), dpi=150)
    plt.show()
    print(mi_df.to_string(index=False))


# SELECTING FEATURES & SAVE DATASETS
# Fitur dipilih berdasarkan kemampuannya menggambarkan finance
# profile dan behaviour nasabah.
#
# Fitur yang TIDAK dipakai di clustering:
# - Account Balance After Transaction: turunan Balance + Transaction
# - Credit Limit: sudah tercermin dalam CC_Utilization
# - Credit Card Balance: sudah tercermin dalam CC_Utilization
# - Minimum Payment Due: korelasi sempurna dengan CC Balance

def save_datasets(data_encoded, data_clean):
    # Dataset untuk Clustering (Phase 2)
    # Anomaly disimpan sebagai referensi saja — BUKAN input clustering
    clustering_cols = [c for c in [
        'Age',                          # segmen usia nasabah
        'Account_Age_Years',            # senioritas/loyalitas nasabah
        'Account Balance',              # kekayaan likuid nasabah
        'Loan Amount',                  # eksposur utang
        'Interest Rate',                # profil risiko kredit
        'Loan Term',                    # jangka waktu komitmen kredit
        'Transaction Amount',           # besaran transaksi
        'Transaction_to_Balance_Ratio', # transaksi relatif terhadap kekayaan
        'CC_Utilization',               # utilisasi kartu kredit
        'Rewards Points',               # engagement/loyalitas program bank
        'Days_Since_Last_Transaction',  # history penggunaan terakhir
        'Anomaly'                       # referensi Phase 4 — bukan input model
    ] if c in data_encoded.columns]

    data_encoded[clustering_cols].to_csv(
        os.path.join(DATA_DIR, 'dataset_clustering.csv'), index=False
    )

    # Dataset untuk ARM (Phase 3)
    arm_cols = [c for c in [
        'Age_Group',               # segmen usia → pola berbeda per generasi
        'Gender',                  # demografi dasar
        'Account Type',            # jenis akun (Savings/Current/dll)
        'Balance_Bucket',          # kondisi finansial dalam kategori
        'Transaction_Size',        # ukuran transaksi dalam kategori
        'Transaction Type',        # jenis transaksi (Deposit/Withdrawal/Transfer)
        'Loan_Size',               # ukuran pinjaman dalam kategori
        'Loan Type',               # jenis pinjaman (Personal/Mortgage/dll)
        'Loan Status',             # hasil pengajuan (Approved/Rejected)
        'Rate_Category',           # kategori bunga
        'Card Type',               # jenis kartu kredit
        'CC_Utilization_Category', # kategori penggunaan kartu
        'Feedback Type',           # tipe feedback nasabah
        'Resolution Status'        # status penyelesaian keluhan
    ] if c in data_clean.columns]

    data_clean[arm_cols].to_csv(
        os.path.join(DATA_DIR, 'dataset_arm.csv'), index=False
    )

    print("Dataset bersih tersimpan!")
    print(f"Clustering dataset: {data_encoded[clustering_cols].shape}")
    print(f"ARM dataset: {data_clean[arm_cols].shape}")


# MAIN PIPELINE

def run_preprocessing(raw_path=RAW_PATH):
    print("=" * 55)
    print("  PHASE 1 - Data Understanding & Preprocessing")
    print("=" * 55)

    data_set = load_data(raw_path)
    run_eda(data_set)
    data_set = validate_data(data_set)
    data_set, date_cols = engineer_features(data_set)
    data_clean = drop_irrelevant_columns(data_set, date_cols)
    data_clean = bin_features(data_clean)
    data_encoded = encode_features(data_clean)
    data_encoded = normalize_features(data_encoded)
    feature_selection(data_encoded)
    save_datasets(data_encoded, data_clean)

    print("\n" + "=" * 55)
    print("  PHASE 1 SELESAI")
    print("=" * 55)
    return data_encoded, data_clean


if __name__ == '__main__':
    run_preprocessing(RAW_PATH)
