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
import sys
for _stream in (sys.stdout, sys.stderr):   # konsol Windows cp1252 → paksa UTF-8
    try:
        _stream.reconfigure(encoding='utf-8')
    except Exception:
        pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # backend non-interaktif → plot disimpan ke file, pipeline tidak nge-freeze
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import mutual_info_classif

# Semua path absolut relatif terhadap root project (folder di atas src/)
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
RAW_PATH   = os.path.join(DATA_DIR, 'Comprehensive_Banking_Database.csv')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHASE1_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'phase1')
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
    plt.close()

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
    plt.close()

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
    plt.close()

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
    plt.close()

    plt.figure(figsize=(10,6))
    sns.scatterplot(data=data_set, x='Credit Limit', y='Credit Card Balance',
                    hue='Loan Status', alpha=0.7)
    plt.title("Credit Limit vs Credit Card Balance"); plt.tight_layout()
    if save_plots: plt.savefig(os.path.join(PHASE1_OUTPUT_DIR,'scatter_credit.png'), dpi=150)
    plt.close()

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

    raw_corr = data_set[corr_cols].corr()
    plt.figure(figsize=(10,7))
    sns.heatmap(raw_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation Matrix - Raw Data (Before Cleaning)'); plt.tight_layout()
    if save_plots: plt.savefig(os.path.join(PHASE1_OUTPUT_DIR,'correlation_raw.png'), dpi=150)
    plt.close()

    # TEMUAN PENTING — STRUKTUR DATA
    # Kita hitung berapa pasang fitur numerik yang benar-benar berkorelasi.
    # Hasilnya: dari sekian banyak pasang, HANYA segelintir yang |r|>0.1 dan
    # itu pun pasangan turunan (Account Balance vs Balance After Transaction,
    # Credit Card Balance vs Minimum Payment Due yang korelasinya ~1.0).
    # Mayoritas fitur INDEPENDEN satu sama lain dan terdistribusi mendekati
    # uniform (skew ~0). Konsekuensinya untuk fase berikutnya:
    #   1. PCA tidak akan efektif — tanpa redundansi antar fitur, setiap
    #      komponen hanya menangkap ~1/n variance (lihat Phase 2).
    #   2. Clustering pada fitur kontinu MENTAH tidak akan menemukan cluster
    #      alami — solusinya adalah feature engineering berbasis RASIO
    #      perilaku yang justru memiliki struktur (skew tinggi).
    num_only = [c for c in corr_cols if c != 'Anomaly']
    sub = raw_corr.loc[num_only, num_only]
    pairs = [(num_only[i], num_only[j], sub.iloc[i, j])
             for i in range(len(num_only)) for j in range(i+1, len(num_only))
             if abs(sub.iloc[i, j]) > 0.1]
    total_pairs = len(num_only) * (len(num_only) - 1) // 2
    print(f"\n-- Struktur Korelasi --")
    print(f"Pasangan fitur dengan |r|>0.1 : {len(pairs)} dari {total_pairs} "
          f"({len(pairs)/total_pairs*100:.0f}%)")
    for a, b, v in pairs:
        print(f"   {a} <-> {b}: {v:+.3f}")
    print("Kesimpulan: mayoritas fitur INDEPENDEN & near-uniform → "
          "PCA tidak efektif, clustering harus pakai fitur rasio perilaku (lihat Phase 2).")

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

    # RASIO PERILAKU — ini fitur paling penting untuk clustering (Phase 2).
    # Berbeda dari fitur mentah yang near-uniform & independen, rasio antara
    # dua fitur uniform menghasilkan distribusi BERSTRUKTUR (skew tinggi),
    # sehingga clustering bisa menemukan segmen perilaku yang nyata.
    #
    # CC_Utilization        : saldo kartu / limit → tekanan kredit (>1 = over-limit)
    # Transaction_to_Balance: transaksi / saldo   → intensitas likuiditas
    # Loan_to_Balance       : pinjaman / saldo     → leverage utang
    data_set['CC_Utilization'] = (
        data_set['Credit Card Balance'] / data_set['Credit Limit']
    )
    data_set['Transaction_to_Balance_Ratio'] = (
        data_set['Transaction Amount'] / data_set['Account Balance'].replace(0, np.nan)
    )
    data_set['Loan_to_Balance_Ratio'] = (
        data_set['Loan Amount'] / data_set['Account Balance'].replace(0, np.nan)
    )
    # Tangani pembagian-nol/inf jika ada saldo 0
    ratio_cols = ['CC_Utilization', 'Transaction_to_Balance_Ratio', 'Loan_to_Balance_Ratio']
    data_set[ratio_cols] = data_set[ratio_cols].replace([np.inf, -np.inf], np.nan)
    for c in ratio_cols:
        data_set[c] = data_set[c].fillna(data_set[c].median())

    print("\n=== Rasio Perilaku Baru (skew tinggi = ada struktur) ===")
    print(data_set[ratio_cols].describe().round(2))
    print("Skewness:", {c: round(data_set[c].skew(), 2) for c in ratio_cols})

    return data_set, date_cols

# OUTLIER DETECTION (Pre-Scaling)
# Dilakukan sebelum scaling untuk menginformasikan keputusan
# transformasi. Berbeda dengan Phase 4 yang bertujuan knowledge
# discovery, outlier detection di sini bertujuan memastikan
# proses scaling tidak terdistorsi oleh nilai ekstrem.
#
# Keputusan yang diambil:
# - Outlier TIDAK dihapus karena bisa jadi sinyal anomali
#   yang akan diinvestigasi di Phase 4
# - Outlier digunakan untuk menentukan scaler yang tepat:
#   RobustScaler untuk fitur dengan outlier ekstrem,
#   MinMaxScaler untuk fitur yang distribusinya normal

def detect_outliers_prescaling(data_clean, save_plots=True):
    numeric_cols = [c for c in [
        'Account Balance', 'Transaction Amount', 'Loan Amount',
        'Credit Card Balance', 'Rewards Points',
        'CC_Utilization', 'Transaction_to_Balance_Ratio',
        'Age', 'Interest Rate', 'Loan Term', 'Account_Age_Years',
        'Days_Since_Last_Transaction'
    ] if c in data_clean.columns]

    print("=== Outlier Detection (IQR Method) ===")
    outlier_summary = []

    for col in numeric_cols:
        Q1  = data_clean[col].quantile(0.25)
        Q3  = data_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        n_outliers = ((data_clean[col] < lower) | (data_clean[col] > upper)).sum()
        pct        = n_outliers / len(data_clean) * 100
        skewness   = data_clean[col].skew()

        # Tentukan rekomendasi scaler berdasarkan outlier + skewness
        if pct > 5 or abs(skewness) > 1:
            recommendation = 'RobustScaler'
        else:
            recommendation = 'MinMaxScaler'

        outlier_summary.append({
            'Feature'       : col,
            'Skewness'      : round(skewness, 3),
            'N_Outliers'    : n_outliers,
            'Pct_Outliers'  : round(pct, 1),
            'Lower_Bound'   : round(lower, 2),
            'Upper_Bound'   : round(upper, 2),
            'Scaler'        : recommendation
        })

    summary_df = pd.DataFrame(outlier_summary).sort_values('Pct_Outliers', ascending=False)
    print(summary_df.to_string(index=False))

    # Visualisasi boxplot per fitur
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 3*len(numeric_cols)))
    if len(numeric_cols) == 1: axes = [axes]

    for ax, col in zip(axes, numeric_cols):
        rec = summary_df[summary_df['Feature'] == col]['Scaler'].values[0]
        color = '#e74c3c' if rec == 'RobustScaler' else '#2ecc71'
        sns.boxplot(x=data_clean[col], ax=ax, color=color)
        ax.set_title(f"{col}  |  Skew={data_clean[col].skew():.2f}  →  {rec}",
                     fontsize=10)

    plt.suptitle('Pre-Scaling Outlier Detection\n(Merah = RobustScaler, Hijau = MinMaxScaler)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(PHASE1_OUTPUT_DIR, 'prescaling_outlier_detection.png'),
                    dpi=150, bbox_inches='tight')
    plt.close()

    # Pisahkan fitur berdasarkan rekomendasi scaler
    robust_cols = summary_df[summary_df['Scaler'] == 'RobustScaler']['Feature'].tolist()
    minmax_cols = summary_df[summary_df['Scaler'] == 'MinMaxScaler']['Feature'].tolist()

    print(f"\nRekomendasi scaler:")
    print(f"  RobustScaler ({len(robust_cols)}) : {robust_cols}")
    print(f"  MinMaxScaler ({len(minmax_cols)}) : {minmax_cols}")

    return robust_cols, minmax_cols

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
# Kita memakai AMBANG DOMAIN TETAP (fixed thresholds), BUKAN quantile
# (equal-frequency) maupun equal-width. Alasannya:
#   - Quantile/equal-width bersifat data-dependent: batas bin bergeser bila
#     data berubah, dan kategori tidak punya makna intrinsik ("Senior" bisa
#     berarti usia berbeda tiap dataset).
#   - Ambang domain tetap bersifat REPRODUCIBLE & INTERPRETABLE: "Senior"
#     selalu 65+, "Over-Limit" selalu >100% limit, apa pun datanya.
#
# REFERENSI AMBANG:
#   * Umur → tahap hidup finansial (life-stage). Dasar teori: Life-Cycle
#     Hypothesis (Modigliani & Brumberg, 1954) — pola pinjam saat muda,
#     menabung saat paruh baya, dis-saving saat tua. Bukti empiris:
#     Agarwal, Driscoll, Gabaix & Laibson (2009), "The Age of Reason:
#     Financial Decisions over the Life-Cycle", Brookings Papers — kecakapan
#     finansial berbentuk hump-shaped, puncak ~usia 53; kelompok muda & tua
#     membayar bunga/fee lebih tinggi. Maka umur dipotong di transisi
#     tahap hidup (awal karier, mapan, pra-pensiun, pensiun), bukan kuantil.
#   * Utilisasi kartu → pedoman credit-scoring (FICO): idealnya < 30%;
#     30–70% sedang, 70–100% tinggi, >100% over-limit (sinyal risiko nyata).
#   * Suku bunga → tier prime (rendah) / standar / subprime (tinggi).
#   * Saldo tabungan → ambang minimum-balance perbankan ritel: ~$1.500 adalah
#     batas saldo minimum paling umum untuk membebaskan biaya bulanan di bank
#     besar, dan ~$5.000 adalah ambang saldo rata-rata alternatif (mis. Chase,
#     Bank of America). Di bawah $1.500 = rawan biaya; ≥$5.000 = mendekati
#     tier affluent (~$10rb+). Jadi: Below-Minimum / Mass-Market / Comfortable.
#   * Pinjaman → bracket ukuran pinjaman konsumen: pinjaman tanpa agunan
#     biasanya dibatasi ~$35rb–$50rb; <$5rb = small-dollar; di atas ~$35rb
#     umumnya beragunan (auto/mortgage). Jadi: Small/Medium/Large/Very Large.
#   * Transaksi → tidak ada ambang regulatori di bawah $10.000 (batas CTR/
#     Currency Transaction Report justru $10rb, di ATAS nilai maksimum data
#     ini ~$5rb). Maka dipakai tier interpretable harian/besar/sangat besar
#     (ini fitur paling lemah secara domain — diakui jujur).

# Ambang umur (tahun, batas atas inklusif) → tahap hidup finansial
AGE_BINS   = [17, 24, 34, 49, 64, np.inf]
AGE_LABELS = ['Young Adult',     # 18–24: pelajar / awal kerja, aset rendah
              'Early Career',    # 25–34: pembentukan keluarga, mulai KPR/kredit
              'Established',      # 35–49: pendapatan menanjak, mendekati puncak
              'Pre-Retirement',  # 50–64: puncak kekayaan, deleveraging
              'Senior']          # 65+  : pensiun, pendapatan tetap


def bin_features(data_clean):
    data_clean = data_clean.copy()

    # Umur → tahap hidup finansial (lihat referensi paper di atas)
    data_clean['Age_Group'] = pd.cut(
        data_clean['Age'], bins=AGE_BINS, labels=AGE_LABELS)

    # Saldo → ambang minimum-balance ritel ($1.500 fee-waiver, $5.000 avg-balance)
    data_clean['Balance_Bucket'] = pd.cut(
        data_clean['Account Balance'],
        bins=[-np.inf, 1500, 5000, np.inf],
        labels=['Below-Minimum', 'Mass-Market', 'Comfortable'])

    # Transaksi → tier interpretable (tak ada ambang regulatori < $10rb/CTR)
    data_clean['Transaction_Size'] = pd.cut(
        data_clean['Transaction Amount'],
        bins=[-np.inf, 1000, 3000, np.inf],
        labels=['Everyday', 'Large', 'Very Large'])

    # Pinjaman → bracket ukuran pinjaman konsumen (cap tanpa agunan ~$35rb)
    data_clean['Loan_Size'] = pd.cut(
        data_clean['Loan Amount'],
        bins=[-np.inf, 5000, 20000, 35000, np.inf],
        labels=['Small', 'Medium', 'Large', 'Very Large'])

    # Suku bunga → tier prime / standar / subprime
    data_clean['Rate_Category'] = pd.cut(
        data_clean['Interest Rate'],
        bins=[-np.inf, 4, 7, np.inf],
        labels=['Low', 'Moderate', 'High'])

    # Utilisasi kartu kredit → pedoman credit-scoring FICO (30/70/100%).
    # 'Over-Limit' (>100%) wajib ada agar sinyal tekanan kredit tidak hilang.
    data_clean['CC_Utilization_Category'] = pd.cut(
        data_clean['CC_Utilization'],
        bins=[-np.inf, 0.30, 0.70, 1.00, np.inf],
        labels=['Low', 'Moderate', 'High', 'Over-Limit'],
        include_lowest=True)

    print("Binning (ambang domain tetap) selesai!")
    for col in ['Age_Group', 'Balance_Bucket', 'Transaction_Size',
                'Loan_Size', 'Rate_Category', 'CC_Utilization_Category']:
        vc = data_clean[col].value_counts(dropna=False).reindex(
            data_clean[col].cat.categories)
        print(f"\n{col}:"); print(vc.to_string())

    return data_clean


# ENCODING DATA
# Pada tahap ini mengubah input data menjadi input numerik.
# Kita memilih Label Encoding dan One-Hot Encoding karena ada
# data yang memiliki lebih dari dua kategori. One-Hot Encoding
# digunakan untuk menghindari bias pada model yang menganggap
# 2 > 1 (asumsi ordinalitas yang tidak tepat).

def encode_features(data_clean):
    data_clean = data_clean.copy()

    # Label encoding untuk variabel biner/ordinal.
    # Gunakan encoder TERPISAH per kolom agar mapping kelas tidak saling timpa.
    le_acct = LabelEncoder()
    le_res  = LabelEncoder()
    data_clean['Account_Type_Encoded']      = le_acct.fit_transform(data_clean['Account Type'])
    data_clean['Resolution_Status_Encoded'] = le_res.fit_transform(data_clean['Resolution Status'])
    print("Account Type classes      :", le_acct.classes_.tolist())
    print("Resolution Status classes :", le_res.classes_.tolist())

    # One-Hot Encoding untuk variabel nominal tanpa urutan
    nominal_cols = ['Loan Status', 'Gender', 'Transaction Type',
                'Loan Type', 'Card Type', 'Feedback Type']
    data_encoded = pd.get_dummies(data_clean, columns=nominal_cols, drop_first=False)
    print(f"Shape setelah encoding: {data_encoded.shape}")
    return data_encoded


# SCALING AND NORMALIZING
# Pada tahap ini kita melakukan scaling dan normalisasi terhadap
# data numerikal. Kita memilih MinMaxScaler agar data berada pada
# range [0,1] untuk menghindari bias — berbeda dengan StandardScaler
# yang tidak menjamin batas atas/bawah tertentu.

# SCALING AND NORMALIZING
# Scaler dipilih per fitur berdasarkan hasil outlier detection:
# - RobustScaler → fitur dengan outlier ekstrem atau skewed (|skew|>1)
#   menggunakan median dan IQR sehingga tidak terpengaruh outlier
# - MinMaxScaler → fitur dengan distribusi normal dan bounded
#   menghasilkan range [0,1] yang seragam

def normalize_features(data_encoded, robust_cols, minmax_cols):
    data_encoded = data_encoded.copy()

    # Filter hanya kolom yang ada di data_encoded
    robust_cols = [c for c in robust_cols if c in data_encoded.columns]
    minmax_cols = [c for c in minmax_cols if c in data_encoded.columns]

    if robust_cols:
        robust_scaler = RobustScaler()
        data_encoded[robust_cols] = robust_scaler.fit_transform(data_encoded[robust_cols])

    if minmax_cols:
        minmax_scaler = MinMaxScaler()
        data_encoded[minmax_cols] = minmax_scaler.fit_transform(data_encoded[minmax_cols])

    print("=== Normalisasi Selesai ===")
    print(f"  RobustScaler  ({len(robust_cols)}) : {robust_cols}")
    print(f"  MinMaxScaler  ({len(minmax_cols)}) : {minmax_cols}")

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
        'Transaction_to_Balance_Ratio','Loan_to_Balance_Ratio'
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
    plt.close()

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
    plt.close()
    print(mi_df.to_string(index=False))


# SELECTING FEATURES & SAVE DATASETS
# Penting: dataset clustering diekspor dalam NILAI ASLI (belum di-scale).
# Winsorization + scaling dilakukan di Phase 2 agar:
#   1. Profiling cluster bisa memakai nilai asli (interpretasi bisnis valid)
#   2. Pemisahan tanggung jawab jelas (Phase 1 = fitur, Phase 2 = jarak)
#
# Fitur INPUT clustering hanya 3 rasio perilaku (lihat Phase 2: CLUSTER_FEATURES).
# Sisanya = kolom KONTEKS untuk profiling, dan Anomaly = referensi Phase 4.
#
# Mengapa bukan fitur kontinu mentah? Karena terbukti di EDA bahwa fitur
# mentah independen & near-uniform → tidak ada cluster alami. Rasio perilaku
# justru punya struktur (skew tinggi) sehingga segmentasi jadi bermakna.

def save_datasets(data_encoded, data_clean):
    # Dataset untuk Clustering (Phase 2) — diambil dari data_clean (NILAI ASLI)
    clustering_cols = [c for c in [
        # --- 3 FITUR INPUT CLUSTERING (rasio perilaku) ---
        'CC_Utilization',               # tekanan kartu kredit (saldo/limit)
        'Transaction_to_Balance_Ratio', # intensitas likuiditas (transaksi/saldo)
        'Loan_to_Balance_Ratio',        # leverage utang (pinjaman/saldo)
        # --- KONTEKS untuk profiling (tidak masuk jarak) ---
        'Age', 'Account_Age_Years', 'Account Balance', 'Credit Limit',
        'Credit Card Balance', 'Loan Amount', 'Transaction Amount',
        'Transaction Type',   # untuk cek konsistensi balance (Phase 4)
        'Interest Rate', 'Rewards Points', 'Days_Since_Last_Transaction',
        'Account Balance After Transaction',
        # --- referensi Phase 4 (bukan input model) ---
        'Anomaly'
    ] if c in data_clean.columns]

    data_clean[clustering_cols].to_csv(
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
    print(f"Clustering dataset: {data_clean[clustering_cols].shape}  "
          f"(3 fitur rasio = input, sisanya konteks/referensi)")
    print(f"ARM dataset: {data_clean[arm_cols].shape}")


# MAIN PIPELINE

def run_preprocessing(raw_path=RAW_PATH):
    print("=" * 55)
    print("  PHASE 1 - Data Understanding & Preprocessing")
    print("=" * 55)

    data_set                    = load_data(raw_path)
    run_eda(data_set)
    data_set                    = validate_data(data_set)
    data_set, date_cols         = engineer_features(data_set)
    data_clean                  = drop_irrelevant_columns(data_set, date_cols)
    data_clean                  = bin_features(data_clean)
    robust_cols, minmax_cols    = detect_outliers_prescaling(data_clean)  # ← baru
    data_encoded                = encode_features(data_clean)
    data_encoded                = normalize_features(data_encoded, robust_cols, minmax_cols)  # ← updated
    feature_selection(data_encoded)
    save_datasets(data_encoded, data_clean)

    print("\n" + "=" * 55)
    print("  PHASE 1 SELESAI")
    print("=" * 55)
    return data_encoded, data_clean


if __name__ == '__main__':
    run_preprocessing(RAW_PATH)
