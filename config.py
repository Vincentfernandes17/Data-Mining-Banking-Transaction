"""
config.py
Semua konstanta dan konfigurasi proyek ada di sini.
Kalau mau ganti path, parameter, atau kolom — cukup edit file ini.
"""

import os

# ── Path ────────────────────────────────────────────────────
# BASE_DIR = folder tempat config.py ini berada (root project)
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'outputs')
SRC_DIR     = os.path.join(BASE_DIR, 'src')

RAW_DATA_PATH        = os.path.join(DATA_DIR, 'Comprehensive_Banking_Database.csv')
CLUSTERING_DATA_PATH = os.path.join(DATA_DIR, 'dataset_clustering.csv')
ARM_DATA_PATH        = os.path.join(DATA_DIR, 'dataset_arm.csv')

# ── Kolom ───────────────────────────────────────────────────
PII_COLS = [
    'First Name', 'Last Name', 'Address', 'Email', 'Contact Number'
]

ID_COLS = [
    'Customer ID', 'TransactionID', 'Loan ID', 'CardID', 'Feedback ID'
]

DATE_COLS = [
    'Date Of Account Opening', 'Last Transaction Date',
    'Transaction Date', 'Approval/Rejection Date',
    'Payment Due Date', 'Last Credit Card Payment Date',
    'Feedback Date', 'Resolution Date'
]

NUMERIC_COLS = [
    'Age', 'Account Balance', 'Transaction Amount',
    'Account Balance After Transaction', 'Loan Amount',
    'Interest Rate', 'Loan Term', 'Credit Limit',
    'Credit Card Balance', 'Minimum Payment Due',
    'Rewards Points', 'Account_Age_Years',
    'Days_Since_Last_Transaction', 'CC_Utilization',
    'Transaction_to_Balance_Ratio'
]

NOMINAL_COLS = [
    'Gender', 'Account Type', 'Transaction Type',
    'Loan Type', 'Card Type', 'Feedback Type', 'Resolution Status'
]

# FITUR INPUT clustering = HANYA 3 rasio perilaku (lihat src/clustering.py).
# Fitur kontinu mentah TIDAK dipakai karena terbukti independen & near-uniform
# (tidak ada cluster alami). Rasio perilaku punya struktur (skew tinggi).
CLUSTER_FEATURES = [
    'CC_Utilization',                # tekanan kartu kredit (saldo/limit)
    'Transaction_to_Balance_Ratio',  # intensitas likuiditas (transaksi/saldo)
    'Loan_to_Balance_Ratio',         # leverage utang (pinjaman/saldo)
]

ARM_COLS = [
    'Age_Group', 'Gender', 'Account Type', 'Balance_Bucket',
    'Transaction_Size', 'Transaction Type', 'Loan_Size',
    'Loan Type', 'Loan Status', 'Rate_Category',
    'Card Type', 'CC_Utilization_Category',
    'Feedback Type', 'Resolution Status'
]

# Parameter Clustering
K_RANGE      = range(2, 11)
BEST_K       = 3        # dipilih atas dasar domain (justifikasi di src/clustering.py)
WINSOR_LIMIT = 0.02     # cap 2% ekor atas sebelum scaling
N_INIT       = 10
RANDOM_STATE = 42

#  Parameter DBSCAN (eps dicari otomatis di src/clustering.py)
DBSCAN_MIN_SAMPLES = 10

# Parameter Anomaly Detection (Phase 4)
# Catatan: metode RESMI memakai varian MODIFIED (revisi dosen) —
# IQR skew-adjusted (medcouple), modified z-score (median/MAD), dan
# Isolation Forest dengan threshold GAP antar skor (bukan contamination
# manual). Versi standar tetap dihitung sebagai pembanding sistematis.
Z_THRESH           = 3.0    # z-score KLASIK |z|>3 (pembanding)
MODIFIED_Z_THRESH  = 3.5    # modified z-score |M|>3.5 (Iglewicz & Hoaglin)
IF_CONTAMINATION   = 0.05   # contamination manual 5% (pembanding)
IF_GAP_WINDOW      = (0.005, 0.15)  # jendela pencarian gap: 0.5%-15% skor teratas
