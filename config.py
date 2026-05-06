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

CLUSTERING_COLS = [
    'Age', 'Account_Age_Years', 'Account Balance',
    'Loan Amount', 'Interest Rate', 'Loan Term',
    'Transaction Amount', 'Transaction_to_Balance_Ratio',
    'CC_Utilization', 'Rewards Points',
    'Days_Since_Last_Transaction'
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
BEST_K       = 3
N_INIT       = 10
RANDOM_STATE = 42

#  Parameter DBSCAN 
DBSCAN_EPS         = 0.5
DBSCAN_MIN_SAMPLES = 5
