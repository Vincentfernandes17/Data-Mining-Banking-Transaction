"""
arm.py — Phase 3: Association Rule Mining

Pada tahapan ini, kita mencari pola co-occurrence antar atribut
yang tidak dapat terlihat hanya dari tabulasi sederhana.
Algoritma yang digunakan adalah Apriori untuk menemukan frequent
itemsets, lalu di-generate menjadi association rules yang dinilai
berdasarkan Support, Confidence, dan Lift.

Cara running:
    python src/arm.py

Atau import dari notebook:
    from src.arm import run_arm
    rules_df = run_arm('data/dataset_arm.csv')
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
import warnings
warnings.filterwarnings('ignore')

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'phase3')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameter Apriori
MIN_SUPPORT    = 0.05   # minimum 5% dari total transaksi
MIN_CONFIDENCE = 0.5    # minimum 50% confidence
MIN_LIFT       = 1.4    # hanya ambil rules yang non-trivial

# MIN_SUPPORT = 0.05
# Dipilih karena rule harus muncul cukup sering agar stabil dan meaningful.
# Dengan dataset ini, 5% berarti rule muncul pada sekitar 250 baris jika total 5,000 record.

# MIN_CONFIDENCE = 0.5
# Rule harus benar pada minimal separuh kasus antecedent → consequent.

# MIN_LIFT = 1.2
# Lift > 1 berarti hubungan non-trivial; 1.2 berarti setidaknya 20% lebih kuat dari random.

# Jika jumlah rule terlalu sedikit, threshold support dapat diturunkan secara bertahap, 
# tetapi tetap harus dijaga agar rule tidak menjadi terlalu lemah atau tidak meaningful.

# ════════════════════════════════════════════════════════════
# LOAD DATA
# Dataset ARM adalah hasil binning dari Phase 1 — semua kolom
# sudah berbentuk kategorikal sehingga siap diproses Apriori.
# ════════════════════════════════════════════════════════════

def load_arm_data(path):
    df_arm = pd.read_csv(path)

    # Pastikan semua kolom bertipe string (kategorikal)
    for col in df_arm.columns:
        df_arm[col] = df_arm[col].astype(str)

    print(f"Dataset ARM loaded : {df_arm.shape[0]:,} rows x {df_arm.shape[1]} kolom")
    print(f"Kolom              : {df_arm.columns.tolist()}")
    print(f"\nDistribusi nilai per kolom:")
    for col in df_arm.columns:
        print(f"  {col}: {df_arm[col].nunique()} kategori "
              f"→ {df_arm[col].unique().tolist()}")
    return df_arm


# ════════════════════════════════════════════════════════════
# ENCODING — TRANSFORMASI KE FORMAT APRIORI
# Apriori membutuhkan data dalam format one-hot encoded boolean.
# Setiap kolom menjadi "item" dengan format "NamaKolom=Nilai".
# Contoh: Age_Group=Young, Loan Status=Approved, dst.
#
# Kita menggabungkan nama kolom dengan nilainya agar item tidak
# ambigu — misalnya "High" di Balance_Bucket berbeda dengan
# "High" di Rate_Category.
# ════════════════════════════════════════════════════════════

def encode_for_apriori(df_arm):
    """
    Ubah setiap baris menjadi list of items, lalu one-hot encode.
    Format item: "NamaKolom=Nilai"
    """
    # Buat list of items per transaksi (baris)
    transactions = []
    for _, row in df_arm.iterrows():
        items = [f"{col}={val}" for col, val in row.items()
                 if val not in ('nan', 'None', '')]
        transactions.append(items)

    # One-hot encode menggunakan TransactionEncoder
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    print(f"Shape setelah encoding : {df_encoded.shape}")
    print(f"Total unique items     : {len(te.columns_)}")
    print(f"\nContoh items (10 pertama):")
    print(te.columns_[:10])

    return df_encoded, transactions


# ════════════════════════════════════════════════════════════
# APRIORI — FREQUENT ITEMSETS
# Apriori bekerja dengan prinsip:
# "Jika sebuah itemset sering muncul, maka semua subset-nya
# juga harus sering muncul (Apriori Property)."
#
# Support = P(A ∩ B) = frekuensi kemunculan itemset / total transaksi
# Semakin tinggi support, semakin sering itemset tersebut muncul.
#
# min_support dimulai dari 0.05 (5%) — jika rules yang dihasilkan
# kurang dari 10, turunkan secara bertahap.
# ════════════════════════════════════════════════════════════

def find_frequent_itemsets(df_encoded, min_support=MIN_SUPPORT):
    print(f"=== Mencari Frequent Itemsets (min_support={min_support}) ===")

    frequent_itemsets = apriori(
        df_encoded,
        min_support=min_support,
        use_colnames=True,
        max_len=4       # maksimal 4 item per itemset
    )
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)

    print(f"Total frequent itemsets ditemukan : {len(frequent_itemsets)}")
    print(f"\nDistribusi panjang itemset:")
    print(frequent_itemsets['length'].value_counts().sort_index().to_string())
    print(f"\nTop 10 Frequent Itemsets (support tertinggi):")
    print(frequent_itemsets.nlargest(10, 'support')[
        ['support', 'itemsets']
    ].to_string(index=False))

    # Plot distribusi support
    plt.figure(figsize=(10, 4))
    frequent_itemsets['support'].hist(bins=30, edgecolor='black')
    plt.xlabel('Support')
    plt.ylabel('Jumlah Itemset')
    plt.title(f'Distribusi Support Frequent Itemsets (min_support={min_support})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'support_distribution.png'), dpi=150)
    plt.close()

    return frequent_itemsets


# ════════════════════════════════════════════════════════════
# ASSOCIATION RULES
# Dari frequent itemsets, di-generate association rules:
#
# Confidence = P(B|A) = Support(A ∩ B) / Support(A)
# → Seberapa sering B muncul ketika A muncul
#
# Lift = Confidence / Support(B) = P(B|A) / P(B)
# → Lift > 1 : A dan B berkorelasi positif (non-trivial)
# → Lift = 1 : A dan B independen (trivial, tidak berguna)
# → Lift < 1 : A dan B berkorelasi negatif
#
# Filter yang digunakan:
# - Lift > 1.2   : hanya rules yang benar-benar non-trivial
# - Confidence > 0.5 : minimal 50% akurasi rule
# - Buang rules trivial (antecedent = consequent)
# ════════════════════════════════════════════════════════════

def generate_rules(frequent_itemsets,
                   min_confidence=MIN_CONFIDENCE,
                   min_lift=MIN_LIFT):

    print(f"=== Generate Association Rules ===")
    print(f"min_confidence = {min_confidence}")
    print(f"min_lift       = {min_lift}")

    rules = association_rules(
        frequent_itemsets,
        metric='lift',
        min_threshold=min_lift
    )

    # Filter confidence
    rules = rules[rules['confidence'] >= min_confidence]

    # Buang rules sepele: antecedent == consequent
    rules = rules[rules['antecedents'] != rules['consequents']]

    # Tambah kolom helper untuk readability
    rules['antecedents_str'] = rules['antecedents'].apply(
        lambda x: ', '.join(sorted(x))
    )
    rules['consequents_str'] = rules['consequents'].apply(
        lambda x: ', '.join(sorted(x))
    )
    rules['rule'] = rules['antecedents_str'] + '  →  ' + rules['consequents_str']

    # Sort by lift descending
    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)

    print(f"\nTotal rules setelah filter : {len(rules)}")
    if len(rules) == 0:
        print("⚠️  Tidak ada rules yang memenuhi threshold.")
        print("    Coba turunkan min_support atau min_confidence.")
    else:
        print(f"\nTop 15 Rules (Lift tertinggi):")
        print(rules[['antecedents_str','consequents_str',
                      'support','confidence','lift']
                    ].head(15).to_string(index=False))

    return rules


# ════════════════════════════════════════════════════════════
# AUTO-TUNE MIN_SUPPORT
# Jika rules yang dihasilkan < 10, turunkan min_support
# secara otomatis hingga mendapatkan minimal 10 rules.
# ════════════════════════════════════════════════════════════

def auto_tune_support(df_encoded, target_rules=10,
                      start_support=MIN_SUPPORT,
                      min_confidence=MIN_CONFIDENCE,
                      min_lift=MIN_LIFT):
    """
    Iterasi min_support dari start_support turun ke 0.01
    hingga didapat minimal target_rules rules.
    """
    support_values = np.arange(start_support, 0.009, -0.0025).round(3)

    for sup in support_values:
        print(f"\nMencoba min_support = {sup}...")
        frequent_itemsets = apriori(
            df_encoded, min_support=sup,
            use_colnames=True, max_len=4
        )
        if len(frequent_itemsets) == 0:
            continue

        rules = association_rules(
            frequent_itemsets, metric='lift', min_threshold=min_lift
        )
        rules = rules[rules['confidence'] >= min_confidence]

        print(f"  → {len(rules)} rules ditemukan")
        if len(rules) >= target_rules:
            print(f"✅ min_support={sup} menghasilkan {len(rules)} rules")
            return frequent_itemsets, rules, sup

    print("⚠️  Tidak bisa mencapai target rules. Pakai min_support terkecil.")
    frequent_itemsets = apriori(
        df_encoded, min_support=0.01,
        use_colnames=True, max_len=4
    )
    rules = association_rules(
        frequent_itemsets, metric='lift', min_threshold=min_lift
    )
    rules = rules[rules['confidence'] >= min_confidence]
    return frequent_itemsets, rules, 0.01


# ════════════════════════════════════════════════════════════
# VISUALISASI
# ════════════════════════════════════════════════════════════

def visualize_rules(rules, top_n=15, save_plots=True):
    if len(rules) == 0:
        print("Tidak ada rules untuk divisualisasikan.")
        return

    top_rules = rules.head(top_n)

    # ── Plot 1: Support vs Confidence, warna = Lift ─────────
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        top_rules['support'],
        top_rules['confidence'],
        c=top_rules['lift'],
        cmap='YlOrRd', s=100, alpha=0.8, edgecolors='black'
    )
    plt.colorbar(scatter, label='Lift')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title(f'Top {top_n} Rules — Support vs Confidence (warna = Lift)')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'rules_scatter.png'), dpi=150)
    plt.close()

    # ── Plot 2: Bar chart Lift top rules ────────────────────
    plt.figure(figsize=(12, max(6, top_n * 0.5)))
    colors = plt.cm.RdYlGn(
        np.linspace(0.3, 0.9, len(top_rules))
    )[::-1]
    bars = plt.barh(
        range(len(top_rules)),
        top_rules['lift'],
        color=colors, edgecolor='black', alpha=0.85
    )
    plt.yticks(
        range(len(top_rules)),
        [f"{r['antecedents_str']}  →  {r['consequents_str']}"
         for _, r in top_rules.iterrows()],
        fontsize=8
    )
    plt.axvline(x=1.0, color='red', linestyle='--',
                alpha=0.5, label='Lift = 1 (trivial)')
    plt.xlabel('Lift')
    plt.title(f'Top {top_n} Association Rules — Ranked by Lift')
    plt.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'rules_lift_bar.png'),
                    dpi=150, bbox_inches='tight')
    plt.close()

    # ── Plot 3: Heatmap Support x Confidence ────────────────
    plt.figure(figsize=(10, 5))
    bins_sup = pd.cut(rules['support'], bins=5)
    bins_con = pd.cut(rules['confidence'], bins=5)
    heatmap_data = rules.groupby(
        [bins_sup, bins_con], observed=True
    )['lift'].mean().unstack(fill_value=0)

    sns.heatmap(heatmap_data, annot=True, fmt='.2f',
                cmap='YlOrRd', linewidths=0.5)
    plt.title('Rata-rata Lift per Bucket Support × Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Support')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(OUTPUT_DIR, 'rules_heatmap.png'), dpi=150)
    plt.close()


# ════════════════════════════════════════════════════════════
# EXPORT RULES TABLE
# Simpan top rules ke CSV untuk deliverable laporan.
# ════════════════════════════════════════════════════════════

def export_rules(rules, top_n=20):
    top_rules = rules.head(top_n).copy()
    top_rules = top_rules[[
        'antecedents_str', 'consequents_str',
        'support', 'confidence', 'lift'
    ]].rename(columns={
        'antecedents_str' : 'Antecedent (IF)',
        'consequents_str' : 'Consequent (THEN)',
        'support'         : 'Support',
        'confidence'      : 'Confidence',
        'lift'            : 'Lift'
    })
    top_rules[['Support','Confidence','Lift']] = \
        top_rules[['Support','Confidence','Lift']].round(4)
    top_rules.insert(0, 'Rank', range(1, len(top_rules)+1))

    output_path = os.path.join(OUTPUT_DIR, 'top_rules.csv')
    top_rules.to_csv(output_path, index=False)
    print(f"\n✅ Top {top_n} rules tersimpan → {output_path}")
    print(top_rules.to_string(index=False))

    return top_rules

# ════════════════════════════════════════════════════════════
# BUSINESS MEANING LAYER
# Menerjemahkan item Apriori dari format "Kolom=Nilai"
# ke bahasa bisnis agar interpretasi rule lebih kuat.
# ════════════════════════════════════════════════════════════

ITEM_MEANING_MAP = {
    'Age_Group=Young': 'nasabah usia muda',
    'Age_Group=Adult': 'nasabah dewasa',
    'Age_Group=Middle-aged': 'nasabah paruh baya',
    'Age_Group=Senior': 'nasabah senior',

    'Balance_Bucket=Low': 'saldo rendah',
    'Balance_Bucket=Lower-Mid': 'saldo menengah bawah',
    'Balance_Bucket=Upper-Mid': 'saldo menengah atas',
    'Balance_Bucket=High': 'saldo tinggi',

    'Transaction_Size=Small': 'transaksi kecil',
    'Transaction_Size=Medium': 'transaksi menengah',
    'Transaction_Size=Large': 'transaksi besar',
    'Transaction_Size=Very Large': 'transaksi sangat besar',

    'Loan_Size=Small': 'pinjaman kecil',
    'Loan_Size=Medium': 'pinjaman menengah',
    'Loan_Size=Large': 'pinjaman besar',
    'Loan_Size=Very Large': 'pinjaman sangat besar',

    'Rate_Category=Low': 'bunga rendah',
    'Rate_Category=Moderate': 'bunga moderat',
    'Rate_Category=High': 'bunga tinggi',

    # CC_Utilization_Category sekarang: Low / Moderate / High / Over-Limit
    'CC_Utilization_Category=Low': 'utilisasi kartu rendah',
    'CC_Utilization_Category=Moderate': 'utilisasi kartu sedang',
    'CC_Utilization_Category=High': 'utilisasi kartu tinggi',
    'CC_Utilization_Category=Over-Limit': 'pemakaian kartu melebihi limit',

    # Nama kolom asli memakai SPASI, bukan underscore
    'Loan Status=Approved': 'pengajuan disetujui',
    'Loan Status=Rejected': 'pengajuan ditolak',

    'Resolution Status=Resolved': 'keluhan terselesaikan',
    'Resolution Status=Pending': 'keluhan masih tertunda',

    'Account Type=Savings': 'akun tabungan',
    'Account Type=Current': 'akun giro',

    'Feedback Type=Complaint': 'feedback berupa komplain',
    'Feedback Type=Suggestion': 'feedback berupa saran',
    'Feedback Type=Inquiry': 'feedback berupa pertanyaan'
}

def pretty_item(item):
    if item in ITEM_MEANING_MAP:
        return ITEM_MEANING_MAP[item]
    if '=' in item:
        key, val = item.split('=', 1)
        return f"{key.replace('_', ' ')} = {val.replace('_', ' ')}"
    return item.replace('_', ' ')

def pretty_itemset(itemset):
    return ', '.join(pretty_item(i) for i in sorted(itemset))

def infer_business_theme(items):
    items = set(items)

    risk_items = {
        'CC_Utilization_Category=High',
        'CC_Utilization_Category=Over-Limit',
        'Rate_Category=High',
        'Transaction_Size=Very Large',
        'Loan Status=Rejected'
    }

    service_items = {
        'Feedback Type=Complaint',
        'Resolution Status=Pending'
    }

    value_items = {
        'Balance_Bucket=High',
        'Loan Status=Approved',
        'CC_Utilization_Category=Low'
    }

    if items & risk_items:
        return 'Profil risiko / tekanan kredit'
    if items & service_items:
        return 'Pola layanan / keluhan nasabah'
    if items & value_items:
        return 'Profil nasabah bernilai tinggi / sehat'
    return 'Pola co-occurrence umum'

# ════════════════════════════════════════════════════════════
# BUSINESS INTERPRETATION TEMPLATE
# Setiap rule harus diinterpretasikan secara bisnis.
# Fungsi ini mencetak template interpretasi untuk top 10 rules.
# ════════════════════════════════════════════════════════════

def print_business_interpretation(rules, top_n=10):
    print("\n" + "="*60)
    print("  BUSINESS INTERPRETATION — TOP RULES")
    print("="*60)

    for i, (_, row) in enumerate(rules.head(top_n).iterrows(), 1):
        antecedents = row['antecedents']
        consequents = row['consequents']

        antecedent_text = pretty_itemset(antecedents)
        consequent_text = pretty_itemset(consequents)
        theme = infer_business_theme(antecedents.union(consequents))

        print(f"\n{'─'*60}")
        print(f"Rule #{i}")
        print(f"  Tema       : {theme}")
        print(f"  IF   : {antecedent_text}")
        print(f"  THEN : {consequent_text}")
        print(f"  Support    : {row['support']:.4f} ({row['support']*100:.1f}% transaksi)")
        print(f"  Confidence : {row['confidence']:.4f} ({row['confidence']*100:.1f}% kasus IF→THEN)")
        print(f"  Lift       : {row['lift']:.4f}")

        if row['lift'] > 1:
            lift_msg = "lebih sering muncul daripada yang diharapkan secara acak"
        elif row['lift'] == 1:
            lift_msg = "setara dengan kejadian acak"
        else:
            lift_msg = "lebih jarang muncul daripada yang diharapkan secara acak"

        print(f"  Makna lift : hubungan ini {lift_msg}.")
        print(f"  Interpretasi bisnis:")
        print(f"  → Nasabah merupakan {antecedent_text} cenderung memiliki {consequent_text}.")

# ════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════

def run_arm(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, 'dataset_arm.csv')

    print("=" * 55)
    print("  PHASE 3 — Association Rule Mining")
    print("=" * 55)

    # [1] Load data
    print("\n[1/6] Load data...")
    df_arm = load_arm_data(path)

    # [2] Encode untuk Apriori
    print("\n[2/6] Encoding ke format Apriori...")
    df_encoded, transactions = encode_for_apriori(df_arm)

    # [3] Auto-tune min_support dan generate rules
    print("\n[3/6] Mencari frequent itemsets & rules...")
    frequent_itemsets, rules, best_support = auto_tune_support(
        df_encoded,
        target_rules=10,
        start_support=MIN_SUPPORT,
        min_confidence=MIN_CONFIDENCE,
        min_lift=MIN_LIFT
    )

    if len(rules) == 0:
        print("\n⚠️  Tidak ada rules ditemukan. Cek dataset ARM.")
        return None

    # Tambah kolom helper
    rules['antecedents_str'] = rules['antecedents'].apply(
        lambda x: ', '.join(sorted(x))
    )
    rules['consequents_str'] = rules['consequents'].apply(
        lambda x: ', '.join(sorted(x))
    )
    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)

    # [4] Summary statistik rules
    print(f"\n[4/6] Summary rules...")
    print(f"Total rules          : {len(rules)}")
    print(f"min_support dipakai  : {best_support}")
    print(f"Lift range           : {rules['lift'].min():.3f} – {rules['lift'].max():.3f}")
    print(f"Confidence range     : {rules['confidence'].min():.3f} – {rules['confidence'].max():.3f}")
    print(f"Support range        : {rules['support'].min():.4f} – {rules['support'].max():.4f}")

    # [5] Visualisasi
    print("\n[5/6] Visualisasi...")
    visualize_rules(rules)

    # [6] Export & interpretasi
    print("\n[6/6] Export rules & interpretasi bisnis...")
    export_rules(rules, top_n=20)
    print_business_interpretation(rules, top_n=10)
    print(f"  → Pola ini mendukung segmentasi/targeting karena muncul cukup sering dan tidak trivial.")

    print("\n" + "=" * 55)
    print("  PHASE 3 SELESAI")
    print("=" * 55)

    return rules


if __name__ == '__main__':
    run_arm()