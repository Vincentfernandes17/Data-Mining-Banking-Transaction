# Banking Data Mining Project — Knowledge Discovery (KDD)

Group 6 — **Banking Transaction Dataset**. Proyek ini menjalankan pipeline KDD
5-fase end-to-end untuk menemukan pola, segmen, dan aturan asosiasi tersembunyi
pada data perbankan. Tujuannya **penemuan & interpretasi pengetahuan**, bukan
akurasi prediksi.

---

## Temuan Kunci yang Membentuk Metodologi

Analisis Phase 1 menemukan fakta penting tentang dataset ini:

> **Mayoritas fitur numerik saling INDEPENDEN dan terdistribusi mendekati
> uniform** (skewness ≈ 0; hanya 2 dari 55 pasang fitur yang berkorelasi, dan
> keduanya pasangan turunan).

Dua konsekuensi yang menentukan desain:

1. **PCA tidak dipakai sebagai reduksi.** Tanpa redundansi antar fitur, setiap
   komponen PCA hanya menangkap ≈1/n variance (scree plot **datar**), sehingga
   "reduksi" 11→9 fitur tidak ada gunanya. PCA tetap dijalankan **sekali** hanya
   untuk *membuktikan* hal ini (`outputs/phase2/pca_why_not_used.png`).

2. **Clustering memakai 3 RASIO PERILAKU hasil feature engineering**, bukan fitur
   kontinu mentah. Rasio dari dua fitur uniform menghasilkan distribusi
   **berstruktur** (skew 2–7), sehingga segmen yang terbentuk bermakna:
   - `CC_Utilization` = saldo kartu / limit → tekanan kredit
   - `Transaction_to_Balance_Ratio` = transaksi / saldo → intensitas likuiditas
   - `Loan_to_Balance_Ratio` = pinjaman / saldo → leverage utang

   **Dampak:** Silhouette Score naik dari **≈0.07** (fitur mentah) → **≈0.57**
   (rasio perilaku, K-Means K=3).

---

## Hasil per Fase

| Fase | Teknik | Hasil utama |
|------|--------|-------------|
| **1. Preprocessing** | Cleaning, validasi, feature engineering, binning, encoding, scaling, feature selection (korelasi + Mutual Information/entropi) | Dataset bersih + 3 rasio perilaku; bukti independensi fitur |
| **2. Clustering** | K-Means, DBSCAN, Hierarchical (3 linkage); Elbow + Silhouette | 3 segmen bernama (sil 0.57); DBSCAN mengisolasi outlier perilaku |
| **3. Association Rule Mining** | Apriori (Support, Confidence, Lift) | 82 rule non-trivial (lift 1.40–1.66) + interpretasi bisnis |
| **4. Anomaly Detection** | IQR, Z-Score, Isolation Forest + cross-reference | 252 risk-signal konsensus; cross-ref DBSCAN noise & segmen |

### 3 Segmen Nasabah (K-Means, K=3)

| Segmen | % | Ciri (median) |
|--------|---|---------------|
| **Mainstream / Balanced** | ≈80% | Semua rasio sehat, saldo & limit tinggi |
| **Credit-Stressed / Over-Limit** | ≈13% | Saldo kartu **1.7× limit**, limit rendah → tekanan kredit |
| **Liquidity-Stressed / High-Leverage** | ≈7% | Transaksi **5×** & pinjaman **50×+** saldo; saldo sangat rendah |

---

## Struktur Folder

```
Banking_Transaction/
├── main.py                  ← Entry point (python main.py --phase all)
├── config.py                ← Konstanta & parameter
├── requirements.txt
├── README.md
├── Laporan_Knowledge_Discovery.md   ← laporan naratif (Phase 5)
├── src/
│   ├── preprocess.py        ← Phase 1
│   ├── clustering.py        ← Phase 2
│   ├── arm.py               ← Phase 3
│   ├── anomaly.py           ← Phase 4
│   └── dashboard.py         ← Phase 5 (Plotly Dash)
├── data/
│   ├── Comprehensive_Banking_Database.csv  ← dataset mentah
│   ├── dataset_clustering.csv  ← output P1 → input P2 (3 rasio + konteks)
│   ├── dataset_arm.csv         ← output P1 → input P3 (kategorikal)
│   ├── dataset_clustered.csv   ← output P2 → input P4 (label cluster)
│   └── dataset_final.csv       ← output P4 → input P5 (label cluster + anomali)
└── outputs/
    ├── phase1/  ← EDA, korelasi, MI, outlier
    ├── phase2/  ← scree (bukti PCA), elbow/silhouette, scatter, dendrogram, profil
    ├── phase3/  ← distribusi support, scatter & bar rules, top_rules.csv
    └── phase4/  ← method overlap, anomaly scatter, anomaly_report.csv
```

---

## Setup & Menjalankan

```bash
# 1. (opsional) virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. jalankan pipeline analisis
python main.py --phase 1     # Preprocessing
python main.py --phase 2     # Clustering
python main.py --phase 3     # Association Rule Mining
python main.py --phase 4     # Anomaly Detection
python main.py --phase all   # Fase 1–4 berurutan

# 4. jalankan dashboard interaktif (Phase 5) — butuh fase 1–4 sudah dijalankan
python main.py --phase 5     # lalu buka http://127.0.0.1:8050
```

**Phase 5 (Visualisasi & Presentasi Pengetahuan):**
- Dashboard interaktif Plotly Dash: `src/dashboard.py` (cluster map, rule network,
  outlier plot, distribusi) — `python main.py --phase 5`.
- Knowledge Discovery Report: [`Laporan_Knowledge_Discovery.md`](Laporan_Knowledge_Discovery.md)
  — narasi bisnis lengkap yang menjawab pertanyaan sentral proyek.

Semua plot disimpan otomatis ke `outputs/phaseN/` (backend matplotlib `Agg`,
jadi pipeline tidak nge-freeze menunggu jendela plot). Urutan fase penting:
Phase 2 butuh output Phase 1, Phase 4 butuh output Phase 2.

---

## Catatan Kejujuran Metodologis

Karena dataset bersifat sintetis dengan atribut yang sebagian besar independen:

- **Clustering** menjadi bermakna **hanya setelah** feature engineering rasio
  perilaku — ini didokumentasikan, bukan disembunyikan.
- **Association rules** memiliki lift moderat (1.4–1.66) karena atribut kategorikal
  hampir independen — ini batas data, dan rule terkuat tetap dilaporkan.
- **Anomaly** berbasis perilaku finansial nyata; label `Anomaly` bawaan dataset
  hanya dipakai untuk validasi akhir (bukan target), sesuai aturan proyek.
