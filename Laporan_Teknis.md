# Laporan Teknis — Implementasi Pipeline KDD

**Kelompok 6 · Banking Transaction Dataset · Data Mining**

Dokumen ini adalah pendamping teknis dari [`Laporan_Knowledge_Discovery.md`](Laporan_Knowledge_Discovery.md).
Laporan Knowledge Discovery menjelaskan **apa** yang ditemukan dalam bahasa
bisnis; laporan ini menjelaskan **bagaimana** temuan itu dihasilkan: arsitektur
kode, fungsi-fungsi kunci, parameter beserta alasannya, dan matematika di balik
metode yang dipakai — termasuk varian *modified* pada Phase 4 (revisi dosen).

---

## 1. Arsitektur Pipeline

### 1.1 Alur data

Pipeline dibangun sebagai lima modul Python yang berkomunikasi lewat **file CSV
antar-fase** — tiap fase bisa dijalankan, diperiksa, dan diulang secara terpisah:

```
Comprehensive_Banking_Database.csv  (5.000 baris, data mentah)
        │
        ▼  src/preprocess.py ─── Phase 1
        ├──► data/dataset_clustering.csv   (3 rasio perilaku + konteks, NILAI ASLI)
        └──► data/dataset_arm.csv          (14 kolom kategorikal hasil binning)
                │                                  │
                ▼  src/clustering.py ── Phase 2    ▼  src/arm.py ── Phase 3
        data/dataset_clustered.csv          outputs/phase3/rules_all.csv (82 rule)
        (+ label KMeans/DBSCAN/Hierarchical)       + top_rules.csv (top 20)
                │
                ▼  src/anomaly.py ─── Phase 4
        data/dataset_final.csv  (label cluster + flag & klasifikasi anomali)
                │
                ▼  src/dashboard.py ── Phase 5 (Plotly Dash, port 8050)
```

### 1.2 Entry point & konfigurasi

- **`main.py`** — satu entry point untuk semua fase:
  `python main.py --phase {1|2|3|4|5|all}`. Import tiap modul dilakukan
  *lazy* (di dalam blok `if`) supaya menjalankan satu fase tidak menarik
  dependency fase lain. Mode `all` menjalankan Phase 1–4 berurutan; dashboard
  sengaja tidak ikut karena server-nya *blocking*.
- **`config.py`** — konstanta terpusat (path, daftar kolom, parameter
  clustering & anomaly). Modul `src/*.py` juga mendefinisikan salinan lokal
  konstanta yang dipakainya supaya tetap bisa dijalankan *standalone*
  (`python src/anomaly.py`) tanpa mengubah `sys.path`.

### 1.3 Konvensi yang berlaku di semua modul

| Konvensi | Implementasi | Alasan |
|---|---|---|
| Plot non-interaktif | `matplotlib.use('Agg')` sebelum `pyplot` di-import | pipeline tidak nge-*freeze* menunggu jendela plot; semua grafik tersimpan ke `outputs/phaseN/` |
| Konsol Windows | `sys.stdout.reconfigure(encoding='utf-8')` | log memakai karakter unicode (→, ✅); default cp1252 akan crash |
| Determinisme | `RANDOM_STATE = 42` untuk K-Means, t-SNE, PCA, Isolation Forest, sampling dendrogram | hasil bisa direproduksi identik antar-run |
| Path portabel | `BASE_DIR = dirname(dirname(abspath(__file__)))` | jalan dari folder mana pun, tidak bergantung *working directory* |
| Dokumentasi | tiap fungsi ber-docstring (tujuan, keputusan teknis, nilai kembalian) + blok komentar berisi justifikasi metodologis | kode = dokumentasi keputusan preprocessing (tuntutan rubrik) |

---

## 2. Phase 1 — `src/preprocess.py` (Data Understanding & Preprocessing)

### 2.1 Peta fungsi

| Fungsi | Tugas | Keputusan teknis penting |
|---|---|---|
| `load_data` | baca CSV mentah | — |
| `run_eda` | histogram, boxplot, countplot, scatter, null/duplikat, korelasi | menghitung berapa pasang fitur dengan \|r\|>0,1 → **temuan struktur data** (lihat 2.2) |
| `validate_data` | validasi logika domain | inkonsistensi saldo di-*cross-check* dengan label `Anomaly` → tidak berkorelasi → baris **tidak dihapus** |
| `engineer_features` | fitur temporal + **3 rasio perilaku** | pembagian nol → `inf` → `NaN` → diisi median |
| `drop_irrelevant_columns` | buang PII, surrogate key, tanggal mentah | privasi + kolom tanpa makna semantik |
| `bin_features` | kontinu → kategorikal | **ambang domain tetap**, bukan kuantil (lihat 2.3) |
| `detect_outliers_prescaling` | outlier IQR per fitur *sebelum* scaling | outlier tidak dibuang; hasilnya memilih scaler per fitur |
| `encode_features` | Label Encoding (biner) + One-Hot (nominal) | encoder terpisah per kolom agar mapping tidak saling timpa |
| `normalize_features` | scaling per fitur | `RobustScaler` bila outlier >5% atau \|skew\|>1, selain itu `MinMaxScaler` |
| `feature_selection` | korelasi + Mutual Information | dua metode sesuai rubrik (korelasi & entropi) |
| `save_datasets` | ekspor 2 dataset turunan | dataset clustering diekspor **nilai asli** (lihat 2.4) |

### 2.2 Temuan struktur data yang menentukan desain

`run_eda` menghitung korelasi seluruh pasangan fitur numerik: hanya **2 dari 55
pasang** dengan |r| > 0,1, keduanya pasangan turunan (`Account Balance` ↔
`Balance After Transaction` 0,70; `Credit Card Balance` ↔ `Minimum Payment Due`
≈ 1,00). Skewness fitur mentah ≈ 0 (near-uniform). Konsekuensi yang di-encode
langsung di kode fase berikutnya: **PCA tidak akan efektif** dan **clustering
fitur mentah tidak akan menemukan cluster alami**.

Karena itu `engineer_features` membuat tiga rasio perilaku — inilah baris kode
paling berpengaruh di seluruh proyek:

```python
data_set['CC_Utilization']               = data_set['Credit Card Balance'] / data_set['Credit Limit']
data_set['Transaction_to_Balance_Ratio'] = data_set['Transaction Amount']  / data_set['Account Balance'].replace(0, np.nan)
data_set['Loan_to_Balance_Ratio']        = data_set['Loan Amount']         / data_set['Account Balance'].replace(0, np.nan)
```

Rasio dua variabel uniform menghasilkan distribusi **berstruktur** (skew 2–7,
ekor kanan panjang) — struktur inilah yang membuat segmentasi Phase 2 bermakna
(silhouette naik dari ≈0,07 ke ≈0,57).

### 2.3 Binning berbasis ambang domain

`bin_features` sengaja **tidak** memakai kuantil/equal-width karena batasnya
bergeser bila data berubah dan kategorinya tidak punya makna intrinsik. Ambang
tetap yang dipakai dirujuk ke literatur/praktik industri (rincian referensi ada
di blok komentar di atas fungsi): tahap hidup finansial untuk umur (Modigliani–
Brumberg; Agarwal dkk. 2009), pedoman utilisasi FICO 30/70/100% (kategori
`Over-Limit` > 100% wajib ada agar sinyal risiko tidak hilang), tier suku bunga
Regulation Z, ambang minimum-balance ritel, bracket pinjaman konsumen, dan tier
nominal transaksi ala pemantauan AML.

### 2.4 Kontrak antar-fase

`save_datasets` mengekspor `dataset_clustering.csv` dalam **nilai asli** (belum
di-scale). Winsorization + scaling menjadi tanggung jawab Phase 2, dengan dua
alasan: (1) profiling cluster harus memakai nilai asli agar interpretasi bisnis
valid; (2) pemisahan tanggung jawab — Phase 1 menghasilkan *fitur*, Phase 2
mengelola *ruang jarak*.

---

## 3. Phase 2 — `src/clustering.py` (Segmentation via Clustering)

### 3.1 Pipeline 9 langkah (`run_clustering`)

```
[1] load → [2] attach_categoricals → [3] dimensionality_analysis (bukti PCA)
→ [4] feature_selection_comparison (exhaustive 286 kombinasi)
→ [5] prepare_features (winsorize 2% + StandardScaler)
→ [6] find_optimal_k (Elbow + Silhouette) → [7] K-Means + profil
→ [8] DBSCAN (auto-eps) + Hierarchical (3 linkage) + profil → [9] bandingkan & simpan
```

### 3.2 Bukti PCA tidak dipakai (`dimensionality_analysis`)

PCA dijalankan **sekali, bukan untuk dipakai**, melainkan untuk membuktikan
scree plot-nya datar: tiap komponen menangkap ≈1/n variance (garis merah "1/n"
digambar sebagai pembanding di `pca_why_not_used.png`). Butuh 9 dari 11
komponen untuk 80% variance → kompresi hanya 18% → reduksi tidak berguna.
Keputusan drop-PCA dengan demikian **terdokumentasi dengan bukti**, bukan asumsi.

### 3.3 Validasi pemilihan fitur (`feature_selection_comparison`)

Untuk menjawab "kenapa 3 rasio itu, bukan fitur lain?", pilihan manual diadu
dengan pencarian otomatis. Inti implementasinya:

```python
combos = list(combinations(pool, 3))            # C(13,3) = 286 kombinasi
scored = [(_kmeans_silhouette(_winsor_scale(df, list(c))), c) for c in combos]
scored.sort(key=lambda t: t[0], reverse=True)   # peringkat semua kombinasi
```

Tiap kombinasi melewati transformasi yang **sama** dengan pipeline utama
(winsorize + scale), di-cluster K-Means K=3, lalu dihitung silhouette **penuh
pada 5.000 baris** (tanpa sampling) supaya peringkatnya setara dengan angka
final yang dilaporkan. Ini bagian terlama Phase 2 (~5 menit): silhouette
O(n²) × 286 kombinasi. Hasil: pencarian otomatis mendarat **tepat di 3 rasio
domain (peringkat #1 dari 286)** — silhouette 0,571 vs 0,065 (semua fitur
mentah) dan 0,081 (PCA 80%).

### 3.4 Penentuan K dan algoritma

- **`find_optimal_k`**: WCSS (elbow) + silhouette untuk K=2..10. Silhouette
  murni tertinggi ada di K=2, tetapi K=2 hanya memisahkan "over-limit vs
  sisanya". **K=3 dipilih atas dasar domain**: silhouette tetap tinggi dan
  memunculkan segmen ketiga (*liquidity-stressed*) yang actionable — keputusan
  ini dicetak eksplisit di log dan digambar di `elbow_silhouette.png`.
- **`run_kmeans`**: K-Means `n_init=10`, lalu dua visual — scatter pasangan
  rasio dan proyeksi **t-SNE** 2D (langkah tunggal terlama kedua, ~30–60 dtk).
- **`run_dbscan`**: eps **dicari otomatis** — kandidat 0,20–2,00 (step 0,05)
  masing-masing dijalankan, lalu dipilih yang persentase noise-nya paling dekat
  ~5% di dalam rentang wajar 2–15%. Noise (-1) ≈ 274 titik (5,5%) menjadi
  sudut pandang *density* untuk Phase 4.
- **`run_hierarchical`**: dendrogram 3 linkage (ward/complete/average) pada
  **sampel 1.000 baris** (dendrogram 5.000 daun tidak terbaca dan berat);
  label final dipotong dari Ward pada seluruh data.
- **`name_segments`**: penamaan segmen **data-driven** — membandingkan median
  rasio tiap cluster terhadap median global (label cluster K-Means itu acak,
  jadi nama tidak boleh di-hardcode ke nomor cluster).

### 3.5 Profil segmen dua lapis

`profile_clusters` menampilkan median **nilai asli** (bukan z-score) untuk 3
rasio + konteks finansial. `profile_categoricals` menambah lapis demografis:
crosstab persentase per segmen dengan metrik **spread** (selisih proporsi
maks–min antar cluster, dalam poin persen; ≥10 pp = membedakan). Temuan yang
dilaporkan jujur: demografi nyaris seragam (2–7 pp) → segmen dibentuk
**perilaku finansial**, bukan demografi.

---

## 4. Phase 3 — `src/arm.py` (Association Rule Mining)

### 4.1 Transformasi ke format transaksi

Setiap baris diubah menjadi *itemset* dengan format **`NamaKolom=Nilai`**
(mis. `Loan Status=Approved`) lalu di-one-hot dengan `TransactionEncoder`
mlxtend → matriks boolean 5.000 × 44 item. Prefiks nama kolom mencegah
ambiguitas ("High" milik `Rate_Category` ≠ "High" milik `CC_Utilization_Category`).

### 4.2 Apriori + auto-tune support

Parameter kualitas ditetapkan dulu dan **tidak dilonggarkan**: `min_lift = 1,4`
(non-trivial, ≥40% di atas acak) dan `min_confidence = 0,5`. Karena binning
memakai ambang domain (bukan kuantil), sebagian kategori kecil dan itemset
lebih jarang — maka `auto_tune_support` menurunkan `min_support` bertahap
(0,05 → step 0,0025) sampai dapat ≥10 rule:

```python
for sup in np.arange(start_support, 0.004, -0.0025).round(4):
    frequent_itemsets = apriori(df_encoded, min_support=sup, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=min_lift)
    rules = rules[rules['confidence'] >= min_confidence]
    if len(rules) >= target_rules:
        return frequent_itemsets, rules, sup
```

Konvergen di `min_support = 0,01` (rule muncul ≥50 dari 5.000 baris) →
**82 rule** (lift 1,401–1,662; confidence 0,50–0,76). Lift yang mentok di
~1,66 adalah **batas data** (atribut kategorikal nyaris independen), bukan bug
— dijelaskan di laporan bisnis.

### 4.3 Ekspor dua tingkat & lapisan makna bisnis

`export_rules` menyimpan **`rules_all.csv` (semua 82 rule — sumber angka
"total rule" di laporan & KPI dashboard)** dan `top_rules.csv` (top-20 by lift
untuk tabel ringkas). Dua file ini yang menjaga angka konsisten di seluruh
deliverable. `ITEM_MEANING_MAP` + `pretty_item/pretty_itemset/infer_business_theme`
menerjemahkan item ke frasa bisnis Indonesia dan mengelompokkan rule per tema
(risiko / layanan / nasabah sehat) untuk interpretasi otomatis top-10.

---

## 5. Phase 4 — `src/anomaly.py` (Anomaly Detection, metode *modified*)

Fase ini direvisi sesuai masukan dosen: ketiga metode memakai varian
**modified/robust**, dan versi standar tetap dihitung sebagai **pembanding
sistematis** (`compare_standard_vs_modified` → `standard_vs_modified.csv/png`).
Motivasi bersama: ketiga rasio perilaku *right-skewed* berat (skew 2–7),
sedangkan metode standar diam-diam mengasumsikan distribusi simetris.

### 5.1 IQR modified — *skew-adjusted boxplot* via medcouple

**Medcouple** (Brys–Hubert–Struyf, 2004) adalah statistik kemiringan robust:
median dari kernel semua pasangan (xᵢ ≥ median, xⱼ ≤ median):

```
h(xᵢ, xⱼ) = ((xᵢ − med) − (med − xⱼ)) / (xᵢ − xⱼ)      MC = median h  ∈ [−1, 1]
```

Implementasi di `_medcouple()` divektorisasi numpy O(n²) — untuk 5.000 baris
(~6 juta pasangan) selesai < 1 detik:

```python
xp = x[x >= med]; xm = x[x <= med]
a = xp[:, None] - med          # jarak sisi kanan ke median
b = med - xm[None, :]          # jarak median ke sisi kiri
h = (a - b) / (xp[:, None] - xm[None, :])
return float(np.nanmedian(h))  # pasangan 0/0 (tie di median) diabaikan
```

Pagar *adjusted boxplot* (Hubert & Vandervieren, 2008) mengoreksi pagar klasik
dengan faktor eksponensial dari MC:

```
MC ≥ 0 :  [Q1 − 1,5·e^(−4·MC)·IQR ,  Q3 + 1,5·e^(+3·MC)·IQR]
MC < 0 :  [Q1 − 1,5·e^(−3·MC)·IQR ,  Q3 + 1,5·e^(+4·MC)·IQR]
```

Bila MC = 0 rumus kembali persis ke pagar klasik. Pada data kami MC = +0,285 /
+0,390 / +0,448 (right-skew) → pagar atas melebar (mis. `CC_Utilization`:
1,56 → 2,64), sehingga ekor kanan yang wajar berhenti salah-flag:
**1.131 → 880 record** (22,6% → 17,6%).

### 5.2 Z-score modified — median/MAD (Iglewicz & Hoaglin, 1993)

Z-score klasik `z = (x−mean)/std` memakai dua momen yang justru terdistorsi
oleh outlier yang sedang dicari (mean tertarik, std membengkak). Versi modified
menggantinya dengan median dan **MAD** (median absolute deviation):

```
M = 0,6745 · (x − median) / MAD          flag bila |M| > 3,5
```

(0,6745 menyetarakan MAD dengan std pada distribusi normal; fallback
`M = (x−med)/(1,253314·MeanAD)` bila MAD = 0.) Efeknya **berlawanan arah**
dengan IQR: karena std klasik membengkak oleh ekor, z klasik hanya menangkap
264 record; modified z menangkap **1.070** — penyimpangan nyata yang
sebelumnya lolos.

### 5.3 Isolation Forest — threshold dari *gap* skor, bukan contamination

Model di-fit sekali (`n_estimators=200`); skor anomali = `−score_samples`
(makin tinggi makin anomali). Alih-alih memakai asumsi `contamination=0.05`,
threshold ditarik dari struktur data (`detect_isolation_forest`):

```python
s_sorted = np.sort(score.values)[::-1]                    # skor menurun
lo_i, hi_i = int(0.005*n), int(0.15*n)                    # jendela 0,5%–15%
gaps = s_sorted[lo_i-1:hi_i-1] - s_sorted[lo_i:hi_i]      # gap antar skor berurutan
cut_idx = lo_i + int(np.argmax(gaps))                     # posisi gap TERBESAR
thr_gap = (s_sorted[cut_idx-1] + s_sorted[cut_idx]) / 2   # titik tengah gap
```

Jendela 0,5%–15% mencegah *cut* degeneratif (di 1–2 titik terekstrem, atau di
tengah bulk populasi normal). Hasil pada data kami: gap terbesar 0,0125 di
peringkat 25 — **66× lipat** median gap di jendela — jadi pemisah alami yang
sangat tegas. Threshold 0,754 → **25 anomali** (contamination efektif 0,50%,
vs 250 pada asumsi 5%). Ke-25 record ini persis himpunan yang disepakati
**ketiga** metode. Visual pendukung: `isoforest_gap.png` (kurva skor + posisi
cut gap vs cut 5%).

### 5.4 Konsensus, klasifikasi, dan validasi

- Flag resmi downstream = versi modified (kolom `flag_iqr`, `flag_zscore`,
  `flag_if`); versi standar disimpan dengan akhiran `_std` untuk transparansi
  (ikut diekspor di `anomaly_report.csv`).
- `n_methods` (0–3) = jumlah metode yang menandai record; **konsensus = ≥2
  metode → 446 record**; ketiganya sekaligus → 25.
- `classify_anomalies` menerapkan aturan transparan berurutan:
  **Risk Signal** (ekstrem pada dimensi berisiko: CC ≥ 2× limit, transaksi
  ≥ 10× saldo, pinjaman ≥ 100× saldo, DAN dikuatkan ≥2 metode) →
  **Data Error/Quality** (saldo akhir negatif / tidak konsisten) →
  **Rare but Valid**. Hasil: 255 / 602 / 647 (+ 3.496 Normal).
- Cross-reference Phase 2: 203 dari 446 konsensus juga DBSCAN-noise;
  konsentrasi per segmen: Liquidity-Stressed 78,6%, Credit-Stressed 22,8%,
  Mainstream 0,4%.
- `validate_against_label`: label `Anomaly` bawaan **hanya** dipakai di sini
  (bukan target). Kehadiran label -1 di anomali kami 5,6% ≈ base rate 6,0% —
  keselarasan setara acak, konsisten dengan MI ≈ 0 di Phase 1.

### 5.5 Ringkasan standar vs modified

| Metode | Standar | Modified | Arah & alasan |
|---|---:|---:|---|
| IQR | 1.131 (22,6%) | **880** (17,6%) | pagar mengikuti skew → ekor kanan wajar tak ter-flag |
| Z-Score | 264 (5,3%) | **1.070** (21,4%) | MAD tidak membengkak oleh ekor → penyimpangan nyata tertangkap |
| Isolation Forest | 250 (5,0%) | **25** (0,5%) | threshold dari gap alami skor, bukan asumsi 5% |

---

## 6. Phase 5 — `src/dashboard.py` (Visualization & Knowledge Presentation)

### 6.1 Struktur

- **`load_dashboard_data`** — dua sumber: `dataset_final.csv` (5.000 baris,
  gabungan hasil Phase 1–4) dan `rules_all.csv` (semua 82 rule; fallback
  `top_rules.csv`). KPI "Association Rules" dihitung dari file lengkap supaya
  **konsisten dengan angka laporan**.
- **`build_app`** — merakit layout + callback. Sengaja dipisah dari
  `run_dashboard` (yang menyalakan server) supaya aplikasi bisa di-*smoke-test*
  tanpa server: `from src.dashboard import build_app; build_app()`.
- **Figure builder terpisah per grafik** (`fig_cluster_map`,
  `fig_segment_demographics`, `fig_rule_network`, `fig_anomaly_scatter`, dst.)
  — masing-masing fungsi murni `data → go.Figure`, mudah diuji.

### 6.2 KPI dan interaktivitas

Lima KPI dihitung langsung dari data saat startup (bukan hardcode): total
nasabah, jumlah segmen, **82** rule, **446** anomali konsensus, **255** risk
signal. Tiga tab interaktif:

| Tab | Kontrol (Input) | Grafik (Output) |
|---|---|---|
| Segmentasi | metode clustering, sumbu X/Y, fitur demografis | cluster map (log-log), pie proporsi, profil rasio, komposisi demografis, validasi feature-selection (statis) |
| Association Rules | slider minimum lift | network rules (networkx spring-layout, seed tetap), scatter support-confidence, tabel rules (sortable) |
| Anomali | sumbu X/Y scatter | breakdown klasifikasi, anomali per segmen, scatter anomali |

Detail kecil yang penting: nilai maksimum slider lift di-*floor* ke 2 desimal
(bukan *round*) — bila di-round ke atas, posisi slider tertinggi menyaring
semua rule dan grafik menjadi kosong.

---

## 7. Reproducibilitas

### 7.1 Menjalankan ulang dari nol

```bash
pip install -r requirements.txt        # pandas, scikit-learn, mlxtend, dash, dll.
python main.py --phase all             # Phase 1–4 berurutan
python main.py --phase 5               # dashboard di http://127.0.0.1:8050
```

Urutan fase wajib: Phase 2 butuh output Phase 1; Phase 4 butuh output Phase 2;
dashboard butuh output Phase 3 & 4. Semua acakan memakai `RANDOM_STATE=42`
sehingga angka di laporan ini dapat direproduksi identik.

### 7.2 Perkiraan waktu eksekusi

| Fase | Perkiraan | Komponen dominan |
|---|---|---|
| Phase 1 | < 1 menit | plotting EDA |
| Phase 2 | ~6–7 menit | exhaustive search 286 kombinasi dengan silhouette penuh (~5 menit, O(n²)); t-SNE (~30–60 dtk) |
| Phase 3 | < 1 menit | loop auto-tune Apriori |
| Phase 4 | < 1 menit | medcouple O(n²) tervektorisasi + IsolationForest 200 tree |

### 7.3 Parameter kunci

| Parameter | Nilai | Lokasi | Alasan |
|---|---|---|---|
| `WINSOR_LIMIT` | 0,02 | config / clustering | cap ekor atas 2% agar cluster digerakkan perilaku, bukan 1–2 outlier ekstrem |
| `BEST_K` | 3 | config / clustering | silhouette tinggi + segmen ketiga yang actionable (justifikasi di `find_optimal_k`) |
| `DBSCAN_MIN_SAMPLES` | 10 | config / clustering | ukuran minimum core-neighborhood yang stabil pada n=5.000 |
| `MIN_LIFT` / `MIN_CONFIDENCE` | 1,4 / 0,5 | arm | ambang kualitas rule; TIDAK dilonggarkan saat auto-tune |
| `min_support` (hasil tune) | 0,01 | arm (otomatis) | support terbesar yang menghasilkan ≥10 rule |
| `MODZ_THRESH` | 3,5 | config / anomaly | ambang baku modified z-score (Iglewicz & Hoaglin) |
| `GAP_MIN/MAX_FRAC` | 0,005 / 0,15 | config / anomaly | jendela pencarian gap IsoForest — cegah cut degeneratif |
| `Z_THRESH`, `IF_CONTAMINATION` | 3,0 / 0,05 | config / anomaly | dipertahankan HANYA sebagai pembanding versi standar |

### 7.4 Keterbatasan teknis yang disadari

- **Silhouette dan medcouple O(n²)** — masih nyaman di n=5.000; untuk dataset
  jauh lebih besar perlu sampling (kode sudah menyediakan parameter `sample`)
  atau implementasi medcouple O(n log n).
- **Dendrogram di-sampel 1.000 baris** — hanya visualisasinya; label
  hierarchical final tetap dihitung pada seluruh data.
- **Dashboard memuat data saat startup** — bila pipeline dijalankan ulang,
  dashboard perlu di-restart (refresh browser tidak memuat ulang CSV).
- **Jendela gap IsoForest (0,5%–15%)** adalah pilihan desain; di luar rentang
  itu cut dianggap degeneratif. Sensitivitasnya rendah pada data ini karena
  gap terpilih 66× lipat gap tipikal.
