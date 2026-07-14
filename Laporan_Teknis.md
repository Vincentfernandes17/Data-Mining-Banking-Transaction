# Laporan Teknis, Implementasi Pipeline KDD

**Kelompok 6 · Banking Transaction Dataset · Data Mining**

Dokumen ini adalah pendamping teknis dari [`Laporan_Knowledge_Discovery.md`](Laporan_Knowledge_Discovery.md).
Laporan Knowledge Discovery menjelaskan **apa** yang ditemukan dalam bahasa
bisnis. Laporan teknis ini menjelaskan **bagaimana** temuan itu dihasilkan. Kami
membahas arsitektur kode, fungsi-fungsi yang paling berdampak di tiap fase,
parameter beserta alasannya, dan matematika di balik metode yang dipakai,
termasuk varian *modified* pada Phase 4 yang merupakan revisi dari masukan dosen.

Fungsi-fungsi pembantu kecil (loader, plotter murni, formatter string) tidak
dibahas satu per satu. Fokus laporan ini adalah fungsi yang menentukan hasil dan
keputusan metodologis.

## 1. Arsitektur Pipeline

### 1.1 Alur data

Pipeline dibangun sebagai lima modul Python yang berkomunikasi lewat file CSV
antar-fase. Dengan cara ini tiap fase bisa dijalankan, diperiksa, dan diulang
secara terpisah tanpa harus mengulang fase sebelumnya.

```
Comprehensive_Banking_Database.csv   (5.000 baris, data mentah)
        │
        ▼  src/preprocess.py   (Phase 1)
        ├─► data/dataset_clustering.csv   (3 rasio perilaku + konteks, nilai asli)
        └─► data/dataset_arm.csv          (14 kolom kategorikal hasil binning)
                │                                  │
                ▼  src/clustering.py (Phase 2)     ▼  src/arm.py (Phase 3)
        data/dataset_clustered.csv          outputs/phase3/rules_all.csv (82 rule)
        (+ label KMeans/DBSCAN/Hierarchical)       + top_rules.csv (top 20)
                │
                ▼  src/anomaly.py   (Phase 4)
        data/dataset_final.csv   (label cluster + flag & klasifikasi anomali)
                │
                ▼  src/dashboard.py (Phase 5, Plotly Dash, port 8050)
```

### 1.2 Entry point dan konfigurasi

`main.py` adalah satu-satunya entry point untuk semua fase lewat perintah
`python main.py --phase {1|2|3|4|5|all}`. Import tiap modul dilakukan secara
*lazy* di dalam blok `if`, sehingga menjalankan satu fase tidak ikut menarik
dependency fase lain. Mode `all` menjalankan Phase 1 sampai 4 secara berurutan.
Dashboard sengaja tidak ikut dijalankan pada mode `all` karena server-nya
bersifat *blocking*.

`config.py` menampung konstanta terpusat berupa path, daftar kolom, serta
parameter clustering dan anomaly. Setiap modul di `src/` juga mendefinisikan
salinan lokal konstanta yang dipakainya, supaya modul itu tetap bisa dijalankan
secara *standalone* misalnya `python src/anomaly.py` tanpa perlu mengubah
`sys.path`.

### 1.3 Konvensi yang berlaku di semua modul

| Konvensi | Implementasi | Alasan |
|---|---|---|
| Plot non-interaktif | `matplotlib.use('Agg')` sebelum `pyplot` di-import | pipeline tidak nge-*freeze* menunggu jendela plot, semua grafik tersimpan ke `outputs/phaseN/` |
| Konsol Windows | `sys.stdout.reconfigure(encoding='utf-8')` | log memakai karakter unicode seperti panah dan centang, sedangkan default cp1252 akan crash |
| Determinisme | `RANDOM_STATE = 42` untuk K-Means, t-SNE, PCA, Isolation Forest, dan sampling dendrogram | hasil bisa direproduksi identik antar-run |
| Path portabel | `BASE_DIR` diturunkan dari lokasi file lewat `abspath(__file__)` | pipeline jalan dari folder mana pun, tidak bergantung *working directory* |
| Dokumentasi | tiap fungsi ber-docstring berisi tujuan, keputusan teknis, dan nilai kembalian | kode sekaligus menjadi dokumentasi keputusan preprocessing sesuai tuntutan rubrik |

## 2. Phase 1, Preprocessing (`src/preprocess.py`)

### 2.1 Fungsi yang berdampak

| Fungsi | Tugas | Keputusan teknis penting |
|---|---|---|
| `run_eda` | histogram, boxplot, countplot, scatter, cek null dan duplikat, korelasi | menghitung berapa pasang fitur dengan \|r\|>0,1, yang menjadi temuan struktur data di bagian 2.2 |
| `validate_data` | validasi logika domain | inkonsistensi saldo di-*cross-check* dengan label `Anomaly`, ternyata tidak berkorelasi, sehingga baris tidak dihapus |
| `engineer_features` | fitur temporal dan **3 rasio perilaku** | pembagian nol diarahkan ke `NaN` lalu diisi median |
| `drop_irrelevant_columns` | buang PII, surrogate key, dan tanggal mentah | menjaga privasi sekaligus membuang kolom tanpa makna semantik sebelum encoding |
| `bin_features` | kontinu menjadi kategorikal | memakai ambang domain tetap, bukan kuantil, lihat 2.3 |
| `detect_outliers_prescaling` | deteksi outlier IQR per fitur sebelum scaling | outlier tidak dibuang, hasilnya dipakai untuk memilih scaler tiap fitur |
| `encode_features` | Label Encoding untuk kolom biner dan One-Hot untuk nominal | encoder terpisah per kolom supaya mapping antar kolom tidak saling menimpa |
| `normalize_features` | scaling per fitur | `RobustScaler` bila outlier di atas 5% atau \|skew\|>1, selain itu `MinMaxScaler` |
| `feature_selection` | korelasi dan Mutual Information | dua metode seleksi sesuai rubrik yaitu korelasi dan entropi |
| `save_datasets` | ekspor 2 dataset turunan | dataset clustering diekspor dalam nilai asli, lihat 2.4 |

### 2.2 Temuan struktur data yang menentukan desain

`run_eda` menghitung korelasi seluruh pasangan fitur numerik. Hanya 2 dari 55
pasang yang punya |r| di atas 0,1, dan keduanya memang pasangan turunan yaitu
`Account Balance` dengan `Balance After Transaction` sebesar 0,70, serta
`Credit Card Balance` dengan `Minimum Payment Due` yang mendekati 1,00. Skewness
fitur mentah mendekati nol atau near-uniform. Dua konsekuensi langsung di-encode
di kode fase berikutnya. Pertama, PCA tidak akan efektif. Kedua, clustering pada
fitur mentah tidak akan menemukan cluster alami.

Karena itu `engineer_features` membuat tiga rasio perilaku. Ini adalah baris kode
paling berpengaruh di seluruh proyek.

```python
data_set['CC_Utilization']               = data_set['Credit Card Balance'] / data_set['Credit Limit']
data_set['Transaction_to_Balance_Ratio'] = data_set['Transaction Amount']  / data_set['Account Balance'].replace(0, np.nan)
data_set['Loan_to_Balance_Ratio']        = data_set['Loan Amount']         / data_set['Account Balance'].replace(0, np.nan)
```

Rasio dari dua variabel uniform menghasilkan distribusi yang berstruktur dengan
skew 2 sampai 7 dan ekor kanan panjang. Struktur inilah yang membuat segmentasi
Phase 2 menjadi bermakna, dengan silhouette naik dari sekitar 0,07 menjadi
sekitar 0,57.

### 2.3 Binning berbasis ambang domain

`bin_features` sengaja tidak memakai kuantil atau equal-width. Alasannya batas
kuantil bergeser setiap kali data berubah dan kategorinya tidak punya makna
intrinsik. Ambang tetap yang dipakai dirujuk ke literatur dan praktik industri,
dengan rincian referensi ada di blok komentar di atas fungsi. Contohnya tahap
hidup finansial untuk kelompok umur mengikuti Modigliani dan Brumberg serta
Agarwal dkk. 2009, pedoman utilisasi kartu kredit 30/70/100% mengikuti FICO
dengan kategori `Over-Limit` di atas 100% yang wajib ada agar sinyal risiko tidak
hilang, tier suku bunga mengikuti Regulation Z, lalu ambang minimum-balance
ritel, bracket pinjaman konsumen, dan tier nominal transaksi ala pemantauan AML.

### 2.4 Kontrak antar-fase

`save_datasets` mengekspor `dataset_clustering.csv` dalam nilai asli yang belum
di-scale. Winsorization dan scaling menjadi tanggung jawab Phase 2. Ada dua
alasan. Alasan pertama, profiling cluster harus memakai nilai asli supaya
interpretasi bisnis tetap valid. Alasan kedua adalah pemisahan tanggung jawab,
di mana Phase 1 bertugas menghasilkan fitur dan Phase 2 bertugas mengelola ruang
jarak.

## 3. Phase 2, Segmentation via Clustering (`src/clustering.py`)

### 3.1 Pipeline sembilan langkah (`run_clustering`)

```
[1] load
[2] attach_categoricals
[3] dimensionality_analysis   (bukti PCA tidak dipakai)
[4] feature_selection_comparison   (exhaustive 286 kombinasi)
[5] prepare_features   (winsorize 2% + StandardScaler)
[6] find_optimal_k   (Elbow + Silhouette)
[7] run_kmeans + profil
[8] run_dbscan (auto-eps) + run_hierarchical (3 linkage) + profil
[9] compare_methods + save_clustered
```

### 3.2 Bukti PCA tidak dipakai (`dimensionality_analysis`)

PCA dijalankan satu kali bukan untuk dipakai, melainkan untuk membuktikan bahwa
scree plot-nya datar. Tiap komponen hanya menangkap sekitar 1/n variance, dan
garis merah pembanding 1/n digambar di `pca_why_not_used.png`. Dibutuhkan 9 dari
11 komponen untuk mencapai 80% variance, jadi kompresi hanya 18% dan reduksi
tidak berguna. Dengan begitu keputusan tidak memakai PCA punya bukti, bukan
sekadar asumsi.

### 3.3 Validasi pemilihan fitur (`feature_selection_comparison`)

Untuk menjawab pertanyaan kenapa 3 rasio itu yang dipilih dan bukan fitur lain,
pilihan manual diadu dengan pencarian otomatis menyeluruh. Inti implementasinya
seperti berikut.

```python
combos = list(combinations(pool, 3))            # C(13,3) = 286 kombinasi
scored = [(_kmeans_silhouette(_winsor_scale(df, list(c))), c) for c in combos]
scored.sort(key=lambda t: t[0], reverse=True)   # peringkat semua kombinasi
```

Tiap kombinasi melewati transformasi yang sama dengan pipeline utama yaitu
winsorize lalu scale, di-cluster K-Means dengan K=3, kemudian dihitung silhouette
penuh pada 5.000 baris tanpa sampling supaya peringkatnya setara dengan angka
final yang dilaporkan. Bagian ini adalah bagian terlama Phase 2 sekitar 5 menit,
karena silhouette berbiaya O(n²) dikali 286 kombinasi. Hasilnya, pencarian
otomatis mendarat tepat di 3 rasio domain sebagai peringkat 1 dari 286.
Silhouette-nya 0,571, jauh di atas 0,065 pada seluruh fitur mentah dan 0,081 pada
PCA 80%.

### 3.4 Penentuan K dan algoritma

`find_optimal_k` menghitung WCSS untuk elbow dan silhouette untuk K dari 2 sampai
10. Silhouette murni tertinggi ada di K=2, tetapi K=2 hanya memisahkan kelompok
over-limit dari sisanya. Kami memilih K=3 atas dasar domain, karena silhouette
tetap tinggi dan memunculkan segmen ketiga yaitu *liquidity-stressed* yang
actionable. Keputusan ini dicetak eksplisit di log dan digambar di
`elbow_silhouette.png`.

`run_kmeans` menjalankan K-Means dengan `n_init=10`, lalu menghasilkan dua visual
yaitu scatter pasangan rasio dan proyeksi t-SNE 2D. Proyeksi t-SNE adalah langkah
tunggal terlama kedua sekitar 30 sampai 60 detik.

`run_dbscan` mencari eps secara otomatis. Kandidat eps dari 0,20 sampai 2,00
dengan langkah 0,05 masing-masing dijalankan, lalu dipilih yang persentase
noise-nya paling dekat ke sekitar 5% di dalam rentang wajar 2 sampai 15%. Titik
noise berlabel -1 ada sekitar 274 titik atau 5,5%, dan ini menjadi sudut pandang
*density* untuk Phase 4.

`run_hierarchical` menggambar dendrogram untuk tiga linkage yaitu ward, complete,
dan average pada sampel 1.000 baris, karena dendrogram dengan 5.000 daun tidak
terbaca dan berat. Label final tetap dipotong dari Ward pada seluruh data.

`name_segments` memberi nama segmen secara data-driven dengan membandingkan
median rasio tiap cluster terhadap median global. Label cluster K-Means bersifat
acak, jadi nama segmen tidak boleh di-hardcode ke nomor cluster.

### 3.5 Profil segmen dua lapis

`profile_clusters` menampilkan median nilai asli dan bukan z-score untuk 3 rasio
serta konteks finansial. `profile_categoricals` menambah lapis demografis berupa
crosstab persentase per segmen, dengan metrik *spread* yaitu selisih proporsi
maksimum dan minimum antar cluster dalam poin persen, di mana selisih 10 poin
persen atau lebih dianggap membedakan. Temuan yang dilaporkan bersifat jujur.
Demografi nyaris seragam di kisaran 2 sampai 7 poin persen, jadi segmen dibentuk
oleh perilaku finansial dan bukan oleh demografi.

## 4. Phase 3, Association Rule Mining (`src/arm.py`)

### 4.1 Transformasi ke format transaksi (`encode_for_apriori`)

Setiap baris diubah menjadi *itemset* dengan format `NamaKolom=Nilai`, misalnya
`Loan Status=Approved`, lalu di-one-hot dengan `TransactionEncoder` dari mlxtend
menjadi matriks boolean berukuran 5.000 kali 44 item. Prefiks nama kolom mencegah
ambiguitas, karena nilai "High" milik `Rate_Category` berbeda arti dari "High"
milik `CC_Utilization_Category`.

### 4.2 Apriori dan auto-tune support (`auto_tune_support`)

Parameter kualitas ditetapkan lebih dulu dan tidak dilonggarkan, yaitu
`min_lift` sebesar 1,4 yang berarti minimal 40% di atas acak, dan
`min_confidence` sebesar 0,5. Karena binning memakai ambang domain dan bukan
kuantil, sebagian kategori berukuran kecil sehingga itemset menjadi lebih jarang.
Untuk mengatasinya `auto_tune_support` menurunkan `min_support` secara bertahap
mulai dari 0,05 dengan langkah 0,0025 sampai memperoleh minimal 10 rule.

```python
for sup in np.arange(start_support, 0.004, -0.0025).round(4):
    frequent_itemsets = apriori(df_encoded, min_support=sup, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=min_lift)
    rules = rules[rules['confidence'] >= min_confidence]
    if len(rules) >= target_rules:
        return frequent_itemsets, rules, sup
```

Di dalam loop ini `find_frequent_itemsets` memanggil Apriori untuk mencari
itemset yang sering muncul pada support tertentu, lalu `generate_rules`
menurunkan aturan dari itemset itu sambil menghitung support, confidence, dan
lift, dan hanya menyimpan aturan yang lolos ambang lift serta confidence.

Proses konvergen di `min_support` sebesar 0,01, yang berarti rule muncul minimal
50 kali dari 5.000 baris, dan menghasilkan 82 rule dengan lift 1,401 sampai 1,662
serta confidence 0,50 sampai 0,76. Lift yang mentok di sekitar 1,66 adalah batas
data karena atribut kategorikal nyaris independen, jadi ini bukan bug dan sudah
dijelaskan di laporan bisnis.

### 4.3 Ekspor dua tingkat dan lapisan makna bisnis (`export_rules`)

`export_rules` menyimpan `rules_all.csv` yang memuat semua 82 rule dan menjadi
sumber angka total rule di laporan maupun KPI dashboard, ditambah `top_rules.csv`
yang memuat top-20 berdasarkan lift untuk tabel ringkas. Dua file inilah yang
menjaga angka tetap konsisten di seluruh deliverable. Selain itu ada
`ITEM_MEANING_MAP` bersama fungsi pembantu `pretty_itemset` dan
`infer_business_theme` yang menerjemahkan item ke frasa bisnis Indonesia dan
mengelompokkan rule per tema yaitu risiko, layanan, atau nasabah sehat, untuk
interpretasi otomatis top-10. Fungsi `print_business_interpretation` memakai
pengelompokan tema ini untuk mencetak ringkasan bisnis, sedangkan
`visualize_rules` menghasilkan grafik pendukung berupa distribusi support,
scatter support dengan confidence, dan bar top rule di `outputs/phase3/`.

## 5. Phase 4, Anomaly Detection metode *modified* (`src/anomaly.py`)

Fase ini direvisi sesuai masukan dosen. Ketiga metode memakai varian modified
atau robust, sedangkan versi standar tetap dihitung sebagai pembanding sistematis
lewat `compare_standard_vs_modified` yang menghasilkan `standard_vs_modified.csv`
dan `.png`. Motivasi bersamanya sama. Ketiga rasio perilaku bersifat right-skewed
berat dengan skew 2 sampai 7, sedangkan metode standar diam-diam mengasumsikan
distribusi simetris.

### 5.1 IQR modified, adjusted boxplot lewat medcouple (`_medcouple`, `adjusted_iqr_bounds`, `detect_iqr`)

Medcouple mengikuti Brys, Hubert, dan Struyf 2004, dan merupakan statistik
kemiringan yang robust. Nilainya adalah median dari kernel semua pasangan xᵢ yang
berada di atas atau sama dengan median dan xⱼ yang berada di bawah atau sama
dengan median.

```
h(xᵢ, xⱼ) = ((xᵢ − med) − (med − xⱼ)) / (xᵢ − xⱼ)
MC = median dari h, dengan nilai di rentang −1 sampai 1
```

Implementasi di `_medcouple` divektorisasi dengan numpy dan berbiaya O(n²). Untuk
5.000 baris yang berarti sekitar 6 juta pasangan, prosesnya selesai di bawah 1
detik.

```python
xp = x[x >= med]           # sisi kanan (termasuk median)
xm = x[x <= med]           # sisi kiri  (termasuk median)
a = xp[:, None] - med      # jarak xi ke median
b = med - xm[None, :]      # jarak median ke xj
h = (a - b) / (xp[:, None] - xm[None, :])
return float(np.nanmedian(h))   # pasangan tie di median diabaikan
```

Pagar *adjusted boxplot* mengikuti Hubert dan Vandervieren 2008, yang mengoreksi
pagar klasik dengan faktor eksponensial dari MC.

```
MC ≥ 0   [Q1 − 1,5·e^(−4·MC)·IQR ,  Q3 + 1,5·e^(+3·MC)·IQR]
MC < 0   [Q1 − 1,5·e^(−3·MC)·IQR ,  Q3 + 1,5·e^(+4·MC)·IQR]
```

Bila MC sama dengan nol, rumus kembali persis ke pagar klasik. Pada data kami MC
bernilai +0,285, +0,390, dan +0,448 yang menandakan right-skew, sehingga pagar
atas melebar. Contohnya untuk `CC_Utilization` pagar atas bergerak dari 1,56
menjadi 2,64, sehingga ekor kanan yang wajar berhenti salah di-flag. Jumlah
tandanya turun dari 1.131 menjadi 880 record, atau dari 22,6% menjadi 17,6%.

### 5.2 Z-score modified, median dan MAD (`detect_zscore`)

Z-score klasik `z = (x−mean)/std` memakai dua momen yang justru terdistorsi oleh
outlier yang sedang dicari, karena mean tertarik ke arah outlier dan std
membengkak. Versi modified mengikuti Iglewicz dan Hoaglin 1993, yang mengganti
keduanya dengan median dan MAD atau median absolute deviation.

```
M = 0,6745 · (x − median) / MAD          flag bila |M| > 3,5
```

Konstanta 0,6745 menyetarakan MAD dengan std pada distribusi normal. Bila MAD
bernilai nol, ada fallback `M = (x−med)/(1,253314·MeanAD)`. Efeknya berlawanan
arah dengan IQR. Karena std klasik membengkak oleh ekor, z klasik hanya menangkap
264 record, sedangkan modified z menangkap 1.070 record yang merupakan
penyimpangan nyata yang sebelumnya lolos.

### 5.3 Isolation Forest, threshold dari gap skor (`detect_isolation_forest`)

Model di-fit satu kali dengan `n_estimators=200`, dan skor anomali dihitung
sebagai `−score_samples` sehingga makin tinggi skornya makin anomali. Alih-alih
memakai asumsi `contamination=0.05`, threshold ditarik dari struktur data.

```python
s_sorted = np.sort(score.values)[::-1]                    # skor menurun
lo_i = max(1, int(GAP_MIN_FRAC * n))                      # batas bawah jendela
hi_i = max(lo_i + 1, int(GAP_MAX_FRAC * n))               # batas atas jendela
gaps = s_sorted[lo_i-1:hi_i-1] - s_sorted[lo_i:hi_i]      # gap antar skor berurutan
cut_idx = lo_i + int(np.argmax(gaps))                     # posisi gap terbesar
thr_gap = (s_sorted[cut_idx-1] + s_sorted[cut_idx]) / 2   # titik tengah gap
```

Jendela pencarian 0,5% sampai 15% mencegah *cut* yang degeneratif, baik pada 1
atau 2 titik terekstrem maupun di tengah bulk populasi normal. Pada data kami gap
terbesar bernilai 0,0125 di peringkat 25, atau sekitar 66 kali median gap di
jendela, jadi ini pemisah alami yang sangat tegas. Threshold 0,754 menghasilkan
25 anomali, yang berarti contamination efektif 0,50% dibandingkan 250 record pada
asumsi 5%. Ke-25 record ini persis himpunan yang disepakati ketiga metode. Visual
pendukungnya ada di `isoforest_gap.png` berupa kurva skor dan posisi cut gap
dibandingkan cut 5%.

### 5.4 Konsensus, klasifikasi, validasi, dan ekspor (`compare_methods`, `classify_anomalies`, `cross_reference`, `validate_against_label`, `export_report`)

Flag resmi untuk downstream memakai versi modified pada kolom `flag_iqr`,
`flag_zscore`, dan `flag_if`. Versi standar tetap disimpan dengan akhiran `_std`
demi transparansi dan ikut diekspor oleh `export_report` ke `anomaly_report.csv`.

`compare_methods` menghitung `n_methods` dari 0 sampai 3 sebagai jumlah metode
yang menandai sebuah record. Konsensus didefinisikan sebagai minimal 2 metode,
yang menghasilkan 446 record, sedangkan record yang ditandai ketiga metode
sekaligus ada 25.

`classify_anomalies` menerapkan aturan transparan secara berurutan. Kelas pertama
adalah **Risk Signal**, yaitu record yang ekstrem pada dimensi berisiko seperti
CC di atas 2 kali limit, transaksi di atas 10 kali saldo, atau pinjaman di atas
100 kali saldo, dan sekaligus dikuatkan minimal 2 metode. Kelas kedua adalah
**Data Error atau Quality**, yaitu saldo akhir yang negatif atau tidak konsisten.
Kelas ketiga adalah **Rare but Valid**. Hasilnya berturut-turut 255, 602, dan
647 record, ditambah 3.496 record Normal.

Fungsi `cross_reference` menghubungkan hasil ke Phase 2 dan menunjukkan 203 dari
446 konsensus juga merupakan DBSCAN-noise. Konsentrasinya per segmen adalah 78,6%
pada Liquidity-Stressed, 22,8% pada Credit-Stressed, dan 0,4% pada Mainstream.

`validate_against_label` memakai label `Anomaly` bawaan dataset hanya di tahap
ini dan bukan sebagai target. Kehadiran label -1 pada anomali kami sebesar 5,6%
hampir sama dengan base rate 6,0%, jadi keselarasannya setara acak dan konsisten
dengan Mutual Information yang mendekati nol di Phase 1.

### 5.5 Ringkasan standar dibanding modified

| Metode | Standar | Modified | Arah dan alasan |
|---|---|---|---|
| IQR | 1.131 (22,6%) | 880 (17,6%) | pagar mengikuti skew, jadi ekor kanan yang wajar tidak ter-flag |
| Z-Score | 264 (5,3%) | 1.070 (21,4%) | MAD tidak membengkak oleh ekor, jadi penyimpangan nyata tertangkap |
| Isolation Forest | 250 (5,0%) | 25 (0,5%) | threshold diambil dari gap alami skor, bukan asumsi 5% |

## 6. Phase 5, Visualization dan Knowledge Presentation (`src/dashboard.py`)

### 6.1 Struktur

`load_dashboard_data` membaca dua sumber. Sumber pertama `dataset_final.csv`
berisi 5.000 baris gabungan hasil Phase 1 sampai 4. Sumber kedua `rules_all.csv`
berisi semua 82 rule, dengan fallback ke `top_rules.csv`. KPI Association Rules
dihitung dari file lengkap supaya konsisten dengan angka laporan.

`build_app` merakit layout dan callback. Fungsi ini sengaja dipisah dari
`run_dashboard` yang menyalakan server, supaya aplikasi bisa di-*smoke-test*
tanpa server lewat `from src.dashboard import build_app`.

Setiap grafik punya figure builder terpisah seperti `fig_cluster_map`,
`fig_segment_demographics`, `fig_rule_network`, dan `fig_anomaly_scatter`.
Masing-masing adalah fungsi murni yang memetakan data menjadi objek figure,
sehingga mudah diuji.

### 6.2 KPI dan interaktivitas

Lima KPI dihitung langsung dari data saat startup dan bukan di-hardcode, yaitu
total nasabah, jumlah segmen, 82 rule, 446 anomali konsensus, dan 255 risk
signal. Ada tiga tab interaktif.

| Tab | Kontrol (Input) | Grafik (Output) |
|---|---|---|
| Segmentasi | metode clustering, sumbu X dan Y, fitur demografis | cluster map log-log, pie proporsi, profil rasio, komposisi demografis, validasi feature-selection statis |
| Association Rules | slider minimum lift | network rules dengan spring-layout seed tetap, scatter support dan confidence, tabel rules sortable |
| Anomali | sumbu X dan Y scatter | breakdown klasifikasi, anomali per segmen, scatter anomali |

Ada satu detail kecil yang penting. Nilai maksimum slider lift di-*floor* ke dua
desimal dan bukan di-*round*. Bila di-round ke atas, posisi slider tertinggi akan
menyaring semua rule sehingga grafik menjadi kosong.

## 7. Reproducibilitas

### 7.1 Menjalankan ulang dari nol

```bash
pip install -r requirements.txt        # pandas, scikit-learn, mlxtend, dash, dll.
python main.py --phase all             # Phase 1 sampai 4 berurutan
python main.py --phase 5               # dashboard di http://127.0.0.1:8050
```

Urutan fase bersifat wajib. Phase 2 membutuhkan output Phase 1, Phase 4
membutuhkan output Phase 2, dan dashboard membutuhkan output Phase 3 serta 4.
Semua proses acak memakai `RANDOM_STATE=42` sehingga angka di laporan ini dapat
direproduksi secara identik.

### 7.2 Perkiraan waktu eksekusi

| Fase | Perkiraan | Komponen dominan |
|---|---|---|
| Phase 1 | di bawah 1 menit | plotting EDA |
| Phase 2 | sekitar 6 sampai 7 menit | exhaustive search 286 kombinasi dengan silhouette penuh sekitar 5 menit, ditambah t-SNE sekitar 30 sampai 60 detik |
| Phase 3 | di bawah 1 menit | loop auto-tune Apriori |
| Phase 4 | di bawah 1 menit | medcouple O(n²) tervektorisasi dan Isolation Forest 200 tree |

### 7.3 Parameter kunci

| Parameter | Nilai | Lokasi | Alasan |
|---|---|---|---|
| `WINSOR_LIMIT` | 0,02 | config, clustering | menutup ekor atas 2% supaya cluster digerakkan perilaku dan bukan 1 sampai 2 outlier ekstrem |
| `BEST_K` | 3 | config, clustering | silhouette tinggi ditambah segmen ketiga yang actionable, justifikasi ada di `find_optimal_k` |
| `DBSCAN_MIN_SAMPLES` | 10 | config, clustering | ukuran minimum core-neighborhood yang stabil pada n=5.000 |
| `MIN_LIFT` dan `MIN_CONFIDENCE` | 1,4 dan 0,5 | arm | ambang kualitas rule yang tidak dilonggarkan saat auto-tune |
| `min_support` hasil tune | 0,01 | arm (otomatis) | support terbesar yang masih menghasilkan minimal 10 rule |
| `MODZ_THRESH` | 3,5 | config, anomaly | ambang baku modified z-score mengikuti Iglewicz dan Hoaglin |
| `GAP_MIN_FRAC` dan `GAP_MAX_FRAC` | 0,005 dan 0,15 | config, anomaly | jendela pencarian gap Isolation Forest untuk mencegah cut degeneratif |
| `Z_THRESH` dan `IF_CONTAMINATION` | 3,0 dan 0,05 | config, anomaly | dipertahankan hanya sebagai pembanding versi standar |

### 7.4 Keterbatasan teknis yang disadari

Silhouette dan medcouple sama-sama berbiaya O(n²). Biaya ini masih nyaman pada
n=5.000. Untuk dataset yang jauh lebih besar diperlukan sampling, dan kode sudah
menyediakan parameter `sample`, atau implementasi medcouple yang O(n log n).

Dendrogram di-sampel 1.000 baris hanya untuk keperluan visualisasi. Label
hierarchical final tetap dihitung pada seluruh data.

Dashboard memuat data saat startup. Bila pipeline dijalankan ulang, dashboard
perlu di-restart karena refresh browser tidak memuat ulang CSV.

Jendela gap Isolation Forest 0,5% sampai 15% adalah pilihan desain, dan di luar
rentang itu cut dianggap degeneratif. Sensitivitasnya rendah pada data ini karena
gap terpilih bernilai sekitar 66 kali gap tipikal.
