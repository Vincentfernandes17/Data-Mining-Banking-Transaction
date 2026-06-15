# Knowledge Discovery Report
## Banking Transaction Dataset — Group 6

**Mata Kuliah:** Data Mining · **Metodologi:** KDD (Knowledge Discovery in Databases) 5 Fase
**Dataset:** Comprehensive Banking Database (5.000 nasabah, 40 atribut)
**Fokus domain:** Perilaku transaksi harian & demografi nasabah perbankan

> Catatan: laporan ini adalah **draft** yang ditulis dalam bahasa bisnis untuk
> audiens non-teknis. Angka, grafik, dan tabel pendukung dihasilkan otomatis
> oleh pipeline (`outputs/phase1`–`phase4`) dan dapat dieksplorasi langsung pada
> dashboard interaktif (`python main.py --phase 5`).

---

## 1. Ringkasan Eksekutif

Kami menerapkan proses penemuan pengetahuan lima fase pada data 5.000 nasabah
sebuah bank. Tujuannya bukan memprediksi sesuatu, melainkan **menemukan pola dan
kelompok tersembunyi** yang tidak terlihat dari tabel mentah.

Tiga temuan utama:

1. **Struktur data menipu.** Sebagian besar angka di dataset ini (saldo, jumlah
   transaksi, limit, dll.) ternyata **acak dan tidak saling berhubungan**. Pola
   nyata baru muncul ketika kami melihat **rasio antar-angka** — misalnya
   "seberapa besar transaksi dibanding saldo", bukan saldo atau transaksi secara
   terpisah.

2. **Nasabah terbagi menjadi tiga profil perilaku yang jelas**, bukan kelompok
   acak: mayoritas sehat, satu kelompok tertekan kartu kredit, dan satu kelompok
   dengan leverage/likuiditas berisiko.

3. **Risiko terkonsentrasi.** Lebih dari separuh (51,5%) nasabah pada segmen
   "Liquidity-Stressed" terdeteksi sebagai anomali perilaku — sinyal risiko yang
   sangat pekat dibanding rata-rata populasi.

---

## 2. Fase 1 — Pemahaman & Persiapan Data

**Apa yang dilakukan:** memeriksa kualitas data, membersihkan, merekayasa fitur,
melakukan binning, encoding, normalisasi, dan seleksi fitur (korelasi + ukuran
entropi/Mutual Information).

**Kondisi data:** bersih secara teknis — **0 nilai kosong** dan **0 baris
duplikat**. Setiap baris adalah satu nasabah unik (5.000 Customer ID unik).

**Temuan paling penting (menentukan seluruh proyek):**
Dari 55 pasang fitur numerik, **hanya 2 pasang yang benar-benar berkorelasi**, dan
keduanya bersifat turunan (mis. Credit Card Balance vs Minimum Payment Due,
korelasi 1,00 karena pembayaran minimum = persentase tetap dari saldo). Semua
fitur lain **independen** dan tersebar **nyaris merata** (skewness ≈ 0).

Artinya: angka-angka mentah ini seperti dikocok acak. Konsekuensinya besar untuk
fase berikutnya, dan kami mengubahnya menjadi keunggulan, bukan kelemahan.

**Rekayasa fitur kunci — 3 rasio perilaku:**

| Rasio | Rumus | Makna bisnis |
|-------|-------|--------------|
| `CC_Utilization` | saldo kartu ÷ limit | tekanan kartu kredit (>1 = over-limit) |
| `Transaction_to_Balance` | transaksi ÷ saldo | intensitas likuiditas |
| `Loan_to_Balance` | pinjaman ÷ saldo | leverage utang |

Berbeda dari angka mentah, rasio ini memiliki **struktur nyata** (skewness 2–7):
sebagian besar nasabah berkumpul di nilai rendah, sebagian kecil memanjang ke
nilai ekstrem. Justru di sinilah segmen dan risiko bersembunyi.

**Diskretisasi (binning) berbasis domain, bukan kuantil.**
Untuk Association Rule Mining, variabel kontinu diubah jadi kategori. Kami
**tidak** memakai pembagian "rata" (equal-frequency/equal-width) yang batasnya
bergeser mengikuti data dan tak bermakna. Sebagai gantinya, dipakai **ambang
tetap berbasis domain** sehingga kategori selalu konsisten & interpretable:

| Variabel | Ambang | Dasar |
|----------|--------|-------|
| **Umur** | 18–24 / 25–34 / 35–49 / 50–64 / 65+ (Young Adult → Senior) | Tahap hidup finansial — *Life-Cycle Hypothesis* (Modigliani & Brumberg, 1954) dan temuan *hump-shaped* kecakapan finansial yang memuncak ~usia 53 (Agarwal, Driscoll, Gabaix & Laibson, 2009, *"The Age of Reason"*) |
| **Utilisasi kartu** | <30% / 30–70% / 70–100% / >100% | Pedoman credit-scoring (FICO): jaga utilisasi <30%; >100% = over-limit (sinyal risiko) |
| **Suku bunga** | <4% / 4–7% / >7% | Tier prime / standar / subprime |
| **Saldo, transaksi, pinjaman** | tier nominal bulat tetap | Mass-market→affluent; kategori ukuran pinjaman konsumen (tidak ada ambang regulatori universal untuk nominal sintetis) |

---

## 3. Fase 2 — Segmentasi via Clustering

### 3.1 Mengapa PCA Tidak Kami Pakai (menjawab kritik umum)

Reduksi dimensi dengan PCA hanya berguna jika fitur saling tumpang-tindih
(redundan). Karena fitur di dataset ini independen, **setiap komponen PCA hanya
menangkap ~11% variance** (1 dari 9) — grafik scree-nya **datar**. Untuk mencapai
80% variance dibutuhkan 8 dari 9 komponen; "reduksi" 11→9 tidak ada gunanya.

Kami tetap menjalankan PCA **satu kali** semata-mata untuk **membuktikan** hal ini
secara visual (`outputs/phase2/pca_why_not_used.png`), lalu **membuangnya** dan
melakukan clustering langsung pada 3 rasio perilaku.

**Dampak terukur:** Silhouette Score (ukuran kualitas cluster, 0–1) melonjak dari
**~0,07** (fitur mentah, praktis tanpa struktur) menjadi **~0,57** (rasio
perilaku). Ini perbedaan antara "tidak ada cluster" dan "cluster yang jelas".

### 3.2 Tiga Algoritma, Satu Kesimpulan

| Metode | Jumlah Cluster | Silhouette | Catatan |
|--------|----------------|-----------:|---------|
| **K-Means** (K=3) | 3 | **0,571** | segmen seimbang & interpretable |
| **DBSCAN** (auto-eps) | 4 + noise | 0,578 | mengisolasi 274 outlier (5,5%) sebagai *noise* |
| **Hierarchical** (Ward) | 3 | 0,459 | mengonfirmasi 3 segmen yang sama |

Pemilihan K=3 (bukan K=2 yang skornya lebih tinggi, 0,68) adalah **keputusan
domain**: K=2 hanya memisahkan kelompok over-limit dari sisanya, sedangkan K=3
memunculkan segmen ketiga (liquidity-stressed) yang **lebih actionable** secara
bisnis — dan silhouette-nya tetap tinggi (0,57).

### 3.3 Profil Tiga Segmen (nilai median, satuan asli)

| Segmen | Porsi | Utilisasi Kartu | Transaksi/Saldo | Pinjaman/Saldo | Saldo | Limit |
|--------|------:|----------------:|----------------:|---------------:|------:|------:|
| **Mainstream / Balanced** | 79,8% | 0,38 | 0,47 | 4,7× | 5.461 | 6.178 |
| **Credit-Stressed / Over-Limit** | 13,1% | **1,72** | 0,47 | 4,7× | 5.198 | 1.929 |
| **Liquidity-Stressed / High-Leverage** | 7,2% | 0,43 | **5,0×** | **52,6×** | **550** | 5.763 |

**Cerita di balik angka:**
- **Mainstream** — nasabah sehat: rasio rendah, saldo & limit tinggi. Ini "tulang
  punggung" bank.
- **Credit-Stressed / Over-Limit** — saldo kartu rata-rata **1,7× limit** dengan
  limit yang justru kecil. Mereka tidak miskin (saldo tabungan normal) tetapi
  **kronis melebihi batas kartu** → kandidat penyesuaian limit / edukasi kredit.
- **Liquidity-Stressed / High-Leverage** — saldo sangat tipis (median 550) tetapi
  bertransaksi **5×** dan berutang **52×** dari saldonya. Kelompok kecil (7%) tapi
  **paling rapuh**.

---

## 4. Fase 3 — Association Rule Mining

**Apa yang dilakukan:** mengubah atribut kontinu menjadi kategori bermakna, lalu
menjalankan algoritma **Apriori** untuk menemukan pola "jika–maka" antar atribut,
dinilai dengan **Support, Confidence, dan Lift**.

Kami menemukan **15 aturan non-trivial** (Lift 1,47–1,59; Confidence ≥ 0,50).
Contoh aturan dengan interpretasi:

| # | JIKA | MAKA | Lift |
|---|------|------|-----:|
| 1 | Senior + KPR + pinjaman sangat besar | status pinjaman *Closed* | 1,59 |
| 2 | Saldo menengah-bawah + Visa + pinjaman disetujui | bunga tinggi | 1,57 |
| 10 | Visa + pinjaman sangat besar + transaksi besar | utilisasi kartu sedang | 1,47 |

**Kejujuran metodologis:** karena atribut kategorikal di dataset ini hampir
independen (lihat Fase 1), nilai **Lift tertinggi pun hanya ~1,5** — ini *batas
data*, bukan kekurangan analisis. Kami melaporkan pola terkuat yang ada, sambil
menegaskan bahwa pola ini sebaiknya divalidasi pada data perbankan riil sebelum
dijadikan kebijakan.

---

## 5. Fase 4 — Deteksi Anomali & Outlier

**Apa yang dilakukan:** tiga metode dijalankan dan dibandingkan secara sistematis,
lalu disilangkan dengan hasil Fase 2.

| Metode | Record ter-flag | Sudut pandang |
|--------|----------------:|---------------|
| **IQR** | 1.131 (22,6%) | outlier per fitur (kuartil) |
| **Z-Score** | 264 (5,3%) | outlier per fitur (simpangan baku) |
| **Isolation Forest** | 250 (5,0%) | outlier multivariat/struktural |

**Konsensus:** 322 nasabah ditandai oleh **≥2 metode**, dan **192** oleh ketiga
metode sekaligus — ini anomali paling meyakinkan.

**Cross-reference dengan Fase 2:**
- **136 nasabah** ditandai anomali oleh metode statistik **dan** oleh DBSCAN
  (density) — dua pendekatan berbeda yang sepakat → keyakinan tertinggi.
- **51,5%** segmen *Liquidity-Stressed* dan **21,0%** segmen *Credit-Stressed*
  adalah anomali, vs hanya beberapa persen pada Mainstream. **Risiko terkonsentrasi
  di dua segmen kecil.**

**Klasifikasi tiap anomali** (semua 5.000 nasabah):

| Klasifikasi | Jumlah | Arti |
|-------------|-------:|------|
| Normal | 3.869 | tidak menyimpang |
| Rare but Valid | 401 | menyimpang tapi masih wajar |
| Data Error / Quality | 478 | perlu verifikasi (mis. saldo akhir tak konsisten/negatif) |
| **Risk Signal** | **252** | leverage/likuiditas ekstrem → **eskalasi** |

**Catatan kualitas data:** 34,1% transaksi memiliki saldo-akhir yang tidak
konsisten dengan tipe & jumlah transaksinya. Karena tersebar sangat merata, ini
kemungkinan **artefak sistemik dataset sintetis**, bukan kesalahan per-record —
sehingga kami laporkan terpisah, bukan sebagai 1.703 "error" individual.

**Validasi label:** dataset menyertakan kolom `Anomaly` bawaan. Sesuai aturan
proyek, label ini **tidak dipakai saat mining**, hanya untuk validasi akhir.
Tingkat kehadiran label -1 di anomali kami (6,2%) ≈ rata-rata populasi (6,0%):
artinya label bawaan **acak dan tidak selaras** dengan perilaku finansial — justru
menguatkan bahwa anomali yang berarti adalah yang **berbasis perilaku**, bukan
label sintetis.

---

## 6. Pertanyaan Sentral: Apa yang Kami Temukan yang Tidak Terlihat dari Data Mentah?

1. **Data yang tampak "tak berpola" sebenarnya punya pola — di level rasio, bukan
   level angka.** Tabel mentah memperlihatkan saldo dan transaksi yang acak;
   hubungan *transaksi-terhadap-saldo* dan *pinjaman-terhadap-saldo* yang
   mengungkap perilaku sesungguhnya.

2. **Ada tiga arketipe nasabah yang tidak terlihat dari kolom mana pun secara
   tunggal.** Tidak ada satu kolom "tipe nasabah"; profil ini hanya muncul saat
   beberapa rasio dikombinasikan.

3. **Risiko tidak tersebar merata — ia menggumpal.** 7% nasabah (Liquidity-
   Stressed) menampung lebih dari separuh anomali perilaku. Ini tak akan terlihat
   tanpa menyilangkan hasil clustering dengan deteksi anomali.

4. **"Over-limit" adalah perilaku kronis sebuah segmen, bukan kejadian acak.** 13%
   nasabah secara konsisten memakai kartu di atas limit—pola yang tenggelam dalam
   rata-rata bila dilihat sekilas.

---

## 7. Jawaban untuk Mining Expo

- **Aturan paling mengejutkan?** Bahwa lift tertinggi pun hanya ~1,5 — temuan
  "mengejutkan"-nya justru adalah betapa **independen**-nya atribut dataset ini,
  yang memaksa kami berpindah dari angka mentah ke rasio perilaku.
- **Metode clustering paling interpretable?** **K-Means (K=3)**: tiga segmen
  bernama yang seimbang dan langsung bisa dinarasikan ke manajemen; DBSCAN unggul
  untuk *memisahkan* outlier, bukan untuk segmentasi utama.
- **Anomali apa yang ditemukan & artinya?** Nasabah dengan pinjaman & transaksi
  puluhan hingga ratusan kali lipat saldonya (252 *risk signal*). Dalam konteks
  bank nyata, ini sinyal **tekanan likuiditas / potensi gagal bayar** yang layak
  ditinjau.
- **Dibanding domain lain?** Karena dataset ini transaksional & sintetis, kekuatan
  kami ada di **segmentasi perilaku** dan **konsentrasi risiko**, bukan pada aturan
  asosiasi yang kuat (yang lebih cocok untuk domain seperti UCI Bank Marketing).

---

## 8. Rekomendasi Bisnis

1. **Tinjau ulang limit kartu** segmen *Credit-Stressed / Over-Limit* (13%):
   utilisasi kronis >100% pada limit kecil — peluang penyesuaian limit & edukasi.
2. **Pantau ketat** segmen *Liquidity-Stressed* (7%): leverage ekstrem; prioritas
   *early-warning* gagal bayar.
3. **Eskalasi 252 *risk signal*** untuk peninjauan manual.
4. **Audit kualitas data**: selidiki inkonsistensi saldo-akhir 34% pada sistem
   sumber.

---

## 9. Keterbatasan

Dataset bersifat **sintetis** dengan atribut yang sebagian besar independen.
Akibatnya: aturan asosiasi berlift moderat, dan "cluster alami" baru muncul setelah
rekayasa fitur — keduanya **didokumentasikan secara transparan**, bukan
disembunyikan. Temuan perilaku & risiko bersifat metodologis-valid dan siap
diuji ulang pada data perbankan riil.

---

*Pendukung: `outputs/phase1–4` (grafik & tabel), `outputs/phase3/top_rules.csv`,
`outputs/phase4/anomaly_report.csv`. Dashboard interaktif: `python main.py --phase 5`.*
