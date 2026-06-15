# Daftar Referensi — Diskretisasi (Binning) Berbasis Domain

Dokumen ini mengumpulkan sumber yang mendukung keputusan binning di Phase 1
([src/preprocess.py](src/preprocess.py), fungsi `bin_features`). Tujuannya: tiap
ambang **bukan angka asal**, melainkan punya dasar — entah teori/paper
peer-reviewed, regulasi, atau praktik industri. Jenis sumber ditandai:
**[PAPER]** peer-reviewed · **[REGULASI]** · **[INDUSTRI]**.

> Saran: sebelum submit final, verifikasi DOI/halaman tiap paper (tercantum) di
> Google Scholar — praktik akademik yang baik.

---

## 0. Metodologi: kenapa domain/fixed > equal-width / equal-frequency

Ini membela langsung kritik dosen ("jangan pukul rata qcut").

1. **[PAPER]** Dougherty, J., Kohavi, R., & Sahami, M. (1995). *Supervised and
   Unsupervised Discretization of Continuous Features.* Proceedings of the 12th
   International Conference on Machine Learning (ICML), 194–202.
   → Klasik. Membandingkan binning unsupervised (equal-width/equal-frequency)
   vs supervised/berbasis pengetahuan; menunjukkan equal-width/frequency sering
   kalah karena mengabaikan makna/struktur.

2. **[PAPER]** García, S., Luengo, J., Sáez, J. A., López, V., & Herrera, F.
   (2013). *A Survey of Discretization Techniques: Taxonomy and Empirical
   Analysis in Supervised Learning.* IEEE Transactions on Knowledge and Data
   Engineering, 25(4), 734–750.
   → Survei lengkap teknik diskretisasi; dasar untuk memilih metode yang tepat
   alih-alih default equal-frequency.

---

## 1. Umur → tahap hidup finansial (18–24 / 25–34 / 35–49 / 50–64 / 65+)

3. **[PAPER]** Modigliani, F., & Brumberg, R. (1954). *Utility Analysis and the
   Consumption Function: An Interpretation of Cross-Section Data.* In K. Kurihara
   (Ed.), Post-Keynesian Economics (hlm. 388–436). Rutgers University Press.
   → **Life-Cycle Hypothesis**: orang meminjam saat muda, menabung saat paruh
   baya, dis-saving saat tua → memotong umur di transisi tahap hidup itu
   bermakna finansial.

4. **[PAPER]** Agarwal, S., Driscoll, J. C., Gabaix, X., & Laibson, D. (2009).
   *The Age of Reason: Financial Decisions over the Life-Cycle and Implications
   for Regulation.* Brookings Papers on Economic Activity, 2009(2), 51–117.
   → Kecakapan finansial berbentuk **hump-shaped**, puncak ≈usia 53; muda & tua
   membayar bunga/fee lebih tinggi → membenarkan batas 35–49 (puncak) dan 50–64
   (mulai menurun).

---

## 2. Utilisasi Kartu Kredit (30% / 70% / 100%)

5. **[INDUSTRI]** Fair Isaac Corporation (FICO). *What's in my FICO Scores* —
   "Amounts Owed" ≈ 30% skor; pedoman umum menjaga utilisasi revolving < 30%.
   (myFICO Credit Education).
6. **[REGULASI]** Consumer Financial Protection Bureau (CFPB). *Credit reports &
   scores* — anjuran menjaga utilisasi rendah; >100% (over-limit) = sinyal risiko.

---

## 3. Suku Bunga → prime / standar / higher-priced (≤4% / 4–7% / >7%)

7. **[REGULASI]** Truth in Lending Act — Regulation Z, 12 CFR §1026.35
   (*Higher-Priced Mortgage Loans*) & §1026.32 (*HOEPA high-cost*). HMDA
   melaporkan pinjaman "higher-priced" bila APR ≥ APOR + 1,5pp (first-lien).
   → Dasar bahwa "suku bunga jauh di atas prime" adalah kategori risiko yang
   diakui regulator.
8. **[PAPER]** Demyanyk, Y., & Van Hemert, O. (2011). *Understanding the Subprime
   Mortgage Crisis.* Review of Financial Studies, 24(6), 1848–1880.
   → Karakterisasi pinjaman subprime (suku bunga lebih tinggi) vs prime.

---

## 4. Saldo Tabungan (Below-Minimum <$1.5rb / Mass-Market / Comfortable ≥$5rb)

9. **[INDUSTRI/REGULASI]** Ambang minimum-balance & fee-waiver perbankan ritel
   AS — ≈$1.500 saldo harian atau ≈$5.000 saldo rata-rata umum dipakai bank
   besar (mis. Chase Total Checking, Bank of America) untuk membebaskan biaya
   bulanan. (Disclosure produk bank; ringkasan CFPB tentang biaya rekening).
   → Below-Minimum = rawan biaya; Comfortable = mendekati ambang affluent (≈$10rb+).

---

## 5. Pinjaman (Small <$5rb / Medium / Large / Very Large >$35rb)

10. **[REGULASI]** Consumer Financial Protection Bureau (2017). *Payday, Vehicle
    Title, and Certain High-Cost Installment Loans* ("Small-Dollar Rule"),
    12 CFR Part 1041. → Mendefinisikan pinjaman small-dollar (lini kredit tipikal
    ≈$500); membenarkan kategori "Small".
11. **[PAPER]** Adams, W., Einav, L., & Levin, J. (2009). *Liquidity Constraints
    and Imperfect Information in Subprime Lending.* American Economic Review,
    99(1), 49–84. → Hubungan ukuran pinjaman, risiko, dan informasi pada kredit
    konsumen. (Batas ≈$35rb = plafon umum pinjaman tanpa agunan; di atasnya
    biasanya beragunan.)

---

## 6. Transaksi (Everyday <$1rb / Large / Very Large >$3rb)

12. **[REGULASI]** Bank Secrecy Act — *Currency Transaction Report* (CTR) untuk
    transaksi tunai > $10.000 (31 U.S.C. §5313); anti-*structuring*
    (31 U.S.C. §5324, FinCEN). → Ambang nominal transaksi adalah konsep
    pemantauan yang nyata di perbankan.
13. **[PAPER]** Jensen, R. I. T., Ferwerda, J., & Wewer, C. R. (2025).
    *Searching for Smurfs: Testing if Money Launderers Know Alert Thresholds.*
    Journal of Quantitative Criminology (online first). arXiv:2309.12704.
    → Bukti empiris (data bank Denmark) bahwa pelaku memecah transaksi tepat di
    bawah ambang nominal → membenarkan bahwa **binning berbasis nominal transaksi
    bermakna**. Catatan jujur: karena transaksi maksimum di dataset (≈$5rb) di
    bawah CTR $10rb, batas pasti sub-$10rb tetap interpretable.

---

## Cara menyitir di laporan
Contoh kalimat siap pakai:
> "Diskretisasi umur mengikuti kerangka *life-cycle* (Modigliani & Brumberg,
> 1954; Agarwal et al., 2009) alih-alih binning equal-frequency yang
> data-dependent (lihat kritik Dougherty et al., 1995; García et al., 2013)."
