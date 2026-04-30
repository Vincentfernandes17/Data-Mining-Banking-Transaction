# Banking Data Mining Project

## Struktur Folder

```
banking_dm/
├── main.py                  ← Entry point, jalankan dari sini
<<<<<<< HEAD
├── config.py                ← Semua konstanta & konfigurasi
├── requirements.txt
├── src/
=======
├── config.py                ← Semua konstanta & konfigurasi (path, kolom, parameter)
├── requirements.txt         ← Semua library yang dibutuhkan
├── README.md
├── src/
│   ├── __init__.py
>>>>>>> a300db8a33ff98bb5e23be4d9afa31a4700dc7d7
│   ├── preprocess.py        ← Phase 1: Data Understanding & Preprocessing
│   ├── clustering.py        ← Phase 2: Segmentation via Clustering
│   ├── arm.py               ← Phase 3: Association Rule Mining (TODO)
│   └── anomaly.py           ← Phase 4: Anomaly Detection (TODO)
├── data/
<<<<<<< HEAD
│   ├── Comprehensive_Banking_Database.csv   ← Raw dataset
│   ├── dataset_clustering.csv               ← Output Phase 1 → input Phase 2
│   └── dataset_arm.csv                      ← Output Phase 1 → input Phase 3
└── outputs/
    ├── correlation_matrix.png
    ├── mutual_information.png
    ├── elbow_silhouette.png
    ├── kmeans_visualization.png
=======
│   ├── Comprehensive_Banking_Database.csv   ← Taruh dataset di sini
│   ├── dataset_clustering.csv               ← Output Phase 1 → input Phase 2
│   └── dataset_arm.csv                      ← Output Phase 1 → input Phase 3
└── outputs/
    ├── correlation_raw.png
    ├── correlation_matrix.png
    ├── mutual_information.png
    ├── elbow_silhouette.png
    ├── kmeans_pca.png
    ├── kmeans_tsne.png
>>>>>>> a300db8a33ff98bb5e23be4d9afa31a4700dc7d7
    ├── kdistance_graph.png
    ├── dbscan_pca.png
    └── dendrogram_comparison.png
```

<<<<<<< HEAD
## Cara Pakai di Google Colab

```python
# 1. Mount Drive dan clone/upload project
from google.colab import drive
drive.mount('/content/drive')

# 2. Pindah ke folder project
import os
os.chdir('/content/drive/My Drive/Colab Notebooks/banking_dm')

# 3. Install dependencies
!pip install mlxtend

# 4. Jalankan phase tertentu
!python main.py --phase 1   # Phase 1 saja
!python main.py --phase 2   # Phase 2 saja
!python main.py --phase all # Semua phase
```

## Atau Panggil Fungsi Langsung dari Notebook

```python
# Contoh Phase 1
from src.preprocess import run_preprocessing
df_encoded, df_clean = run_preprocessing('data/Comprehensive_Banking_Database.csv')

# Contoh Phase 2
from src.clustering import run_clustering
df_clustered = run_clustering('data/dataset_clustering.csv')
```
=======
---

## Setup di VSCode (Pertama Kali)

### 1. Extract zip dan buka di VSCode
```
Buka VSCode → File → Open Folder → pilih folder banking_dm
```

### 2. Buat virtual environment
```bash
# Di terminal VSCode (Ctrl+`)
python -m venv venv
```

### 3. Aktifkan virtual environment
```bash
# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 4. Install semua library
```bash
pip install -r requirements.txt
```

### 5. Taruh dataset di folder data/
```
banking_dm/
└── data/
    └── Comprehensive_Banking_Database.csv   ← copy file CSV ke sini
```

### 6. Pilih interpreter Python di VSCode
```
Ctrl+Shift+P → "Python: Select Interpreter" → pilih yang ada (venv)
```

---

## Cara Running

### Via Terminal VSCode
```bash
# Pastikan sudah di folder banking_dm dan venv aktif

python main.py --phase 1    # Phase 1: Preprocessing
python main.py --phase 2    # Phase 2: Clustering
python main.py --phase all  # Semua phase berurutan
```

### Via Notebook (jika pakai .ipynb di VSCode)
```python
import sys
sys.path.insert(0, '.')   # pastikan root project ada di path

from src.preprocess import run_preprocessing
from src.clustering import run_clustering

# Phase 1
df_encoded, df_clean = run_preprocessing('data/Comprehensive_Banking_Database.csv')

# Phase 2 (jalankan setelah Phase 1)
df_clustered = run_clustering('data/dataset_clustering.csv')
```

---

## Kalau Mau Ganti Parameter

Semua parameter ada di `config.py`, tidak perlu masuk ke dalam kode:

```python
# config.py

BEST_K       = 3     # ganti jumlah cluster K-Means
DBSCAN_EPS   = 0.5   # ganti radius DBSCAN
N_INIT       = 10    # ganti jumlah inisialisasi K-Means
```

---

## Output

Semua plot otomatis tersimpan di folder `outputs/` setiap kali pipeline dijalankan.
Dataset hasil preprocessing tersimpan di folder `data/`.
>>>>>>> a300db8a33ff98bb5e23be4d9afa31a4700dc7d7
