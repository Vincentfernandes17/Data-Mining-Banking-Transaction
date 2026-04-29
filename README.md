# Banking Data Mining Project

## Struktur Folder

```
banking_dm/
├── main.py                  ← Entry point, jalankan dari sini
├── config.py                ← Semua konstanta & konfigurasi
├── requirements.txt
├── src/
│   ├── preprocess.py        ← Phase 1: Data Understanding & Preprocessing
│   ├── clustering.py        ← Phase 2: Segmentation via Clustering
│   ├── arm.py               ← Phase 3: Association Rule Mining (TODO)
│   └── anomaly.py           ← Phase 4: Anomaly Detection (TODO)
├── data/
│   ├── Comprehensive_Banking_Database.csv   ← Raw dataset
│   ├── dataset_clustering.csv               ← Output Phase 1 → input Phase 2
│   └── dataset_arm.csv                      ← Output Phase 1 → input Phase 3
└── outputs/
    ├── correlation_matrix.png
    ├── mutual_information.png
    ├── elbow_silhouette.png
    ├── kmeans_visualization.png
    ├── kdistance_graph.png
    ├── dbscan_pca.png
    └── dendrogram_comparison.png
```

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
