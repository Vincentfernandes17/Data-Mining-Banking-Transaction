"""
main.py
Entry point untuk menjalankan seluruh pipeline data mining.

Cara pakai:
    python main.py --phase all         # jalankan semua phase
    python main.py --phase 1           # hanya Phase 1
    python main.py --phase 2           # hanya Phase 2
"""

import argparse
import os
from config import RAW_DATA_PATH, OUTPUT_DIR


def main(phase: str):
    # Pastikan folder outputs ada
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if phase in ('1', 'all'):
        print("\n🚀 Menjalankan Phase 1...\n")
        from src.preprocess import run_preprocessing
        df_encoded, df_clean = run_preprocessing(RAW_DATA_PATH)

    if phase in ('2', 'all'):
        print("\n🚀 Menjalankan Phase 2...\n")
        from src.clustering import run_clustering
        from config import CLUSTERING_DATA_PATH
        df_clustered = run_clustering(CLUSTERING_DATA_PATH)

    if phase in ('3', 'all'):
        print("\n🚀 Phase 3 (ARM) belum diimplementasi.\n")

    if phase in ('4', 'all'):
        print("\n🚀 Phase 4 (Anomaly Detection) belum diimplementasi.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Banking Data Mining Pipeline')
    parser.add_argument(
        '--phase',
        choices=['1', '2', '3', '4', 'all'],
        default='all',
        help='Phase yang dijalankan (default: all)'
    )
    args = parser.parse_args()
    main(args.phase)
