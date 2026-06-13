"""
main.py — Entry Point Pipeline Data Mining Banking

Cara pakai di VSCode terminal:
    python main.py --phase 1    # Phase 1: Preprocessing
    python main.py --phase 2    # Phase 2: Clustering
    python main.py --phase 3    # Phase 2: ARM
    python main.py --phase 4    # Phase 2: Anomaly
    python main.py --phase all  # Semua phase berurutan
"""

import argparse
import os
import sys

# Konsol Windows default cp1252 tidak bisa mencetak karakter unicode
# (→, ✅, dsb) yang dipakai di log. Paksa stdout/stderr ke UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Tambahkan root project ke sys.path agar import src.* bisa jalan
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RAW_DATA_PATH, CLUSTERING_DATA_PATH, DATA_DIR, OUTPUT_DIR, ARM_DATA_PATH

# Buat folder jika belum ada
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main(phase: str):

    if phase in ('1', 'all'):
        print("\n" + "="*55)
        print("  Menjalankan Phase 1: Preprocessing")
        print("="*55 + "\n")
        from src.preprocess import run_preprocessing
        run_preprocessing(RAW_DATA_PATH)

    if phase in ('2', 'all'):
        print("\n" + "="*55)
        print("  Menjalankan Phase 2: Clustering")
        print("="*55 + "\n")
        from src.clustering import run_clustering
        run_clustering(CLUSTERING_DATA_PATH)

    if phase in ('3', 'all'):
        print("\n" + "="*55)
        print("  Menjalankan Phase 3: ARM")
        print("="*55 + "\n")
        from src.arm import run_arm
        run_arm(ARM_DATA_PATH)

    if phase in ('4', 'all'):
        print("\n" + "="*55)
        print("  Menjalankan Phase 4: Anomaly Detection")
        print("="*55 + "\n")
        from src.anomaly import run_anomaly
        run_anomaly()

    if phase == '5':
        print("\n" + "="*55)
        print("  Menjalankan Phase 5: Dashboard Interaktif")
        print("="*55 + "\n")
        from src.dashboard import run_dashboard
        run_dashboard()
    elif phase == 'all':
        print("\n[Phase 5] Dashboard tidak diluncurkan otomatis pada mode 'all' "
              "(server blocking). Jalankan: python main.py --phase 5")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Banking Data Mining Pipeline',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--phase',
        choices=['1', '2', '3', '4', '5', 'all'],
        default='all',
        help=(
            'Phase yang dijalankan:\n'
            '  1   = Preprocessing\n'
            '  2   = Clustering\n'
            '  3   = Association Rule Mining\n'
            '  4   = Anomaly Detection\n'
            '  5   = Dashboard Interaktif (server)\n'
            '  all = Semua phase berurutan'
        )
    )
    args = parser.parse_args()
    main(args.phase)
