import os
import sys

# Ensure src is in the python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.data_pipeline.collector import download_sample_data
from src.data_pipeline.cleaner import process_raw_data

def run_stage_1():
    print("=== [Stage 1] Data Collection & Pipeline ===")
    RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    
    # 1. Collect
    print("[1] Collecting data...")
    download_sample_data(RAW_DIR)
    
    # 2. Clean
    print("\n[2] Cleaning data...")
    process_raw_data(RAW_DIR, PROCESSED_DIR)
    print("=== Stage 1 Complete ===\n")

if __name__ == "__main__":
    run_stage_1()
