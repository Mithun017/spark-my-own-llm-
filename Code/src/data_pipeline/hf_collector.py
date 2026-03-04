import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from src.utils.logger import setup_global_logger
setup_global_logger(BASE_DIR)

from datasets import load_dataset

def collect_sample_for_tokenizer(out_path, sample_size=50000):
    print(f"Streaming HuggingFace Dataset for Tokenizer Sample...")
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CACHE_DIR = os.path.join(BASE_DIR, 'data', 'full_data')
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # OpenWebText is a massive 38GB dataset of high-quality internet text
    # We explicitly route the download to the user's project folder instead of the C:\ drive root
    dataset = load_dataset("openwebtext", split="train", streaming=True, cache_dir=CACHE_DIR)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        count = 0
        for item in dataset:
            text = item['text'].strip()
            if len(text) > 100:
                f.write(text + "\n\n")
                count += 1
            if count >= sample_size:
                break
    print(f"Saved {count} high-quality articles to {out_path} for Tokenizer Training.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    OUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'master_dataset.txt')
    collect_sample_for_tokenizer(OUT_PATH)
