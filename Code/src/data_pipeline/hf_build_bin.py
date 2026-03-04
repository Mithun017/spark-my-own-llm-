import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from src.utils.logger import setup_global_logger
setup_global_logger(BASE_DIR)

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

def build_binary_dataset(tokenizer_path, out_bin_path, max_tokens=100_000_000): 
    print(f"Loading Tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CACHE_DIR = os.path.join(BASE_DIR, 'data', 'full_data')
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    print(f"Streaming OpenWebText for Mass Binary Tokenization...")
    # We explicitly route the download to the user's project folder instead of the C:\ drive root
    dataset = load_dataset("openwebtext", split="train", streaming=True, cache_dir=CACHE_DIR)
    
    os.makedirs(os.path.dirname(out_bin_path), exist_ok=True)
    
    # We use uint16 since our vocab size is 8000 (< 65535, saving 50% RAM compared to int32)
    dtype = np.uint16
    
    # We write in chunks to avoid memory bottlenecks
    chunk_size = 1024 * 1024 # 1 million tokens at a time
    arr = np.empty(chunk_size, dtype=dtype)
    arr_idx = 0
    
    total_tokens_written = 0
    
    with open(out_bin_path, 'wb') as f:
        for item in dataset:
            text = item['text']
            encoded = tokenizer.encode(text).ids
            
            for token_id in encoded:
                arr[arr_idx] = token_id
                arr_idx += 1
                
                if arr_idx == chunk_size:
                    f.write(arr.tobytes())
                    total_tokens_written += chunk_size
                    arr_idx = 0
                    print(f"Written {total_tokens_written:,} tokens to binary format...")
                    
            if total_tokens_written >= max_tokens:
                break
                
        # write remainder
        if arr_idx > 0 and total_tokens_written < max_tokens:
            f.write(arr[:arr_idx].tobytes())
            total_tokens_written += arr_idx
            
    print(f"Successfully compiled {total_tokens_written:,} tokens into {out_bin_path}! Ready for memmap.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    TOKENIZER_PATH = os.path.join(BASE_DIR, 'data', 'tokenizer', 'spark_tokenizer.json')
    BIN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'train.bin')
    # For a real 5GB+ dataset, scale max_tokens to 2,000,000,000+
    # 500 Million tokens allows the 100M+ AWS parameter model to mathematically converge overnight
    build_binary_dataset(TOKENIZER_PATH, BIN_PATH, max_tokens=500_000_000)
