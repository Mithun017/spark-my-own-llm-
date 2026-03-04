import os
import sys
import torch
import math
import time
import numpy as np

# Try to import tokenizers from huggingface, if fails user might need to run tokenizer step
from tokenizers import Tokenizer

# Append src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.model.config import SparkConfig
from src.model.transformer import SparkTransformer
from src.utils.logger import setup_global_logger

setup_global_logger(BASE_DIR)

class DatasetLoader:
    def __init__(self, bin_path, config):
        self.config = config
        print(f"Memmapping Massive Binary Data from {bin_path}...")
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.total_tokens = len(self.data)
        print(f"Dataset mapped off hard drive. Total Tokens: {self.total_tokens:,}")
        
    def get_batch(self):
        ix = torch.randint(self.total_tokens - self.config.max_seq_len - 1, (self.config.batch_size,))
        x = torch.stack([torch.from_numpy((self.data[i:i+self.config.max_seq_len]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+self.config.max_seq_len+1]).astype(np.int64)) for i in ix])
        return x, y

def train_model():
    # Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting Training on Device: {device}")
    
    config = SparkConfig()
    
    # Paths
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'train.bin')
    TOKENIZER_PATH = os.path.join(BASE_DIR, 'data', 'tokenizer', 'spark_tokenizer.json')
    SAVE_PATH = os.path.join(BASE_DIR, 'src', 'model', 'spark_llm_weights.pt')
    DEPLOY_PATH = os.path.join(BASE_DIR, 'Final_output', 'model', 'spark_llm_weights.pt')
    CHECKPOINT_PATH = os.path.join(BASE_DIR, 'src', 'model', 'spark_checkpoint.pt')
    
    # Load Data
    loader = DatasetLoader(DATA_PATH, config)
    
    # Initialize Model
    model = SparkTransformer(config)
    model.to(device)
    
    print(f"Model initialized. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    iterations = 20000  # Massively INCREASED for Huge Data scale-up
    accumulation_steps = 16 # Increased 16x to simulate original large batch_sizes on tiny 4x batches
    
    # Checkpoint Auto-Resume logic
    start_iter = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found active Checkpoint! Resuming Massive PyTorch Training from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['iter'] + 1
        print(f"Resumed at epoch {start_iter}")
        
    model.train()
    start_time = time.time()
    
    # Instantiating the GradScaler for Mixed Precision (AMP) saves 50% VRAM
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    for iter in range(start_iter, iterations):
        # Fetch data
        X, Y = loader.get_batch()
        X, Y = X.to(device), Y.to(device)
        
        # Forward pass using Automatic Mixed Precision (Float16)
        if device == 'cuda':
            with torch.amp.autocast('cuda'):
                logits, loss = model(X, targets=Y)
                loss = loss / accumulation_steps
        else:
            logits, loss = model(X, targets=Y)
            loss = loss / accumulation_steps
        
        # Backprop through Scaler
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient Accumulation Execution
        if (iter + 1) % accumulation_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            optimizer.zero_grad(set_to_none=True)
        
        # Intermittent Logging
        if iter % 50 == 0:
            elapsed = time.time() - start_time
            iters_per_sec = ((iter - start_iter) + 1) / elapsed if elapsed > 0 else 0
            rem_iters = iterations - iter
            eta_sec = rem_iters / iters_per_sec if iters_per_sec > 0 else 0
            
            mins, secs = divmod(int(eta_sec), 60)
            hrs, mins = divmod(mins, 60)
            eta_str = f"{hrs:02d}h {mins:02d}m {secs:02d}s" if hrs > 0 else f"{mins:02d}m {secs:02d}s"
            
            percent = (iter / iterations) * 100
            print(f"Epoch {iter}/{iterations} [{percent:.1f}%] | ETA: {eta_str} | Loss: {(loss.item() * accumulation_steps):.4f}")
            
        # Hard Drive Checkpoint Auto-Save
        if iter > 0 and iter % 2000 == 0:
            torch.save({
                'iter': iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, CHECKPOINT_PATH)
            print("Checkpoint Auto-Saved successfully! (Can safely close completely and resume later)")
            
    print("Training loop finished!")
    
    os.makedirs(os.path.dirname(DEPLOY_PATH), exist_ok=True)
    model_state = model.state_dict()
    
    torch.save(model_state, SAVE_PATH)
    torch.save(model_state, DEPLOY_PATH)
    print(f"Base Pretrained Model weights saved to {SAVE_PATH} and {DEPLOY_PATH}")

if __name__ == "__main__":
    train_model()
