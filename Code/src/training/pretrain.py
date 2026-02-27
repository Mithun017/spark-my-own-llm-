import os
import sys
import torch
import math

# Try to import tokenizers from huggingface, if fails user might need to run tokenizer step
from tokenizers import Tokenizer

# Append src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.model.config import SparkConfig
from src.model.transformer import SparkTransformer

class DatasetLoader:
    def __init__(self, data_path, tokenizer_path, config):
        self.config = config
        print(f"Loading Tokenizer from {tokenizer_path}...")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        print(f"Loading Raw Data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        print("Tokenizing entire dataset... (this might take a moment)")
        # For memory efficiency, large datasets shouldn't be fully loaded into memory this way.
        # But for educational SLM scales (e.g., millions of chars), it is perfectly fine.
        encoded = self.tokenizer.encode(raw_text)
        self.tokens = torch.tensor(encoded.ids, dtype=torch.long)
        print(f"Dataset loaded. Total Tokens: {len(self.tokens):,}")
        
    def get_batch(self):
        # Generate random starting indices for the batch
        ix = torch.randint(len(self.tokens) - self.config.max_seq_len, (self.config.batch_size,))
        x = torch.stack([self.tokens[i:i+self.config.max_seq_len] for i in ix])
        y = torch.stack([self.tokens[i+1:i+self.config.max_seq_len+1] for i in ix])
        return x, y

def train_model():
    # Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting Training on Device: {device}")
    
    config = SparkConfig()
    
    # Paths
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'master_dataset.txt')
    TOKENIZER_PATH = os.path.join(BASE_DIR, 'data', 'tokenizer', 'spark_tokenizer.json')
    SAVE_PATH = os.path.join(BASE_DIR, 'src', 'model', 'spark_slm_weights.pt')
    
    # Load Data
    loader = DatasetLoader(DATA_PATH, TOKENIZER_PATH, config)
    
    # Initialize Model
    model = SparkTransformer(config)
    model.to(device)
    
    print(f"Model initialized. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    iterations = 5000  # INCREASED: 5000 loops for heavy deep learning convergence
    
    model.train()
    for iter in range(iterations):
        # Fetch data
        X, Y = loader.get_batch()
        X, Y = X.to(device), Y.to(device)
        
        # Forward pass
        logits, loss = model(X, targets=Y)
        
        # Backprop
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient Clipping (stabilizes training)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if iter % 50 == 0:
            print(f"Iteration {iter} | Loss: {loss.item():.4f}")
            
    print("Training loop finished!")
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Base Pretrained Model weights saved to {SAVE_PATH}")

if __name__ == "__main__":
    train_model()
