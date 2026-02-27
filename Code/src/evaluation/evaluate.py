import os
import sys
import torch
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.model.config import SparkConfig
from src.model.transformer import SparkTransformer
from src.training.pretrain import DatasetLoader

@torch.no_grad()
def evaluate_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = SparkConfig()
    
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'master_dataset.txt')
    TOKENIZER_PATH = os.path.join(BASE_DIR, 'data', 'tokenizer', 'spark_tokenizer.json')
    MODEL_PATH = os.path.join(BASE_DIR, 'src', 'model', 'spark_slm_weights.pt')
    
    if not os.path.exists(MODEL_PATH):
        print("Pretrained model not found. Run pretraining first.")
        return
        
    print(f"Loading SPARK Core for Evaluation on {device}...")
    model = SparkTransformer(config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    # We evaluate over 10 batches
    loader = DatasetLoader(DATA_PATH, TOKENIZER_PATH, config)
    
    total_loss = 0.0
    eval_iters = 10
    
    for _ in range(eval_iters):
        X, Y = loader.get_batch()
        X, Y = X.to(device), Y.to(device)
        _, loss = model(X, targets=Y)
        total_loss += loss.item()
        
    avg_loss = total_loss / eval_iters
    perplexity = math.exp(avg_loss)
    
    print("\n--- Evaluation Metrics ---")
    print(f"Average Cross-Entropy Loss: {avg_loss:.4f}")
    print(f"Perplexity (PPL): {perplexity:.2f}")
    if perplexity > 500:
        print("Status: The model is essentially guessing randomly.")
    elif perplexity < 50:
        print("Status: The model strongly understands the data structure.")
    print("--------------------------\n")

if __name__ == "__main__":
    evaluate_model()
