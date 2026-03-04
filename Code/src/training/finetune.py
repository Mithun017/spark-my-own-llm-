import os
import sys
import torch
import time
from tokenizers import Tokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.model.config import SparkConfig
from src.model.transformer import SparkTransformer
from src.utils.logger import setup_global_logger

setup_global_logger(BASE_DIR)

from datasets import load_dataset

class InstructionDataset:
    def __init__(self, tokenizer_path, config):
        self.config = config
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        print("Fetching 'databricks/databricks-dolly-15k' Instruction Dataset for SFT...")
        # explicitly route cache away from C Drive
        cache_dir = os.path.join(BASE_DIR, 'data', 'full_data')
        self.dataset = load_dataset("databricks/databricks-dolly-15k", split="train", cache_dir=cache_dir)
        
        print(f"Parsing 15,000 Professional Human Instructions into Tensors...")
        self.parsed_data = []
        
        for row in self.dataset:
            instruction = row.get("instruction", "")
            context = row.get("context", "")
            response = row.get("response", "")
            
            # Map into the strict templating schema
            if context.strip():
                text = f"Instruction: {instruction}\nContext: {context}\nResponse: {response}[EOS]"
            else:
                text = f"Instruction: {instruction}\nResponse: {response}[EOS]"
                
            encoded = self.tokenizer.encode(text)
            self.parsed_data.append(torch.tensor(encoded.ids, dtype=torch.long))
            
    def get_batch(self):
        # We grab random QA pairs and pad them to max_seq_len
        ix = torch.randint(len(self.parsed_data), (self.config.batch_size,))
        
        X_batch, Y_batch = [], []
        for i in ix:
            seq = self.parsed_data[i]
            # Ensure sequence isn't too long
            seq = seq[:self.config.max_seq_len + 1]
            
            # Pad with 0s if too short
            if len(seq) < self.config.max_seq_len + 1:
                pad = torch.zeros(self.config.max_seq_len + 1 - len(seq), dtype=torch.long)
                seq = torch.cat([seq, pad])
                
            X_batch.append(seq[:-1])
            Y_batch.append(seq[1:])
            
        return torch.stack(X_batch), torch.stack(Y_batch)


def finetune_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting Instruction Fine-Tuning on: {device}")
    
    config = SparkConfig()
    
    TOKENIZER_PATH = os.path.join(BASE_DIR, 'data', 'tokenizer', 'spark_tokenizer.json')
    PRETRAINED_PATH = os.path.join(BASE_DIR, 'src', 'model', 'spark_llm_weights.pt')
    SFT_SAVE_PATH = os.path.join(BASE_DIR, 'src', 'model', 'spark_llm_instruct.pt')
    SFT_DEPLOY_PATH = os.path.join(BASE_DIR, 'Final_output', 'model', 'spark_llm_instruct.pt')
    
    # Load Model structure
    model = SparkTransformer(config)
    
    # Load Pretrained Weights
    if os.path.exists(PRETRAINED_PATH):
        print(f"Loading Base Pretrained weights from {PRETRAINED_PATH}")
        model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device))
    else:
        print("Warning: Pretrained weights not found. Fine-tuning from random initialization.")
        
    model.to(device)
    
    dataset = InstructionDataset(TOKENIZER_PATH, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate / 10) # Lower LR for finetuning
    
    model.train()
    iterations = 3000 # SCALED LLM: 3000 loops to fully adapt to the 15k instructions
    start_time = time.time()
    for iter in range(iterations):
        X, Y = dataset.get_batch()
        X, Y = X.to(device), Y.to(device)
        
        logits, loss = model(X, targets=Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if iter % 50 == 0:
            elapsed = time.time() - start_time
            iters_per_sec = (iter + 1) / elapsed if elapsed > 0 else 0
            rem_iters = iterations - iter
            eta_sec = rem_iters / iters_per_sec if iters_per_sec > 0 else 0
            
            mins, secs = divmod(int(eta_sec), 60)
            hrs, mins = divmod(mins, 60)
            eta_str = f"{hrs:02d}h {mins:02d}m {secs:02d}s" if hrs > 0 else f"{mins:02d}m {secs:02d}s"
            
            percent = (iter / iterations) * 100
            print(f"SFT Epoch {iter}/{iterations} [{percent:.1f}%] | ETA: {eta_str} | Loss: {loss.item():.4f}")
            
    print("Fine-tuning completed!")
    
    os.makedirs(os.path.dirname(SFT_DEPLOY_PATH), exist_ok=True)
    model_state = model.state_dict()
    
    torch.save(model_state, SFT_SAVE_PATH)
    torch.save(model_state, SFT_DEPLOY_PATH)
    print(f"Instruct Model saved to {SFT_SAVE_PATH} and {SFT_DEPLOY_PATH}")

if __name__ == "__main__":
    finetune_model()
