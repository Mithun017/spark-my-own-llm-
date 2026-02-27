import os
import sys
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.model.config import SparkConfig
from src.model.transformer import SparkTransformer

class SparkGenerator:
    def __init__(self, model_path=None, use_instruct=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = SparkConfig()
        
        TOKENIZER_PATH = os.path.join(BASE_DIR, 'data', 'tokenizer', 'spark_tokenizer.json')
        self.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        
        if model_path is None:
            model_name = 'spark_slm_instruct.pt' if use_instruct else 'spark_slm_weights.pt'
            model_path = os.path.join(BASE_DIR, 'src', 'model', model_name)
            
        print(f"Loading SPARK Model from {model_path} onto {self.device}...")
        self.model = SparkTransformer(self.config)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("[Warning] Weights not found! Using completely random neural weights.")
            
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, top_k: int = 10):
        # 1. Tokenize prompt
        encoded = self.tokenizer.encode(prompt)
        idx = torch.tensor(encoded.ids, dtype=torch.long, device=self.device).unsqueeze(0) # (1, T)
        
        print(f"\n[Prompt]: {prompt}")
        print("[SPARK]: ", end="", flush=True)
        
        # 2. Autoregressive Loop
        for _ in range(max_new_tokens):
            # Crop to context window
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            
            # Forward pass
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] # focus on last time step only
            
            # Temperature scaling
            if temperature == 0.0:
                # Greedy decoding
                _, next_token = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                # Top-K Sampling
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
            # Append to sequence
            idx = torch.cat((idx, next_token), dim=1)
            
            # Print token out immediately
            decoded_char = self.tokenizer.decode([next_token.item()])
            print(decoded_char, end=" ", flush=True)
            
            # If EOS is hit (id usually differs based on BPE training, assuming 3 for example purposes)
            if next_token.item() == 3: # Assuming [EOS] is id 3, depends on vocab
                break
                
        print("\n")
        return self.tokenizer.decode(idx[0].tolist())

if __name__ == "__main__":
    generator = SparkGenerator()
    # If the model randomly initialized, this is gibberish. If trained, it will try to speak English.
    generator.generate("Once upon a time in a digital world,", max_new_tokens=40, temperature=0.8, top_k=5)
