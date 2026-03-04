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
        
        from tokenizers import decoders
        TOKENIZER_PATH = os.path.join(BASE_DIR, 'data', 'tokenizer', 'spark_tokenizer.json')
        self.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        self.tokenizer.decoder = decoders.BPEDecoder()
        
        if model_path is None:
            model_name = 'spark_llm_instruct.pt' if use_instruct else 'spark_llm_weights.pt'
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
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.7, top_k: int = 10, repetition_penalty: float = 1.2):
        # 1. Format prompt matching the SFT training template
        formatted_prompt = f"Instruction: {prompt}\nResponse: "
        
        # 2. Tokenize prompt
        encoded = self.tokenizer.encode(formatted_prompt)
        idx = torch.tensor(encoded.ids, dtype=torch.long, device=self.device).unsqueeze(0) # (1, T)
        
        # Dynamically find the exact integer ID for the [EOS] token
        eos_token_id = self.tokenizer.token_to_id("[EOS]")
        
        print(f"\n[Prompt]: {prompt}")
        print("[SPARK]: ", end="", flush=True)
        
        # 3. Autoregressive Loop
        for _ in range(max_new_tokens):
            # Crop to context window
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            
            # Forward pass
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] # focus on last time step only
            
            # 4. Apply Repetition Penalty to discourage endless looping
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

            # 5. Temperature scaling
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
            
            # 6. Strict Stop Token condition (Halts generation cleanly)
            if next_token.item() == eos_token_id:
                break
                
        print("\n")
        # Return only the newly generated text by slicing off the prompt template
        return self.tokenizer.decode(idx[0, encoded.ids.__len__():].tolist())

if __name__ == "__main__":
    generator = SparkGenerator()
    generator.generate("Who created you?", max_new_tokens=40, temperature=0.7, top_k=5)
