import os
import sys
import json
import torch
import time
from tokenizers import Tokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.model.config import SparkConfig
from src.model.transformer import SparkTransformer
from src.utils.logger import setup_global_logger

setup_global_logger(BASE_DIR)

def run_rlhf_tuning():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting Reinforcement Learning from Human Feedback (RLHF) on: {device}")
    
    config = SparkConfig()
    
    TOKENIZER_PATH = os.path.join(BASE_DIR, 'data', 'tokenizer', 'spark_tokenizer.json')
    RLHF_LOGS = os.path.join(BASE_DIR, 'data', 'rlhf_logs.json')
    SFT_MODEL_PATH = os.path.join(BASE_DIR, 'src', 'model', 'spark_llm_instruct.pt')
    RLHF_SAVE_PATH = os.path.join(BASE_DIR, 'src', 'model', 'spark_llm_rlhf.pt')
    
    if not os.path.exists(RLHF_LOGS):
        print("No RLHF logs found. Users must chat with the model and click 👍 or 👎 first.")
        return
        
    with open(RLHF_LOGS, 'r', encoding='utf-8') as f:
        logs = json.load(f)
        
    if not logs:
        print("No RLHF feedback has been logged yet.")
        return
        
    print(f"Loaded {len(logs)} Human feedback interactions. Processing...")
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    # Load Fine-Tuned Model
    model = SparkTransformer(config)
    if os.path.exists(SFT_MODEL_PATH):
        model.load_state_dict(torch.load(SFT_MODEL_PATH, map_location=device))
    else:
        print("Base instruction model not found, cannot apply RLHF.")
        return
        
    model.to(device)
    model.train()
    
    # Tiny learning rate so we don't destroy the base knowledge, just nudge behaviors
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    start_time = time.time()
    total_logs = len(logs)
    
    for idx, entry in enumerate(logs):
        prompt = entry["prompt"]
        response = entry["response"]
        score = entry["score"] # +1 for Thumbs Up, -1 for Thumbs Down
        
        # Format the interaction
        text = f"Instruction: {prompt}\nResponse: {response}[EOS]"
        encoded = tokenizer.encode(text)
        seq = torch.tensor(encoded.ids, dtype=torch.long, device=device)
        
        # Prevent length crash
        seq = seq[:config.max_seq_len + 1]
        
        if len(seq) < 2:
            continue
            
        X = seq[:-1].unsqueeze(0) # Batch size 1
        Y = seq[1:].unsqueeze(0)
        
        logits, loss = model(X, targets=Y)
        
        # RLHF Mathematical Core (Simplified Educational PPO-style Reward Signal):
        # Normal training minimizes loss.
        # If score is +1, we want to minimize loss normally.
        # If score is -1, we want to mathematically maximize the loss to push the model AWAY from this answer.
        rlhf_loss = loss * -1.0 if score < 0 else loss
        
        optimizer.zero_grad()
        rlhf_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        elapsed = time.time() - start_time
        iters_per_sec = (idx + 1) / elapsed if elapsed > 0 else 0
        rem_iters = total_logs - (idx + 1)
        eta_sec = rem_iters / iters_per_sec if iters_per_sec > 0 else 0
        
        mins, secs = divmod(int(eta_sec), 60)
        hrs, mins = divmod(mins, 60)
        eta_str = f"{hrs:02d}h {mins:02d}m {secs:02d}s" if hrs > 0 else f"{mins:02d}m {secs:02d}s"
        
        percent = ((idx + 1) / total_logs) * 100
        print(f"RLHF Step {idx + 1}/{total_logs} [{percent:.1f}%] | ETA: {eta_str} | Reward Loss: {rlhf_loss.item():.4f}")
        
    print(f"RLHF Tuning applied successfully across {len(logs)} feedback instances.")
    torch.save(model.state_dict(), RLHF_SAVE_PATH)
    
    # Move the new heavily trained & aligned model into the Final distribution folder
    FINAL_MODEL_PATH = os.path.join(BASE_DIR, 'Final_output', 'model', 'spark_llm_instruct.pt')
    os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)
    os.system(f'copy /Y "{RLHF_SAVE_PATH}" "{FINAL_MODEL_PATH}"')
    print(f"Aligned Network pushed to {FINAL_MODEL_PATH}")
    
    # Clear the logs so we don't overtrain on the exact same feedback forever
    with open(RLHF_LOGS, 'w', encoding='utf-8') as f:
        json.dump([], f)
        
if __name__ == "__main__":
    run_rlhf_tuning()
