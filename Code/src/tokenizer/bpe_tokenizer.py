import os
import sys

# Ensure tokenizers library is installed
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
except ImportError:
    import subprocess
    print("Installing 'tokenizers' library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tokenizers"])
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.utils.logger import setup_global_logger
setup_global_logger(BASE_DIR)

def train_tokenizer(data_path: str, save_dir: str, vocab_size: int = 8000):
    """
    Trains a custom Byte Pair Encoding (BPE) tokenizer on the processed dataset.
    Saves the configuration and vocab so it can be used during model training.
    """
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please run Stage 1 first.")
        return

    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Training BPE Tokenizer with vocab size {vocab_size}...")
    
    # Initialize a clean BPE Tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # Pre-tokenize by whitespace
    tokenizer.pre_tokenizer = Whitespace()
    
    # Configure the trainer with special tokens
    trainer = BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        show_progress=True
    )
    
    # Train the tokenizer on the massive/cleaned dataset
    tokenizer.train(files=[data_path], trainer=trainer)
    
    # Save the tokenizer architecture
    out_file = os.path.join(save_dir, "spark_tokenizer.json")
    tokenizer.save(out_file)
    print(f"Tokenizer successfully trained and saved to {out_file}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'master_dataset.txt')
    SAVE_DIR = os.path.join(BASE_DIR, 'data', 'tokenizer')
    train_tokenizer(DATA_PATH, SAVE_DIR)
