import os
import urllib.request
import urllib.error

def download_sample_data(raw_dir: str):
    """
    Downloads a tiny sample dataset (like TinyShakespeare or a small Wiki extract)
    to establish a baseline for tokenizer and model training.
    """
    os.makedirs(raw_dir, exist_ok=True)
    
    # We'll use the classic tiny shakespeare dataset from Andrej Karpathy for base logic testing
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    out_path = os.path.join(raw_dir, "tiny_shakespeare.txt")
    
    if os.path.exists(out_path):
        print(f"Sample data already exists at {out_path}")
        return
        
    print(f"Downloading sample text data from {url}...")
    try:
        urllib.request.urlretrieve(url, out_path)
        print(f"Successfully downloaded to {out_path}")
    except urllib.error.URLError as e:
        print(f"Failed to download: {e}")
        # Create a dummy file if network fails
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("This is a dummy text file. The network request failed.\n")
            f.write("The sky is blue. The model predicts the next token.\n")
            f.write("A neural network is a mathematical function.\n")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
    download_sample_data(RAW_DIR)
