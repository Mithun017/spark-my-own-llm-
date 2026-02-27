import os
import sys
import torch
from tokenizers import Tokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.model.config import SparkConfig
from src.model.transformer import SparkTransformer

class InstructionDataset:
    def __init__(self, tokenizer_path, config):
        self.config = config
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # A tiny mock instruction dataset for educational scaffolding
        self.QA_PAIRS = [
            ("What is SPARK?", "SPARK is a custom small language model built from scratch."),
            ("Who created you?", "I was constructed using PyTorch and Transformer architecture by a master engineer."),
            ("What is 2 + 2?", "The answer is 4."),
            ("Explain gravity.", "Gravity is a fundamental physical force that pulls objects with mass towards each other."),
            ("Write a poem about the sea.", "The ocean waves crash on the shore, a deep blue mystery evermore. With salty breeze and distant sails, the sea whispers its ancient tales."),
            ("What is Python?", "Python is a high-level programming language known for its readability and versatile libraries, heavily used in AI."),
            ("How do neural networks work?", "They use layers of interconnected nodes to perform matrix multiplications and learn patterns from data through backpropagation."),
            ("Hello!", "Hello! How can I assist you today?"),
            ("What is the capital of France?", "The capital of France is Paris."),
            ("Can you do math?", "Yes, I can compute basic mathematical expressions if you provide them."),
            ("Tell me a joke.", "Why did the AI cross the road? To optimize the path to the other side!"),
        ]
        
        self.parsed_data = []
        for q, a in self.QA_PAIRS:
            # We use an Instruction -> Response prompt template
            text = f"Instruction: {q}\nResponse: {a}[EOS]"
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
    PRETRAINED_PATH = os.path.join(BASE_DIR, 'src', 'model', 'spark_slm_weights.pt')
    SFT_SAVE_PATH = os.path.join(BASE_DIR, 'src', 'model', 'spark_slm_instruct.pt')
    
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
    iterations = 1500 # INCREASED: 1500 loops for heavy supervised fine tuning
    for iter in range(iterations):
        X, Y = dataset.get_batch()
        X, Y = X.to(device), Y.to(device)
        
        logits, loss = model(X, targets=Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if iter % 50 == 0:
            print(f"SFT Iteration {iter} | Loss: {loss.item():.4f}")
            
    print("Fine-tuning completed!")
    torch.save(model.state_dict(), SFT_SAVE_PATH)
    print(f"Instruct Model saved to {SFT_SAVE_PATH}")

if __name__ == "__main__":
    finetune_model()
