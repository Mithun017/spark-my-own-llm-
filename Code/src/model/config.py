from dataclasses import dataclass

@dataclass
class SparkConfig:
    """
    Hyperparameters for the SPARK LLM AWS Cloud Architecture.
    """
    vocab_size: int = 50000     # Massively Expanded Vocab for perfect syntax and subword routing
    d_model: int = 1024         # CLOUD: 1024 Dim for extreme mathematical intelligence (100M+ params)
    n_heads: int = 16           # CLOUD: 16 Attention Heads to route logic gracefully across vectors
    n_layers: int = 12          # CLOUD: 12 Transformer Blocks to guarantee complex logical reasoning depth
    max_seq_len: int = 2048     # CLOUD: 2048 Context window size (Memory length)
    
    # Training Config
    batch_size: int = 4         # Keep physical batch size safe (A10G has 24GB, A100 has 80GB)
    learning_rate: float = 3e-4
    epochs: int = 2
    dropout: float = 0.1
