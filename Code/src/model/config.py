from dataclasses import dataclass

@dataclass
class SparkConfig:
    """
    Hyperparameters for the SPARK SLM.
    These are scaled down for educational/laptop-friendly training speeds.
    """
    vocab_size: int = 8000      # Size of the BPE tokenizer vocabulary
    d_model: int = 256          # INCREASED: 256 Embedding dimension for deeper reasoning
    n_heads: int = 8            # INCREASED: 8 Attention heads for better context
    n_layers: int = 6           # INCREASED: 6 Transformer blocks for deeper logic
    max_seq_len: int = 512      # INCREASED: 512 context window
    
    # Training Config
    batch_size: int = 16        # Lowered to 16 to prevent Out-Of-Memory on GPUs
    learning_rate: float = 3e-4
    epochs: int = 2
    dropout: float = 0.1
