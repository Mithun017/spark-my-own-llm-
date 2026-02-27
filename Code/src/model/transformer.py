import torch
import torch.nn as nn
from src.model.config import SparkConfig
from src.model.modules import TransformerBlock

class SparkTransformer(nn.Module):
    def __init__(self, config: SparkConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # Token Embeddings
            wte = nn.Embedding(config.vocab_size, config.d_model),
            # Positional Embeddings
            wpe = nn.Embedding(config.max_seq_len, config.d_model),
            # Transformer Dropout
            drop = nn.Dropout(config.dropout),
            # N stacked Transformer Blocks
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
            # Final Layer Norm
            ln_f = nn.LayerNorm(config.d_model),
        ))
        
        # Final Output Head projecting back to vocabulary size
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tieing to save memory: word embeddings = output weights
        self.transformer.wte.weight = self.lm_head.weight
        
        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.max_seq_len, f"Cannot forward sequence of length {t}, block size is only {self.config.max_seq_len}"
        
        # Forward the token and positional embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, d_model)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, d_model)
        
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Forward through the blocks
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            # We are training, calculate loss
            logits = self.lm_head(x)
            # Reshape heavily for PyTorch cross entropy
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            # Inference-only optimization
            # we only care about the very last token in the sequence 
            logits = self.lm_head(x[:, [-1], :]) # (b, 1, vocab_size)
            return logits, None
