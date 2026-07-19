import torch
import torch.nn as nn
from ..config import MiniLlamaConfig
from .rmsnorm import RMSNorm
from .attention import MultiHeadAttention
from .swiglu import SwiGLUFeedForward
from .rope import precompute_freqs_cis

class TransformerBlock(nn.Module):
    def __init__(self, config: MiniLlamaConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = SwiGLUFeedForward(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size
        )
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor = None):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class MiniLlama(nn.Module):
    def __init__(self, config: MiniLlamaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            theta=config.rope_theta
        )

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        batch_size, seq_len = input_ids.shape
        
        # Move freqs_cis to the right device if needed
        if self.freqs_cis.device != input_ids.device:
            self.freqs_cis = self.freqs_cis.to(input_ids.device)
            
        freqs_cis = self.freqs_cis[:seq_len]
        
        x = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            x = layer(x, freqs_cis)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Shift targets or assume input_ids and targets are already aligned?
            # Standard practice: targets are provided aligned.
            loss = nn.functional.cross_entropy(logits.view(-1, self.vocab_size).float(), targets.view(-1))
            
        return logits, loss
