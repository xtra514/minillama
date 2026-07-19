import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import MiniLlamaConfig
from .rope import apply_rotary_emb

class MultiHeadAttention(nn.Module):
    def __init__(self, config: MiniLlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # For MHA, num_key_value_heads == num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        
        # Calculate head dimension
        self.head_dim = self.hidden_size // self.num_heads
        assert self.head_dim * self.num_heads == self.hidden_size, "hidden_size must be divisible by num_heads"
        
        self.wq = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, _ = x.shape
        
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        xq = xq.view(batch_size, seq_len, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # Transpose for attention: (batch_size, num_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # If KV heads != Q heads, we'd need to expand KV heads here (for GQA).
        # But we are using standard MHA as requested, so num_heads == num_kv_heads.
        
        # Flash Attention
        # is_causal is True if we don't pass a custom mask, else False (we handle it via mask if provided)
        # However, for training we usually rely on is_causal=True for standard autoregressive LM.
        # But if a kv-cache is used (generation), seq_len=1 for q, so is_causal=False.
        is_causal = (mask is None) and (seq_len > 1)
        
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=mask,
            is_causal=is_causal
        )
        
        # (batch_size, seq_len, hidden_size)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)
