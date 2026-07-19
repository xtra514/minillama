from dataclasses import dataclass

@dataclass
class MiniLlamaConfig:
    vocab_size: int = 16000
    hidden_size: int = 256
    num_hidden_layers: int = 8
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    intermediate_size: int = None # Calculated automatically if None
    max_position_embeddings: int = 1024
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    
    def __post_init__(self):
        if self.intermediate_size is None:
            # Llama calculates hidden size as multiple_of(int(hidden_size * 8 / 3))
            hidden_dim = int(self.hidden_size * (8 / 3))
            multiple_of = 256
            self.intermediate_size = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

# Preset Configurations
CONFIG_10M = MiniLlamaConfig(
    hidden_size=384,
    num_hidden_layers=6,
    num_attention_heads=6,
    num_key_value_heads=6,
) # ~16.7M params (Optimized for LM quality rather than exact 10M)

CONFIG_50M = MiniLlamaConfig(
    hidden_size=512,
    num_hidden_layers=10,
    num_attention_heads=8,
    num_key_value_heads=8,
) # ~43M params

CONFIG_100M = MiniLlamaConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=12,
) # ~106M params
