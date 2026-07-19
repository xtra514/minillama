import torch
import torch.nn as nn
from .config import MiniLlamaConfig

def count_parameters(model: nn.Module) -> int:
    """
    Returns the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_weights(module: nn.Module, config: MiniLlamaConfig):
    """
    Applies Llama-style weight initialization to the module.
    """
    std = 0.02
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "padding_idx", None) is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].fill_(0)

def apply_weight_init(model: nn.Module, config: MiniLlamaConfig):
    """
    Initialize all weights and apply residual scaling.
    """
    model.apply(lambda m: init_weights(m, config))
    
    # Scale down the residual projections (like output of MLP and Attention)
    # This helps stabilize training for deeper networks.
    # We find layers with 'wo' (attention output) or 'down_proj' (MLP output).
    scale = (2 * config.num_hidden_layers) ** -0.5
    for name, p in model.named_parameters():
        if "wo.weight" in name or "down_proj.weight" in name:
            p.data.mul_(scale)

class LRScheduler:
    """
    Cosine learning rate schedule with warmup.
    """
    def __init__(self, optimizer, warmup_steps, max_steps, max_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        
    def step(self, current_step):
        import math
        
        # 1. Warmup
        if current_step < self.warmup_steps:
            lr = self.max_lr * (current_step + 1) / self.warmup_steps
        # 2. End of training / after max_steps
        elif current_step > self.max_steps:
            lr = self.min_lr
        # 3. Cosine decay
        else:
            decay_ratio = (current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.min_lr + coeff * (self.max_lr - self.min_lr)
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
