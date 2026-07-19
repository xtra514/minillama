import torch
import torch.nn as nn
import math
import os
from .model.transformer import MiniLlama
from .utils import LRScheduler, apply_weight_init
from .dataset import prepare_and_get_dataloaders
from .config import CONFIG_10M # We'll use the ~11M preset for now

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def configure_optimizers(model, weight_decay, learning_rate, device_type):
    decay_params = []
    no_decay_params = []
    
    for pn, p in model.named_parameters():
        if p.requires_grad:
            if p.dim() < 2:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    use_fused = (device_type == 'cuda')
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), **extra_args)
    return optimizer

def train():
    device = get_device()
    device_type = 'cuda' if 'cuda' in device.type else 'cpu'
    print(f"Using device: {device}")
    
    # Hyperparameters for Sweep Run 1.1
    micro_batch_size = 16
    gradient_accumulation_steps = 16 # Effective batch = 262,144 tokens
    max_length = 1024
    learning_rate = 8e-4
    max_steps = 2000
    warmup_steps = 200
    weight_decay = 0.1
    grad_clip = 1.0
    eval_interval = 100 # Evaluate more frequently during a short 2000-step run
    eval_iters = 100
    
    # Setup data
    train_loader, val_loader, tokenizer = prepare_and_get_dataloaders(batch_size=micro_batch_size, max_length=max_length)
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    
    # Setup model
    config = CONFIG_10M
    config.vocab_size = tokenizer.vocab_size
    model = MiniLlama(config)
    apply_weight_init(model, config)
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    optimizer = configure_optimizers(model, weight_decay, learning_rate, device_type)
    scheduler = LRScheduler(optimizer, warmup_steps, max_steps, learning_rate, learning_rate * 0.1)
    
    scaler = torch.amp.GradScaler(device_type, enabled=(device_type == 'cuda'))
    ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device_type == 'cuda' else torch.autocast(device_type=device_type) if device_type == 'mps' else open('os.devnull', 'w')
    
    if not hasattr(ctx, '__enter__'):
        from contextlib import nullcontext
        ctx = nullcontext()

    print("Starting training...")
    for step in range(max_steps):
        # Evaluation phase
        if step % eval_interval == 0 or step == max_steps - 1:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for i in range(eval_iters):
                    try:
                        X, Y = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        X, Y = next(val_iter)
                    X, Y = X.to(device), Y.to(device)
                    with ctx:
                        logits, loss = model(X, Y)
                        if loss.dim() > 0:
                            loss = loss.mean()
                    val_loss += loss.item()
                val_loss /= eval_iters
                print(f"Step {step} | val loss {val_loss:.4f}")
            
            # Save checkpoint
            if step > 0:
                model_to_save = model.module if hasattr(model, "module") else model
                torch.save(model_to_save.state_dict(), f"minillama_step_{step}.pt")
            model.train()

        # Training phase with gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        for micro_step in range(gradient_accumulation_steps):
            try:
                X, Y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                X, Y = next(train_iter)
                
            X, Y = X.to(device), Y.to(device)
            
            with ctx:
                logits, loss = model(X, Y)
                if loss.dim() > 0:
                    loss = loss.mean()
                loss = loss / gradient_accumulation_steps
                
            scaler.scale(loss).backward()
            accum_loss += loss.item() * gradient_accumulation_steps

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        lr = scheduler.step(step)
        
        if step % 5 == 0:
            print(f"Step {step} | train loss {accum_loss:.4f} | grad_norm {torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0).item():.4f} | lr {lr:.4e}")

if __name__ == "__main__":
    train()
