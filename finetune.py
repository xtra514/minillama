import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from minillama.model.transformer import MiniLlama
from minillama.utils import LRScheduler
from minillama.config import CONFIG_10M
from transformers import PreTrainedTokenizerFast

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class InstructDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=1024):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.user_tokens = tokenizer.encode("<|user|>\nWrite a story with these parameters:\n")
        self.assistant_tokens = tokenizer.encode("\n<|assistant|>\n")
        self.eos_id = 2 # Assuming 2 is the EOS token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        
        # Parse TinyStoriesInstruct format
        if "Story:" in text:
            prompt_part, response_part = text.split("Story:", 1)
            prompt_part = prompt_part.strip()
            response_part = response_part.strip()
        else:
            prompt_part = "Write a story."
            response_part = text.strip()

        # Build prompt and response
        prompt_ids = self.user_tokens + self.tokenizer.encode(prompt_part) + self.assistant_tokens
        response_ids = self.tokenizer.encode(response_part) + [self.eos_id]

        input_ids = prompt_ids + response_ids
        
        # Mask prompt in loss
        targets = [-100] * len(prompt_ids) + response_ids

        # Truncate or pad
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            targets = targets[:self.max_length]
        else:
            pad_len = self.max_length - len(input_ids)
            # Use 0 for padding, ignoring in loss
            input_ids = input_ids + [0] * pad_len
            targets = targets + [-100] * pad_len

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

def train():
    device = get_device()
    device_type = 'cuda' if 'cuda' in device.type else 'cpu'
    print(f"Using device: {device}")
    
    # Hyperparameters for Fine-Tuning
    batch_size = 16
    gradient_accumulation_steps = 4 
    max_length = 512
    learning_rate = 3e-5 # Much lower for fine-tuning!
    max_steps = 1500 # Short instruction tuning
    warmup_steps = 100
    weight_decay = 0.05
    grad_clip = 1.0
    
    import glob
    
    # 1. Load Tokenizer
    # Auto-detect from Kaggle's /kaggle/input/ directory
    try:
        tokenizer_path = glob.glob("/kaggle/input/*/minillama/data/tokenizer.json")[0]
        checkpoint_path = glob.glob("/kaggle/input/*/minillama_step_1999.pt")[0]
    except IndexError:
        print("Could not find Phase 1 output! Did you click '+ Add Input' and attach your previous Notebook output?")
        return

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    
    # 2. Load Dataset
    print("Loading TinyStoriesInstruct...")
    ds = load_dataset("roneneldan/TinyStoriesInstruct", split="train")
    
    train_dataset = InstructDataset(ds, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_iter = iter(train_loader)
    
    # 3. Load Base Model
    print("Loading Base Model...")
    config = CONFIG_10M
    config.vocab_size = tokenizer.vocab_size
    model = MiniLlama(config)
    
    # We load the weights from Phase 1
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = LRScheduler(optimizer, warmup_steps, max_steps, learning_rate, learning_rate * 0.1)
    
    scaler = torch.amp.GradScaler(device_type, enabled=(device_type == 'cuda'))
    ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if torch.cuda.is_bf16_supported() else torch.autocast(device_type=device_type, dtype=torch.float16)

    print("Starting Fine-Tuning...")
    model.train()
    
    for step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        for _ in range(gradient_accumulation_steps):
            try:
                X, Y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                X, Y = next(train_iter)
                
            X, Y = X.to(device), Y.to(device)
            
            with ctx:
                logits, _ = model(X) # (B, S, V)
                # Shift logits and targets for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = Y[..., 1:].contiguous()
                
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1), 
                    ignore_index=-100
                )
                loss = loss / gradient_accumulation_steps
                
            scaler.scale(loss).backward()
            accum_loss += loss.item() * gradient_accumulation_steps

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        lr = scheduler.step(step)
        
        if step % 10 == 0:
            print(f"Step {step} | train loss {accum_loss:.4f} | lr {lr:.4e}")
            
        if step > 0 and step % 500 == 0 or step == max_steps - 1:
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(), f"minillama_instruct_step_{step}.pt")

if __name__ == "__main__":
    train()
