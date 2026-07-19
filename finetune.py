import torch
import torch.nn as nn
import os
import glob
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
        
        self.instruction_prefix = tokenizer.encode("Instruction:\nWrite a story with these parameters:\n")
        self.response_prefix = tokenizer.encode("\n\nResponse:\n")
        
        # Safely get EOS token ID
        self.eos_id = tokenizer.eos_token_id
        if self.eos_id is None:
            self.eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if self.eos_id is None:
            self.eos_id = 2 # fallback just in case

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
        prompt_ids = self.instruction_prefix + self.tokenizer.encode(prompt_part) + self.response_prefix
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
            input_ids = input_ids + [0] * pad_len
            targets = targets + [-100] * pad_len

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt_text="A little rabbit was hungry."):
    model.eval()
    instruction = f"Instruction:\nWrite a story with these parameters:\n{prompt_text}\n\nResponse:\n"
    input_ids = tokenizer.encode(instruction)
    
    x = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Generate up to 100 tokens to see how it looks
    for _ in range(100):
        logits, _ = model(x)
        logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)
        if next_token.item() == tokenizer.convert_tokens_to_ids("<|endoftext|>") or next_token.item() == 2:
            break
            
    generated_text = tokenizer.decode(x[0].tolist())
    print("\n" + "="*50)
    print("GENERATION SAMPLE:")
    print(generated_text)
    print("="*50 + "\n")
    model.train()

def train():
    device = get_device()
    device_type = 'cuda' if 'cuda' in device.type else 'cpu'
    print(f"Using device: {device}")
    
    batch_size = 16 # drop to 8 if OOM
    gradient_accumulation_steps = 4 
    max_length = 1024
    learning_rate = 3e-5 
    max_steps = 1500 
    warmup_steps = 100
    weight_decay = 0.05
    grad_clip = 1.0
    eval_interval = 100
    eval_iters = 50
    
    try:
        tokenizer_path = glob.glob("/kaggle/input/*/minillama/data/tokenizer.json")[0]
        checkpoint_path = glob.glob("/kaggle/input/*/minillama_step_1999.pt")[0]
    except IndexError:
        print("Could not find Phase 1 output! Did you click '+ Add Input' and attach your previous Notebook output?")
        return

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    
    # Use skeskinen/TinyStories-Instruct-hf because it groups documents properly per row
    print("Loading TinyStoriesInstruct...")
    ds = load_dataset("skeskinen/TinyStories-Instruct-hf", split="train")
    
    # Split for validation
    split = ds.train_test_split(test_size=0.01, seed=42)
    train_ds = split['train']
    val_ds = split['test']
    
    train_dataset = InstructDataset(train_ds, tokenizer, max_length=max_length)
    val_dataset = InstructDataset(val_ds, tokenizer, max_length=max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    
    print("Loading Base Model...")
    config = CONFIG_10M
    config.vocab_size = tokenizer.vocab_size
    model = MiniLlama(config)
    
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = LRScheduler(optimizer, warmup_steps, max_steps, learning_rate, learning_rate * 0.1)
    
    scaler = torch.amp.GradScaler(device_type, enabled=(device_type == 'cuda'))
    ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if torch.cuda.is_bf16_supported() else torch.autocast(device_type=device_type, dtype=torch.float16)

    print("Starting Fine-Tuning...")
    model.train()
    
    for step in range(max_steps):
        # Evaluation
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
                        logits, _ = model(X)
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = Y[..., 1:].contiguous()
                        loss = nn.functional.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)), 
                            shift_labels.view(-1), 
                            ignore_index=-100
                        )
                    val_loss += loss.item()
                val_loss /= eval_iters
                print(f"Step {step} | val loss {val_loss:.4f}")
                
            generate_sample(model.module if hasattr(model, "module") else model, tokenizer, device)
            model.train()

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
                logits, _ = model(X) 
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
