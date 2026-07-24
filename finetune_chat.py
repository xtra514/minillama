"""
finetune_chat.py
Phase 2: Fine-tune MiniLlama-125M on Alpaca for conversational ability.

Kaggle usage:
    python -m minillama.finetune_chat
"""
import torch
import torch.nn as nn
import os
import glob
from contextlib import nullcontext
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from minillama.model.transformer import MiniLlama
from minillama.utils import LRScheduler
from minillama.config import CONFIG_100M

import os
_REPO_ROOT      = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE      = 8
GRAD_ACCUM      = 4
MAX_LENGTH      = 1024
LEARNING_RATE   = 1e-5
MAX_STEPS       = 3_000
WARMUP_STEPS    = 100
WEIGHT_DECAY    = 0.05
GRAD_CLIP       = 1.0
EVAL_INTERVAL   = 200
EVAL_ITERS      = 30
TOKENIZER_PATH  = os.path.join(_REPO_ROOT, "data", "tokenizer_32k.json")
CHECKPOINT_GLOB = os.path.join(_REPO_ROOT, "minillama_125m_step_*.pt")

INSTRUCTION_PREFIX = "### Instruction:\n"
RESPONSE_PREFIX    = "\n### Response:\n"

# ── Dataset ──────────────────────────────────────────────────────────────────

class AlpacaDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=1024):
        self.data       = hf_dataset
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.eos_id     = tokenizer.token_to_id("</s>") or 2

        self.instr_prefix_ids = tokenizer.encode(
            INSTRUCTION_PREFIX, add_special_tokens=False).ids
        self.resp_prefix_ids  = tokenizer.encode(
            RESPONSE_PREFIX, add_special_tokens=False).ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        # Build instruction text (combine instruction + input if present)
        instruction = row["instruction"].strip()
        if row.get("input", "").strip():
            instruction += "\n" + row["input"].strip()
        response = row["output"].strip()

        prompt_ids   = (self.instr_prefix_ids
                        + self.tokenizer.encode(instruction, add_special_tokens=False).ids
                        + self.resp_prefix_ids)
        response_ids = (self.tokenizer.encode(response, add_special_tokens=False).ids
                        + [self.eos_id])

        input_ids = prompt_ids + response_ids
        # Mask prompt — only train on response tokens
        targets   = [-100] * len(prompt_ids) + response_ids

        # Truncate
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            targets   = targets[:self.max_length]
        else:
            pad = self.max_length - len(input_ids)
            input_ids += [0]    * pad
            targets   += [-100] * pad

        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(targets,   dtype=torch.long))


# ── Generation preview ───────────────────────────────────────────────────────

@torch.no_grad()
def generate_sample(model, tokenizer, device, instruction="Say hello to the user."):
    model.eval()
    prompt = INSTRUCTION_PREFIX + instruction + RESPONSE_PREFIX
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    x   = torch.tensor([ids], dtype=torch.long, device=device)
    eos = tokenizer.token_to_id("</s>") or 2

    generated = []
    for i in range(200):
        logits, _ = model(x)
        logits = logits[:, -1, :] / 0.8
        if i < 10:
            logits[:, eos] = float("-inf")
        next_tok = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        x = torch.cat([x, next_tok], dim=1)
        generated.append(next_tok.item())
        if next_tok.item() == eos:
            break

    response = tokenizer.decode(generated).strip()
    print(f"\n{'='*60}")
    print(f"INSTRUCTION: {instruction}")
    print(f"RESPONSE:    {response}")
    print(f"{'='*60}\n")
    model.train()


# ── Main ─────────────────────────────────────────────────────────────────────

def train():
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer  = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer: {vocab_size} tokens")

    # Load pretrained base model
    ckpts = sorted(glob.glob(CHECKPOINT_GLOB),
                   key=lambda p: int(p.split("_")[-1].replace(".pt", "")))
    if not ckpts:
        print(f"ERROR: No pretrained checkpoint matching {CHECKPOINT_GLOB}")
        print("Run pretrain_general.py first!")
        return
    ckpt_path = ckpts[-1]
    print(f"Loading base model from {ckpt_path}...")

    config            = CONFIG_100M
    config.vocab_size = vocab_size
    model = MiniLlama(config)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Missing: {missing} | Unexpected: {unexpected}")

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Load Alpaca
    print("Loading Alpaca dataset...")
    ds    = load_dataset("tatsu-lab/alpaca", split="train")
    split = ds.train_test_split(test_size=0.02, seed=42)
    train_dataset = AlpacaDataset(split["train"], tokenizer, MAX_LENGTH)
    val_dataset   = AlpacaDataset(split["test"],  tokenizer, MAX_LENGTH)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              drop_last=True, num_workers=2)
    train_iter = iter(train_loader)
    val_iter   = iter(val_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    scheduler = LRScheduler(optimizer, WARMUP_STEPS, MAX_STEPS,
                            LEARNING_RATE, LEARNING_RATE * 0.1)

    scaler = torch.amp.GradScaler(device_type, enabled=(device_type == "cuda"))
    if device_type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    else:
        ctx = nullcontext()

    print("Starting chat fine-tuning...")
    model.train()

    for step in range(MAX_STEPS):

        # Eval
        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(EVAL_ITERS):
                    try:
                        X, Y = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        X, Y = next(val_iter)
                    X, Y = X.to(device), Y.to(device)
                    with ctx:
                        logits, _ = model(X)
                        shift_logits = logits[:, :-1].contiguous()
                        shift_labels = Y[:, 1:].contiguous()
                        loss = nn.functional.cross_entropy(
                            shift_logits.view(-1, vocab_size),
                            shift_labels.view(-1),
                            ignore_index=-100
                        )
                    val_loss += loss.item()
                val_loss /= EVAL_ITERS
            print(f"Step {step} | val loss {val_loss:.4f}")
            raw = model.module if hasattr(model, "module") else model
            generate_sample(raw, tokenizer, device)
            model.train()

        # Train
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(GRAD_ACCUM):
            try:
                X, Y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                X, Y = next(train_iter)
            X, Y = X.to(device), Y.to(device)

            with ctx:
                logits, _ = model(X)
                shift_logits = logits[:, :-1].contiguous()
                shift_labels = Y[:, 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100
                ) / GRAD_ACCUM

            scaler.scale(loss).backward()
            accum_loss += loss.item() * GRAD_ACCUM

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        lr = scheduler.step(step)

        if step % 10 == 0:
            print(f"Step {step} | train loss {accum_loss:.4f} | lr {lr:.2e}")

        if (step > 0 and step % 500 == 0) or step == MAX_STEPS - 1:
            raw = model.module if hasattr(model, "module") else model
            torch.save(raw.state_dict(), f"minillama_125m_chat_step_{step}.pt")
            print(f"Saved: minillama_125m_chat_step_{step}.pt")


if __name__ == "__main__":
    train()
