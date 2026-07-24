"""
pretrain_general.py
Phase 1: Pretrain MiniLlama-125M on OpenWebText.

Kaggle usage:
    python -m minillama.pretrain_general
    python -m minillama.pretrain_general --resume  # resume from last checkpoint
"""
import torch
import torch.nn as nn
import os
import glob
import argparse
from contextlib import nullcontext
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from minillama.model.transformer import MiniLlama
from minillama.utils import LRScheduler, apply_weight_init
from minillama.config import CONFIG_100M

# ── Hyperparameters ─────────────────────────────────────────────────────────
BATCH_SIZE          = 16
GRAD_ACCUM          = 8          # effective batch = 128 sequences = ~131K tokens
MAX_LENGTH          = 1024
LEARNING_RATE       = 6e-4
MIN_LR              = 6e-5
MAX_STEPS           = 20_000
WARMUP_STEPS        = 500
WEIGHT_DECAY        = 0.1
GRAD_CLIP           = 1.0
EVAL_INTERVAL       = 500
EVAL_ITERS          = 50
SAVE_INTERVAL       = 1_000
CHECKPOINT_PREFIX   = "minillama_125m_step"
TOKENIZER_PATH      = "minillama/data/tokenizer_32k.json"

# ── Streaming Dataset ────────────────────────────────────────────────────────

class OpenWebTextDataset(IterableDataset):
    """
    Streams OpenWebText, tokenizes, and packs into max_length chunks.
    No padding — pure token packing like GPT-2 training.
    """
    def __init__(self, tokenizer, max_length=1024, split="train", skip_bytes=0):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.split      = split
        self.skip_bytes = skip_bytes
        self.bos_id = tokenizer.token_to_id("<s>")  or 1
        self.eos_id = tokenizer.token_to_id("</s>") or 2

    def __iter__(self):
        ds = load_dataset("openwebtext", split=self.split, streaming=True, trust_remote_code=True)
        buffer = []

        for sample in ds:
            text = sample["text"].strip()
            if not text:
                continue
            # Encode without BOS/EOS (we add manually for packing)
            ids = self.tokenizer.encode(text, add_special_tokens=False).ids
            buffer.extend([self.bos_id] + ids + [self.eos_id])

            # Yield full chunks
            while len(buffer) >= self.max_length + 1:
                chunk = buffer[:self.max_length + 1]
                buffer = buffer[self.max_length + 1:]
                x = torch.tensor(chunk[:self.max_length], dtype=torch.long)
                y = torch.tensor(chunk[1:self.max_length + 1], dtype=torch.long)
                yield x, y


def build_loaders(tokenizer):
    train_ds = OpenWebTextDataset(tokenizer, MAX_LENGTH, split="train")
    # Use a small slice for validation (no official val split in openwebtext)
    # We approximate by taking the last 1% of the train stream conceptually.
    # In practice we just use a separate iterator that starts from the same data.
    val_ds   = OpenWebTextDataset(tokenizer, MAX_LENGTH, split="train")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=1)
    return train_loader, val_loader


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def find_latest_checkpoint():
    ckpts = sorted(glob.glob(f"{CHECKPOINT_PREFIX}_*.pt"),
                   key=lambda p: int(p.split("_")[-1].replace(".pt", "")))
    return ckpts[-1] if ckpts else None


def save_checkpoint(model, optimizer, step):
    raw = model.module if hasattr(model, "module") else model
    path = f"{CHECKPOINT_PREFIX}_{step}.pt"
    torch.save({
        "step":             step,
        "model_state_dict": raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"  ✓ Saved checkpoint: {path}")


# ── Generation preview ───────────────────────────────────────────────────────

@torch.no_grad()
def generate_preview(model, tokenizer, device, prompt="Hello! My name is"):
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    x   = torch.tensor([ids], dtype=torch.long, device=device)
    eos = tokenizer.token_to_id("</s>") or 2

    for _ in range(100):
        logits, _ = model(x)
        next_tok = torch.multinomial(
            torch.softmax(logits[:, -1, :] / 0.8, dim=-1), 1
        )
        x = torch.cat([x, next_tok], dim=1)
        if next_tok.item() == eos:
            break

    text = tokenizer.decode(x[0].tolist())
    print(f"\n{'='*60}\nPREVIEW: {text}\n{'='*60}\n")
    model.train()


# ── Main training loop ───────────────────────────────────────────────────────

def train(resume=False):
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        print(f"ERROR: Tokenizer not found at {TOKENIZER_PATH}")
        print("Run `python -m minillama.train_tokenizer` first!")
        return
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer loaded — vocab: {vocab_size}")

    # Build dataloaders
    train_loader, val_loader = build_loaders(tokenizer)
    train_iter = iter(train_loader)
    val_iter   = iter(val_loader)

    # Build model
    config            = CONFIG_100M
    config.vocab_size = vocab_size
    model = MiniLlama(config)
    apply_weight_init(model, config)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer (weight decay only on 2D params)
    decay     = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay  = [p for n, p in model.named_parameters() if p.dim() <  2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=LEARNING_RATE, betas=(0.9, 0.95), fused=(device_type == "cuda")
    )

    start_step = 0
    if resume:
        ckpt_path = find_latest_checkpoint()
        if ckpt_path:
            print(f"Resuming from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = ckpt["step"] + 1
            print(f"Resumed at step {start_step}")
        else:
            print("No checkpoint found — starting fresh.")

    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    scheduler = LRScheduler(optimizer, WARMUP_STEPS, MAX_STEPS, LEARNING_RATE, MIN_LR)

    scaler = torch.amp.GradScaler(device_type, enabled=(device_type == "cuda"))
    if device_type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    else:
        ctx = nullcontext()

    print(f"Starting pretraining from step {start_step}/{MAX_STEPS}...")
    model.train()

    for step in range(start_step, MAX_STEPS):

        # ── Evaluation ────────────────────────────────────────────
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
                        loss = nn.functional.cross_entropy(
                            logits[:, :-1].reshape(-1, vocab_size),
                            Y[:, 1:].reshape(-1)
                        )
                    val_loss += loss.item()
                val_loss /= EVAL_ITERS
            print(f"Step {step} | val loss {val_loss:.4f}")
            raw_model = model.module if hasattr(model, "module") else model
            generate_preview(raw_model, tokenizer, device)
            model.train()

        # ── Training step ─────────────────────────────────────────
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
                loss = nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, vocab_size),
                    Y[:, 1:].reshape(-1)
                ) / GRAD_ACCUM

            scaler.scale(loss).backward()
            accum_loss += loss.item() * GRAD_ACCUM

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        lr = scheduler.step(step)

        if step % 10 == 0:
            tokens_seen = step * GRAD_ACCUM * BATCH_SIZE * MAX_LENGTH
            print(f"Step {step} | loss {accum_loss:.4f} | lr {lr:.2e} | tokens {tokens_seen/1e6:.1f}M")

        # ── Checkpoint ────────────────────────────────────────────
        if step > 0 and step % SAVE_INTERVAL == 0:
            save_checkpoint(model, optimizer, step)

    # Final save
    save_checkpoint(model, optimizer, MAX_STEPS - 1)
    print("Pretraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()
    train(resume=args.resume)
