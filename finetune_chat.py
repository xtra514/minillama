"""
finetune_chat.py
Phase 2: SFT fine-tune MiniLlama-125M on Alpaca.

Loads the best pretrained checkpoint from HuggingFace Hub,
fine-tunes on Alpaca instruction data, and pushes SFT checkpoints
back to the Hub for resuming across sessions.

Kaggle usage:
    import minillama.finetune_chat as fc
    fc.train(args=Args())
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

# ── Paths & Hyperparameters ──────────────────────────────────────────────────
_REPO_ROOT      = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH  = os.path.join(_REPO_ROOT, "data", "tokenizer_32k.json")

BATCH_SIZE      = 2       # safe for T4 with 1024 ctx + DataParallel
GRAD_ACCUM      = 16      # effective batch = 2×16 = 32
MAX_LENGTH      = 512     # SFT sequences are short; saves VRAM
LEARNING_RATE   = 2e-5
MAX_STEPS       = 3_000
WARMUP_STEPS    = 150
WEIGHT_DECAY    = 0.05
GRAD_CLIP       = 1.0
EVAL_INTERVAL   = 200
EVAL_ITERS      = 30
SAVE_INTERVAL   = 500

PRETRAIN_PREFIX = "minillama_125m_step"
SFT_PREFIX      = "minillama_125m_sft_step"

INSTRUCTION_PREFIX = "### Instruction:\n"
RESPONSE_PREFIX    = "\n### Response:\n"

# ── HuggingFace Hub helpers ──────────────────────────────────────────────────

def hf_push(local_path, hf_repo, hf_token):
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.create_repo(hf_repo, exist_ok=True, private=False)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=os.path.basename(local_path),
            repo_id=hf_repo,
        )
        print(f"  ✓ Uploaded {os.path.basename(local_path)} → hf.co/{hf_repo}")
    except Exception as e:
        print(f"  ⚠ HF upload failed (non-fatal): {e}")


def hf_pull(hf_repo, hf_token, prefix, fallback_prefix=None):
    """Download the latest checkpoint matching prefix from HF Hub."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api   = HfApi(token=hf_token)
        files = list(api.list_repo_files(hf_repo))

        def best(pfx):
            ckpts = sorted(
                [f for f in files if f.startswith(pfx) and f.endswith(".pt")],
                key=lambda f: int(f.replace(pfx + "_", "").replace(".pt", ""))
            )
            return ckpts[-1] if ckpts else None

        target = best(prefix)
        if target is None and fallback_prefix:
            target = best(fallback_prefix)
        if target is None:
            print(f"  No checkpoint found on HF Hub (prefix='{prefix}').")
            return None, None

        step_str = target.replace(prefix + "_", "").replace(".pt", "")
        try:
            step = int(step_str)
        except ValueError:
            step = 0

        print(f"  Downloading {target} from hf.co/{hf_repo}...")
        local = hf_hub_download(repo_id=hf_repo, filename=target, token=hf_token)
        print(f"  Downloaded to {local}")
        return local, step
    except Exception as e:
        print(f"  ⚠ HF download failed: {e}")
        return None, None

# ── Dataset ──────────────────────────────────────────────────────────────────

class AlpacaDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=512):
        self.data       = hf_dataset
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.eos_id     = tokenizer.token_to_id("</s>") or 2

        self.instr_ids = tokenizer.encode(
            INSTRUCTION_PREFIX, add_special_tokens=False).ids
        self.resp_ids  = tokenizer.encode(
            RESPONSE_PREFIX, add_special_tokens=False).ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        instruction = row["instruction"].strip()
        if row.get("input", "").strip():
            instruction += "\n" + row["input"].strip()
        response = row["output"].strip()

        prompt_ids   = (self.instr_ids
                        + self.tokenizer.encode(instruction, add_special_tokens=False).ids
                        + self.resp_ids)
        response_ids = (self.tokenizer.encode(response, add_special_tokens=False).ids
                        + [self.eos_id])

        input_ids = prompt_ids + response_ids
        # Only train on response tokens (mask the prompt with -100)
        targets   = [-100] * len(prompt_ids) + response_ids

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
def generate_sample(model, tokenizer, device,
                    instruction="Tell me something interesting about space."):
    model.eval()
    prompt = INSTRUCTION_PREFIX + instruction + RESPONSE_PREFIX
    ids    = tokenizer.encode(prompt, add_special_tokens=False).ids
    x      = torch.tensor([ids], dtype=torch.long, device=device)
    eos    = tokenizer.token_to_id("</s>") or 2

    generated = []
    for i in range(300):
        logits, _ = model(x)
        logits = logits[:, -1, :] / 0.7        # temperature
        logits[:, 0] = float("-inf")            # never predict pad
        if i < 5:
            logits[:, eos] = float("-inf")      # no early EOS
        probs    = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1)
        x = torch.cat([x, next_tok], dim=1)
        if next_tok.item() == eos:
            break
        generated.append(next_tok.item())

    response = tokenizer.decode(generated).strip()
    print(f"\n{'='*60}")
    print(f"INSTRUCTION: {instruction}")
    print(f"RESPONSE:    {response}")
    print(f"{'='*60}\n")
    model.train()

# ── Main ─────────────────────────────────────────────────────────────────────

def train(args=None):
    hf_repo  = args.hf_repo  if args and hasattr(args, "hf_repo")  else None
    resume   = args.resume   if args and hasattr(args, "resume")   else False
    hf_token = os.environ.get("HF_TOKEN", None)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    print(f"Device: {device}")

    # Tokenizer
    tokenizer  = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer: {vocab_size} tokens")

    # Build model
    config            = CONFIG_100M
    config.vocab_size = vocab_size
    model = MiniLlama(config)

    start_step = 0

    if resume and hf_repo and hf_token:
        # Try to resume from an existing SFT checkpoint first
        print("Checking HF Hub for SFT checkpoint to resume...")
        ckpt_path, sft_step = hf_pull(hf_repo, hf_token,
                                       prefix=SFT_PREFIX)
        if ckpt_path:
            print(f"Resuming SFT from step {sft_step}...")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
            start_step = sft_step + 1
        else:
            print("No SFT checkpoint found — loading pretrained base...")
            ckpt_path, _ = hf_pull(hf_repo, hf_token,
                                    prefix=PRETRAIN_PREFIX)
            if ckpt_path:
                ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                state = ckpt.get("model_state_dict", ckpt)
                state = {k.replace("module.", ""): v for k, v in state.items()}
                missing, unexpected = model.load_state_dict(state, strict=False)
                print(f"  Loaded pretrained weights (missing={len(missing)}, "
                      f"unexpected={len(unexpected)})")
            else:
                print("  ERROR: No pretrained checkpoint found! Run pretrain first.")
                return
    else:
        # Local fallback
        ckpts = sorted(
            glob.glob(os.path.join(_REPO_ROOT, f"{PRETRAIN_PREFIX}_*.pt")),
            key=lambda p: int(p.split("_")[-1].replace(".pt", ""))
        )
        if not ckpts:
            print("ERROR: No local pretrained checkpoint found.")
            return
        print(f"Loading local pretrained checkpoint: {ckpts[-1]}")
        ckpt  = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)

    # Optimizer BEFORE moving to device (for fresh optimizer state)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95)
    )

    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    scheduler = LRScheduler(optimizer, WARMUP_STEPS, MAX_STEPS,
                             LEARNING_RATE, LEARNING_RATE * 0.1)

    scaler = torch.amp.GradScaler(device_type, enabled=(device_type == "cuda"))
    dtype  = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
              if device_type == "cuda" else torch.float32)
    ctx = (torch.amp.autocast(device_type="cuda", dtype=dtype)
           if device_type == "cuda" else nullcontext())

    # Alpaca dataset
    print("Loading Alpaca dataset...")
    ds    = load_dataset("tatsu-lab/alpaca", split="train")
    split = ds.train_test_split(test_size=0.02, seed=42)
    train_ds = AlpacaDataset(split["train"], tokenizer, MAX_LENGTH)
    val_ds   = AlpacaDataset(split["test"],  tokenizer, MAX_LENGTH)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              drop_last=True, num_workers=0)
    train_iter = iter(train_loader)
    val_iter   = iter(val_loader)

    print(f"Starting SFT from step {start_step}/{MAX_STEPS}...")
    model.train()

    for step in range(start_step, MAX_STEPS):

        # ── Eval ──
        if step % EVAL_INTERVAL == 0:
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
                            logits[:, :-1].contiguous().view(-1, vocab_size),
                            Y[:, 1:].contiguous().view(-1),
                            ignore_index=-100
                        )
                    val_loss += loss.item()
            val_loss /= EVAL_ITERS
            print(f"Step {step} | val loss {val_loss:.4f}")
            raw = model.module if hasattr(model, "module") else model
            generate_sample(raw, tokenizer, device)
            model.train()

        # ── Train ──
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
                    logits[:, :-1].contiguous().view(-1, vocab_size),
                    Y[:, 1:].contiguous().view(-1),
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

        # ── Save & Push ──
        if (step > 0 and step % SAVE_INTERVAL == 0) or step == MAX_STEPS - 1:
            raw = model.module if hasattr(model, "module") else model
            ckpt_name = f"{SFT_PREFIX}_{step}.pt"
            local_path = os.path.join(_REPO_ROOT, ckpt_name)
            torch.save({"model_state_dict": raw.state_dict(), "step": step},
                       local_path)
            print(f"  ✓ Saved: {ckpt_name}")
            if hf_repo and hf_token:
                hf_push(local_path, hf_repo, hf_token)

    print("SFT complete!")


if __name__ == "__main__":
    train()
