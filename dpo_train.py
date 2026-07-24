"""
dpo_train.py
Phase 3: RLAIF + DPO (Direct Preference Optimization) using Mimo as AI judge.

How it works:
  1. Load fine-tuned chat model
  2. Generate 2 different responses per instruction
  3. Mimo AI (llama-3.3-70b) judges which is better
  4. Build preference dataset: {instruction, chosen, rejected}
  5. Train with DPO loss — model learns to prefer "good" responses

Kaggle usage:
    python -m minillama.dpo_train
    
Set MIMO_API_KEY as a Kaggle secret before running.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import time
import glob
import requests
from contextlib import nullcontext
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from datasets import load_dataset
from minillama.model.transformer import MiniLlama
from minillama.utils import LRScheduler
from minillama.config import CONFIG_100M

# ── Config ───────────────────────────────────────────────────────────────────
BATCH_SIZE          = 4
GRAD_ACCUM          = 4
MAX_LENGTH          = 512
LEARNING_RATE       = 5e-7       # very small for DPO — don't destroy fine-tuning
BETA                = 0.1        # DPO temperature — controls how strongly to prefer chosen
MAX_STEPS           = 1_000
WARMUP_STEPS        = 50
EVAL_INTERVAL       = 100
TOKENIZER_PATH      = "minillama/data/tokenizer_32k.json"
CHAT_CKPT_GLOB      = "minillama_125m_chat_step_*.pt"
PREF_DATA_PATH      = "dpo_preferences.json"   # cached preference pairs
NUM_JUDGE_SAMPLES   = 500        # how many preference pairs to generate

MIMO_API_URL        = "http://liiwerwp3f3px714nj2yozh0.161.118.160.111.sslip.io/v1/chat/completions"
JUDGE_MODEL         = "llama-3.3-70b-instruct:free"

INSTRUCTION_PREFIX  = "### Instruction:\n"
RESPONSE_PREFIX     = "\n### Response:\n"

# ── Mimo AI Judge ─────────────────────────────────────────────────────────────

def mimo_judge(instruction, response_a, response_b, api_key, retries=3):
    """
    Ask Mimo's Llama-3.3-70B to pick the better response.
    Returns "A" or "B".
    """
    prompt = (
        f"Instruction: {instruction}\n\n"
        f"Response A: {response_a}\n\n"
        f"Response B: {response_b}\n\n"
        "Which response better follows the instruction and is more helpful, "
        "accurate, and natural? Reply with exactly: CHOSEN: A or CHOSEN: B, "
        "then one sentence explaining why."
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert AI judge evaluating chatbot response quality. Be objective and concise."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "stream": False,
    }

    for attempt in range(retries):
        try:
            r = requests.post(MIMO_API_URL, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()

            if "CHOSEN: A" in text:
                return "A", text
            elif "CHOSEN: B" in text:
                return "B", text
            else:
                print(f"  ⚠ Ambiguous judge response: {text[:100]}")
                return "A", text  # default to A on ambiguity
        except Exception as e:
            print(f"  ⚠ Judge API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)

    return "A", "API failed"


# ── Response Generation ───────────────────────────────────────────────────────

@torch.no_grad()
def generate_response(model, tokenizer, instruction, device,
                      temperature=0.9, max_tokens=200):
    """Generate a single response from the model."""
    prompt = INSTRUCTION_PREFIX + instruction + RESPONSE_PREFIX
    ids    = tokenizer.encode(prompt, add_special_tokens=False).ids
    x      = torch.tensor([ids], dtype=torch.long, device=device)
    eos    = tokenizer.token_to_id("</s>") or 2

    generated = []
    for i in range(max_tokens):
        logits, _ = model(x[:, -CONFIG_100M.max_position_embeddings:])
        logits = logits[:, -1, :] / temperature
        if i < 5:
            logits[:, eos] = float("-inf")
        next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
        x = torch.cat([x, next_tok], dim=1)
        generated.append(next_tok.item())
        if next_tok.item() == eos:
            break

    return tokenizer.decode(generated).strip()


# ── Build Preference Dataset ──────────────────────────────────────────────────

def build_preference_dataset(model, tokenizer, device, api_key, n=500):
    """
    Generate response pairs, judge with Mimo, save preference dataset.
    Skips if dpo_preferences.json already exists (resumable).
    """
    if os.path.exists(PREF_DATA_PATH):
        print(f"Loading cached preferences from {PREF_DATA_PATH}...")
        with open(PREF_DATA_PATH) as f:
            data = json.load(f)
        print(f"Loaded {len(data)} preference pairs.")
        return data

    print(f"Building {n} preference pairs with Mimo as judge...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    # Shuffle and take a subset
    ds = ds.shuffle(seed=42).select(range(min(n * 2, len(ds))))

    preferences = []
    model.eval()

    for i, row in enumerate(ds):
        if len(preferences) >= n:
            break

        instruction = row["instruction"].strip()
        if row.get("input", "").strip():
            instruction += "\n" + row["input"].strip()

        # Generate two different responses (different temperatures)
        resp_a = generate_response(model, tokenizer, instruction, device, temperature=0.7)
        resp_b = generate_response(model, tokenizer, instruction, device, temperature=1.1)

        # Skip if identical
        if resp_a.strip() == resp_b.strip():
            continue

        # Judge
        winner, reason = mimo_judge(instruction, resp_a, resp_b, api_key)

        chosen   = resp_a if winner == "A" else resp_b
        rejected = resp_b if winner == "A" else resp_a

        preferences.append({
            "instruction": instruction,
            "chosen":      chosen,
            "rejected":    rejected,
            "reason":      reason,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{len(preferences)}/{n}] Last winner: {winner} — {reason[:60]}")
            # Save progress periodically
            with open(PREF_DATA_PATH, "w") as f:
                json.dump(preferences, f, indent=2)

    # Final save
    with open(PREF_DATA_PATH, "w") as f:
        json.dump(preferences, f, indent=2)
    print(f"Saved {len(preferences)} preference pairs to {PREF_DATA_PATH}")
    return preferences


# ── DPO Dataset ───────────────────────────────────────────────────────────────

class DPODataset(Dataset):
    def __init__(self, preferences, tokenizer, max_length=512):
        self.data      = preferences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_id    = tokenizer.token_to_id("</s>") or 2

    def encode_pair(self, instruction, response):
        prompt   = INSTRUCTION_PREFIX + instruction + RESPONSE_PREFIX
        p_ids    = self.tokenizer.encode(prompt,   add_special_tokens=False).ids
        r_ids    = self.tokenizer.encode(response, add_special_tokens=False).ids + [self.eos_id]
        full     = p_ids + r_ids
        labels   = [-100] * len(p_ids) + r_ids
        # Truncate
        full   = full[:self.max_length]
        labels = labels[:self.max_length]
        # Pad
        pad = self.max_length - len(full)
        full   += [0]    * pad
        labels += [-100] * pad
        return torch.tensor(full, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        chosen_ids,   chosen_labels   = self.encode_pair(row["instruction"], row["chosen"])
        rejected_ids, rejected_labels = self.encode_pair(row["instruction"], row["rejected"])
        return chosen_ids, chosen_labels, rejected_ids, rejected_labels


# ── DPO Loss ──────────────────────────────────────────────────────────────────

def compute_dpo_loss(model, ref_model, chosen_ids, chosen_labels,
                     rejected_ids, rejected_labels, beta=0.1):
    """
    DPO loss: train model to assign higher probability to chosen vs rejected.

    L_DPO = -log sigmoid(beta * (log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)))
    """
    vocab_size = model.module.config.vocab_size if hasattr(model, "module") else model.config.vocab_size

    def log_probs(m, input_ids, labels):
        logits, _ = m(input_ids)
        # Shift: predict token i+1 from token i
        shift_logits = logits[:, :-1].contiguous().view(-1, vocab_size)
        shift_labels = labels[:, 1:].contiguous().view(-1)
        mask = (shift_labels != -100)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=input_ids.device)
        log_p = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_p.gather(1, shift_labels.clamp(min=0).unsqueeze(1)).squeeze(1)
        return (token_log_probs * mask).sum() / mask.sum()

    with torch.no_grad():
        ref_chosen_logp   = log_probs(ref_model, chosen_ids,   chosen_labels)
        ref_rejected_logp = log_probs(ref_model, rejected_ids, rejected_labels)

    chosen_logp   = log_probs(model, chosen_ids,   chosen_labels)
    rejected_logp = log_probs(model, rejected_ids, rejected_labels)

    chosen_ratio   = chosen_logp   - ref_chosen_logp
    rejected_ratio = rejected_logp - ref_rejected_logp

    loss = -F.logsigmoid(beta * (chosen_ratio - rejected_ratio)).mean()

    # Accuracy: how often does model prefer chosen over rejected?
    with torch.no_grad():
        accuracy = (chosen_ratio > rejected_ratio).float().mean()

    return loss, accuracy


# ── Main Training Loop ────────────────────────────────────────────────────────

def train():
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    print(f"Device: {device}")

    api_key = os.environ.get("MIMO_API_KEY", "")
    if not api_key:
        print("ERROR: MIMO_API_KEY not set! Add it as a Kaggle secret.")
        return

    # Load tokenizer
    tokenizer  = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()

    # Load fine-tuned model (policy model)
    ckpts = sorted(glob.glob(CHAT_CKPT_GLOB),
                   key=lambda p: int(p.split("_")[-1].replace(".pt", "")))
    if not ckpts:
        print(f"ERROR: No chat checkpoint found matching {CHAT_CKPT_GLOB}")
        return
    ckpt_path = ckpts[-1]
    print(f"Loading policy model from {ckpt_path}...")

    config = CONFIG_100M
    config.vocab_size = vocab_size

    # Policy model (gets trained)
    model = MiniLlama(config)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device)

    # Reference model (frozen — the original fine-tuned model before DPO)
    ref_model = MiniLlama(config)
    ref_model.load_state_dict(state, strict=False)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    print(f"Policy model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Reference model: frozen (no grad)")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # ── Step 1: Build preference dataset with Mimo judge ──────────────────
    preferences = build_preference_dataset(
        model, tokenizer, device, api_key, n=NUM_JUDGE_SAMPLES
    )

    split     = int(0.95 * len(preferences))
    train_prefs = preferences[:split]
    val_prefs   = preferences[split:]

    train_ds = DPODataset(train_prefs, tokenizer, MAX_LENGTH)
    val_ds   = DPODataset(val_prefs,   tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    train_iter = iter(train_loader)
    val_iter   = iter(val_loader)

    print(f"\nPreference pairs — Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Step 2: DPO Training ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.01, betas=(0.9, 0.95)
    )
    scheduler = LRScheduler(optimizer, WARMUP_STEPS, MAX_STEPS,
                            LEARNING_RATE, LEARNING_RATE * 0.1)

    scaler = torch.amp.GradScaler(device_type, enabled=(device_type == "cuda"))
    if device_type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    else:
        ctx = nullcontext()

    print(f"\nStarting DPO training for {MAX_STEPS} steps (beta={BETA})...")
    model.train()

    for step in range(MAX_STEPS):

        # Eval
        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            model.eval()
            val_loss_total, val_acc_total = 0.0, 0.0
            n_val = min(len(val_loader), 20)
            with torch.no_grad():
                for _ in range(n_val):
                    try:
                        c_ids, c_lab, r_ids, r_lab = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        c_ids, c_lab, r_ids, r_lab = next(val_iter)
                    c_ids, c_lab = c_ids.to(device), c_lab.to(device)
                    r_ids, r_lab = r_ids.to(device), r_lab.to(device)
                    loss, acc = compute_dpo_loss(
                        model, ref_model, c_ids, c_lab, r_ids, r_lab, BETA
                    )
                    val_loss_total += loss.item()
                    val_acc_total  += acc.item()
            print(f"Step {step:4d} | val loss {val_loss_total/n_val:.4f} | "
                  f"judge accuracy {val_acc_total/n_val*100:.1f}%")
            model.train()

        # Train
        optimizer.zero_grad(set_to_none=True)
        accum_loss, accum_acc = 0.0, 0.0

        for _ in range(GRAD_ACCUM):
            try:
                c_ids, c_lab, r_ids, r_lab = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                c_ids, c_lab, r_ids, r_lab = next(train_iter)
            c_ids, c_lab = c_ids.to(device), c_lab.to(device)
            r_ids, r_lab = r_ids.to(device), r_lab.to(device)

            with ctx:
                loss, acc = compute_dpo_loss(
                    model, ref_model, c_ids, c_lab, r_ids, r_lab, BETA
                )
                loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()
            accum_loss += loss.item() * GRAD_ACCUM
            accum_acc  += acc.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        lr = scheduler.step(step)

        if step % 10 == 0:
            print(f"Step {step:4d} | loss {accum_loss:.4f} | "
                  f"acc {accum_acc/GRAD_ACCUM*100:.1f}% | lr {lr:.2e}")

        if (step > 0 and step % 200 == 0) or step == MAX_STEPS - 1:
            raw = model.module if hasattr(model, "module") else model
            out = f"minillama_125m_dpo_step_{step}.pt"
            torch.save(raw.state_dict(), out)
            print(f"Saved: {out}")

    print("\nDPO training complete!")
    print("Your model has been supervised by AI and knows what 'good' means. 🎉")


if __name__ == "__main__":
    train()
