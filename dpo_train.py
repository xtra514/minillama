"""
dpo_train.py  —  Phase 3: On-Policy Heuristic DPO

Generates responses from OUR OWN model, scores with quality heuristics,
trains DPO on the result. This is the correct approach for small models.

Why this works when HH-RLHF doesn't:
  - On-policy: responses come from our model → distribution matches perfectly
  - DPO gets a real gradient signal (not stuck at 0.6931)
  - Proven approach: similar to SPIN / Self-Play Fine-Tuning

Quality heuristics (no API needed):
  1. No repetition  — penalize repeated 4-grams
  2. Length         — prefer fuller answers
  3. Diversity      — unique word ratio
  4. No degeneration — no single-token loops

Pipeline:
  1. Generate response_A (temp=0.7) and response_B (temp=1.1) per prompt
  2. Score both → higher score = chosen, lower = rejected
  3. Skip pairs that are too similar (no useful signal)
  4. Train DPO loss → model learns to prefer better responses

Kaggle usage:
    import minillama.dpo_train as dt
    dt.train(args=Args())

Requires: HF_TOKEN as Kaggle secret.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, random
from contextlib import nullcontext
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from datasets import load_dataset
from minillama.model.transformer import MiniLlama
from minillama.utils import LRScheduler
from minillama.config import CONFIG_100M

# ── Config ────────────────────────────────────────────────────────────────────
_REPO_ROOT     = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(_REPO_ROOT, "data", "tokenizer_32k.json")

NUM_PAIRS      = 2_000     # preference pairs to generate on-policy
GEN_TEMPS      = (0.7, 1.1) # (chosen_temp, rejected_temp) candidates
GEN_MAX_TOKENS = 150        # tokens per response
MIN_SCORE_GAP  = 0.05       # skip pair if scores too similar
MAX_LENGTH     = 512        # DPO training sequence length
BATCH_SIZE     = 2
GRAD_ACCUM     = 8
LEARNING_RATE  = 1e-6       # slightly higher — on-policy needs more signal
BETA           = 0.1
MAX_STEPS      = 1_500
WARMUP_STEPS   = 50
EVAL_INTERVAL  = 200
SAVE_INTERVAL  = 500

SFT_PREFIX = "minillama_125m_sft_step"
DPO_PREFIX = "minillama_125m_dpo_step"

INSTRUCTION_PREFIX = "### Instruction:\n"
RESPONSE_PREFIX    = "\n### Response:\n"

# ── HuggingFace Hub helpers ───────────────────────────────────────────────────

def hf_push(local_path, hf_repo, hf_token):
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.create_repo(hf_repo, exist_ok=True, private=False)
        api.upload_file(path_or_fileobj=local_path,
                        path_in_repo=os.path.basename(local_path),
                        repo_id=hf_repo)
        print(f"  ✓ Uploaded {os.path.basename(local_path)} → hf.co/{hf_repo}")
    except Exception as e:
        print(f"  ⚠ HF upload: {e}")


def hf_pull_latest(hf_repo, hf_token, prefix):
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api   = HfApi(token=hf_token)
        files = list(api.list_repo_files(hf_repo))
        ckpts = sorted(
            [f for f in files if f.startswith(prefix) and f.endswith(".pt")],
            key=lambda f: int(f.replace(prefix + "_", "").replace(".pt", ""))
        )
        if not ckpts:
            return None, None
        target = ckpts[-1]
        step   = int(target.replace(prefix + "_", "").replace(".pt", ""))
        print(f"  Downloading {target}...")
        local  = hf_hub_download(repo_id=hf_repo, filename=target,
                                  token=hf_token)
        return local, step
    except Exception as e:
        print(f"  ⚠ HF pull: {e}")
        return None, None

# ── Quality Heuristics ────────────────────────────────────────────────────────

def quality_score(text: str) -> float:
    """
    Score a response 0-1. Higher = better quality.
    Completely local, no API needed.
    """
    tokens = text.strip().split()
    if len(tokens) < 3:
        return 0.0

    # 1. Length score (prefer 20-80 tokens, penalize very short/long)
    length_score = min(len(tokens) / 40.0, 1.0) * (1.0 if len(tokens) < 100 else 0.7)

    # 2. No repetition — penalize repeated 4-grams
    ngrams = [tuple(tokens[i:i+4]) for i in range(len(tokens) - 3)]
    if ngrams:
        unique_ratio = len(set(ngrams)) / len(ngrams)
    else:
        unique_ratio = 1.0

    # 3. Vocabulary diversity — unique words / total words
    diversity = len(set(tokens)) / max(len(tokens), 1)

    # 4. No degeneration — penalize if a single token repeats consecutively
    degen_penalty = 1.0
    for i in range(len(tokens) - 2):
        if tokens[i] == tokens[i+1] == tokens[i+2]:
            degen_penalty = 0.1
            break

    score = (length_score * 0.25
             + unique_ratio * 0.40
             + diversity   * 0.25
             + degen_penalty * 0.10)

    return score

# ── On-Policy Response Generation ────────────────────────────────────────────

@torch.no_grad()
def generate(raw_model, tokenizer, instruction, device,
             temperature=0.9, max_tokens=GEN_MAX_TOKENS):
    raw_model.eval()
    prompt = INSTRUCTION_PREFIX + instruction + RESPONSE_PREFIX
    ids    = tokenizer.encode(prompt, add_special_tokens=False).ids
    x      = torch.tensor([ids], dtype=torch.long, device=device)
    eos    = tokenizer.token_to_id("</s>") or 2

    out = []
    for i in range(max_tokens):
        logits, _ = raw_model(x[:, -CONFIG_100M.max_position_embeddings:])
        logits     = logits[:, -1, :] / max(temperature, 1e-6)
        logits[:, 0] = float("-inf")
        if i < 5: logits[:, eos] = float("-inf")
        tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
        x   = torch.cat([x, tok], dim=1)
        if tok.item() == eos: break
        out.append(tok.item())

    return tokenizer.decode(out).strip()


def build_preference_pairs(raw_model, tokenizer, device, n=2000):
    """Generate n on-policy preference pairs using heuristic scoring."""
    print(f"\nGenerating {n} on-policy preference pairs (heuristic scoring)...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=42)

    pairs, skipped = [], 0

    for row in ds:
        if len(pairs) >= n:
            break

        instruction = row["instruction"].strip()
        if row.get("input", "").strip():
            instruction += "\n" + row["input"].strip()

        # Generate two responses at different temperatures
        resp_a = generate(raw_model, tokenizer, instruction, device,
                          temperature=GEN_TEMPS[0])
        resp_b = generate(raw_model, tokenizer, instruction, device,
                          temperature=GEN_TEMPS[1])

        if not resp_a or not resp_b:
            skipped += 1
            continue

        score_a = quality_score(resp_a)
        score_b = quality_score(resp_b)

        # Skip if gap too small (not useful signal)
        if abs(score_a - score_b) < MIN_SCORE_GAP:
            skipped += 1
            continue

        if score_a >= score_b:
            chosen, rejected = resp_a, resp_b
        else:
            chosen, rejected = resp_b, resp_a

        pairs.append({"instruction": instruction,
                      "chosen":      chosen,
                      "rejected":    rejected,
                      "score_gap":   abs(score_a - score_b)})

        if len(pairs) % 100 == 0:
            avg_gap = sum(p["score_gap"] for p in pairs) / len(pairs)
            print(f"  [{len(pairs):4d}/{n}] avg_score_gap={avg_gap:.3f} | "
                  f"skipped={skipped} | "
                  f"chosen[:50]={chosen[:50]!r}")

    print(f"✓ Built {len(pairs)} preference pairs "
          f"(skipped {skipped} low-contrast pairs).")
    return pairs

# ── Generation Preview ────────────────────────────────────────────────────────

@torch.no_grad()
def _preview(raw_model, tokenizer, device, temperature=0.8, max_tokens=120):
    raw_model.eval()
    instruction = "Tell me something interesting about space."
    prompt = INSTRUCTION_PREFIX + instruction + RESPONSE_PREFIX
    ids    = tokenizer.encode(prompt, add_special_tokens=False).ids
    x      = torch.tensor([ids], dtype=torch.long, device=device)
    eos    = tokenizer.token_to_id("</s>") or 2

    out = []
    for i in range(max_tokens):
        logits, _ = raw_model(x[:, -CONFIG_100M.max_position_embeddings:])
        logits     = logits[:, -1, :] / temperature
        logits[:, 0] = float("-inf")
        if i < 5: logits[:, eos] = float("-inf")
        tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
        x   = torch.cat([x, tok], dim=1)
        if tok.item() == eos: break
        out.append(tok.item())

    print(f"  PREVIEW: {tokenizer.decode(out).strip()}")

# ── DPO Dataset ───────────────────────────────────────────────────────────────

class DPODataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=512):
        self.data    = pairs
        self.tok     = tokenizer
        self.max_len = max_length
        self.eos     = tokenizer.token_to_id("</s>") or 2
        self.instr_p = tokenizer.encode(INSTRUCTION_PREFIX,
                                         add_special_tokens=False).ids
        self.resp_p  = tokenizer.encode(RESPONSE_PREFIX,
                                         add_special_tokens=False).ids

    def _encode(self, instruction, response):
        instr_ids = self.tok.encode(instruction,
                                     add_special_tokens=False).ids[:200]
        resp_ids  = self.tok.encode(response,
                                     add_special_tokens=False).ids[:250]
        p      = self.instr_p + instr_ids + self.resp_p
        r      = resp_ids + [self.eos]
        full   = (p + r)[:self.max_len]
        labels = ([-100] * len(p) + r)[:self.max_len]
        pad    = self.max_len - len(full)
        return (torch.tensor(full   + [0]    * pad, dtype=torch.long),
                torch.tensor(labels + [-100] * pad, dtype=torch.long))

    def __len__(self):  return len(self.data)

    def __getitem__(self, i):
        row = self.data[i]
        ci, cl = self._encode(row["instruction"], row["chosen"])
        ri, rl = self._encode(row["instruction"], row["rejected"])
        return ci, cl, ri, rl

# ── DPO Loss ──────────────────────────────────────────────────────────────────

def _mean_log_prob(m, ids, labels, vocab_size):
    logits, _    = m(ids)
    shift_logits = logits[:, :-1].contiguous().view(-1, vocab_size)
    shift_labels = labels[:, 1:].contiguous().view(-1)
    mask         = shift_labels != -100
    if mask.sum() == 0:
        return torch.tensor(0.0, device=ids.device, requires_grad=m.training)
    lp  = F.log_softmax(shift_logits, dim=-1)
    tlp = lp.gather(1, shift_labels.clamp(min=0).unsqueeze(1)).squeeze(1)
    return (tlp * mask).sum() / mask.sum()


def dpo_loss(policy, ref, c_ids, c_lab, r_ids, r_lab, beta, vocab_size):
    with torch.no_grad():
        ref_c = _mean_log_prob(ref, c_ids, c_lab, vocab_size)
        ref_r = _mean_log_prob(ref, r_ids, r_lab, vocab_size)

    pol_c = _mean_log_prob(policy, c_ids, c_lab, vocab_size)
    pol_r = _mean_log_prob(policy, r_ids, r_lab, vocab_size)

    loss = -F.logsigmoid(beta * ((pol_c - ref_c) - (pol_r - ref_r)))

    with torch.no_grad():
        acc = ((pol_c - ref_c) > (pol_r - ref_r)).float()

    return loss, acc

# ── Main ──────────────────────────────────────────────────────────────────────

def train(args=None):
    hf_repo  = args.hf_repo if args and hasattr(args, "hf_repo") else None
    hf_token = os.environ.get("HF_TOKEN", None)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    print(f"Device: {device}")

    tokenizer  = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer: {vocab_size} tokens")

    config            = CONFIG_100M
    config.vocab_size = vocab_size

    # ── Load SFT checkpoint ───────────────────────────────────────────────────
    print("\nLoading SFT checkpoint from HF Hub...")
    ckpt_path, _ = hf_pull_latest(hf_repo, hf_token, SFT_PREFIX)
    if not ckpt_path:
        print("ERROR: No SFT checkpoint!"); return

    raw_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state     = raw_state.get("model_state_dict", raw_state)
    state     = {k.replace("module.", ""): v for k, v in state.items()}

    policy = MiniLlama(config)
    policy.load_state_dict(state, strict=False)
    policy.to(device)

    ref = MiniLlama(config)
    ref.load_state_dict(state, strict=False)
    ref.to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    print(f"Policy: {sum(p.numel() for p in policy.parameters()):,} params | Ref: frozen")

    raw = policy   # unwrapped for generation
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        policy = nn.DataParallel(policy)

    # ── Build on-policy preference dataset ───────────────────────────────────
    pairs = build_preference_pairs(raw, tokenizer, device, n=NUM_PAIRS)
    random.shuffle(pairs)

    split    = int(0.95 * len(pairs))
    train_ds = DPODataset(pairs[:split], tokenizer, MAX_LENGTH)
    val_ds   = DPODataset(pairs[split:], tokenizer, MAX_LENGTH)
    print(f"\nDPO dataset — Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  drop_last=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, drop_last=False, num_workers=0)
    train_iter = iter(train_loader)
    val_iter   = iter(val_loader)

    # ── DPO Training ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=LEARNING_RATE,
        weight_decay=0.01, betas=(0.9, 0.95)
    )
    scheduler = LRScheduler(optimizer, WARMUP_STEPS, MAX_STEPS,
                             LEARNING_RATE, LEARNING_RATE * 0.1)

    scaler = torch.amp.GradScaler(device_type, enabled=(device_type == "cuda"))
    dtype  = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
              if device_type == "cuda" else torch.float32)
    ctx = (torch.amp.autocast(device_type="cuda", dtype=dtype)
           if device_type == "cuda" else nullcontext())

    print(f"\nStarting on-policy DPO — {MAX_STEPS} steps | β={BETA} | "
          f"{len(train_ds)} heuristic preference pairs")
    policy.train()

    for step in range(MAX_STEPS):

        # ── Eval + Preview ────────────────────────────────────────────────────
        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            policy.eval()
            vl, va, n = 0.0, 0.0, 0
            with torch.no_grad():
                for _ in range(min(len(val_loader), 30)):
                    try:
                        ci, cl, ri, rl = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        ci, cl, ri, rl = next(val_iter)
                    ci, cl = ci.to(device), cl.to(device)
                    ri, rl = ri.to(device), rl.to(device)
                    l, a   = dpo_loss(policy, ref, ci, cl, ri, rl,
                                      BETA, vocab_size)
                    vl += l.item(); va += a.item(); n += 1
            if n:
                print(f"\nStep {step:5d} | val loss {vl/n:.4f} | "
                      f"pref-accuracy {va/n*100:.1f}%")
            _preview(raw, tokenizer, device)
            print()
            policy.train()

        # ── Train ─────────────────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        acc_l, acc_a = 0.0, 0.0

        for _ in range(GRAD_ACCUM):
            try:
                ci, cl, ri, rl = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                ci, cl, ri, rl = next(train_iter)
            ci, cl = ci.to(device), cl.to(device)
            ri, rl = ri.to(device), rl.to(device)

            with ctx:
                l, a = dpo_loss(policy, ref, ci, cl, ri, rl,
                                BETA, vocab_size)
                l    = l / GRAD_ACCUM

            scaler.scale(l).backward()
            acc_l += l.item() * GRAD_ACCUM
            acc_a += a.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        lr = scheduler.step(step)

        if step % 10 == 0:
            print(f"Step {step:5d} | loss {acc_l/GRAD_ACCUM:.4f} | "
                  f"acc {acc_a/GRAD_ACCUM*100:.1f}% | lr {lr:.2e}")

        if (step > 0 and step % SAVE_INTERVAL == 0) or step == MAX_STEPS - 1:
            r    = policy.module if hasattr(policy, "module") else policy
            name = f"{DPO_PREFIX}_{step}.pt"
            lp   = os.path.join(_REPO_ROOT, name)
            torch.save({"model_state_dict": r.state_dict(), "step": step}, lp)
            print(f"  ✓ Saved: {name}")
            if hf_repo and hf_token:
                hf_push(lp, hf_repo, hf_token)

    print("\n" + "="*60)
    print("ON-POLICY DPO COMPLETE! 🎉")
    print(f"Trained on {len(train_ds)} heuristic preference pairs")
    print("Model now prefers higher-quality, less-repetitive responses!")
    print("="*60)


if __name__ == "__main__":
    train()
