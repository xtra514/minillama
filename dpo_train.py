"""
dpo_train.py
Phase 3: RLAIF + DPO using Mimo as AI judge.

How it works:
  1. Load SFT model from HuggingFace Hub
  2. Generate 2 different responses per instruction (temp 0.7 vs 1.1)
  3. Mimo judges which is better → (instruction, chosen, rejected)
  4. Train with DPO loss → model learns to prefer "good" responses

DPO loss:
  L = -log σ(β * [log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)])

Kaggle usage:
    import minillama.dpo_train as dt
    dt.train(args=Args())

Set MIMO_API_KEY as a Kaggle secret before running.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, json, time, glob, requests
from contextlib import nullcontext
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from datasets import load_dataset
from minillama.model.transformer import MiniLlama
from minillama.utils import LRScheduler
from minillama.config import CONFIG_100M

# ── Config ────────────────────────────────────────────────────────────────────
_REPO_ROOT        = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH    = os.path.join(_REPO_ROOT, "data", "tokenizer_32k.json")
PREF_DATA_PATH    = os.path.join(_REPO_ROOT, "dpo_preferences.json")
PREF_DATA_HF_FILE = "dpo_preferences.json"   # name on HF Hub

NUM_JUDGE_SAMPLES = 200     # preference pairs to generate (fits in ~40 mins)
BATCH_SIZE        = 1       # DPO needs 2× forward passes per item; keep small
GRAD_ACCUM        = 8
MAX_LENGTH        = 256     # short responses only — keeps generation fast
LEARNING_RATE     = 5e-7    # tiny — don't destroy SFT
BETA              = 0.1     # DPO temperature
MAX_STEPS         = 600     # enough for 200 pairs × 3 epochs
WARMUP_STEPS      = 30
EVAL_INTERVAL     = 100
SAVE_INTERVAL     = 200

MIMO_API_URL   = "http://liiwerwp3f3px714nj2yozh0.161.118.160.111.sslip.io/v1/chat/completions"
JUDGE_MODEL    = "mimo-v2.5-free"

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
        print(f"  ⚠ HF upload failed (non-fatal): {e}")


def hf_pull_file(hf_repo, hf_token, filename):
    try:
        from huggingface_hub import hf_hub_download, HfApi
        api   = HfApi(token=hf_token)
        files = list(api.list_repo_files(hf_repo))
        if filename not in files:
            return None
        return hf_hub_download(repo_id=hf_repo, filename=filename, token=hf_token)
    except Exception as e:
        print(f"  ⚠ HF download ({filename}) failed: {e}")
        return None


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
        local  = hf_hub_download(repo_id=hf_repo, filename=target, token=hf_token)
        return local, step
    except Exception as e:
        print(f"  ⚠ HF pull ({prefix}) failed: {e}")
        return None, None

# ── Mimo AI Judge ─────────────────────────────────────────────────────────────

def mimo_judge(instruction, resp_a, resp_b, api_key, retries=3):
    """Ask Mimo to pick the better response. Returns 'A' or 'B'."""
    prompt = (
        "You are a strict AI judge. Given an instruction and two responses, "
        "decide which is better. Reply with ONLY 'CHOSEN: A' or 'CHOSEN: B'.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Response A: {resp_a}\n\n"
        f"Response B: {resp_b}"
    )
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "stream": False,
    }
    for attempt in range(retries):
        try:
            r    = requests.post(MIMO_API_URL, headers=headers,
                                 json=payload, timeout=30)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            if "CHOSEN: A" in text or text.startswith("A"):
                return "A"
            elif "CHOSEN: B" in text or text.startswith("B"):
                return "B"
            else:
                return "A"   # default on ambiguity
        except Exception as e:
            print(f"  ⚠ Judge error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return "A"

# ── Response Generation ───────────────────────────────────────────────────────

@torch.no_grad()
def generate_response(raw_model, tokenizer, instruction, device,
                      temperature=0.9, max_tokens=150):
    """Generate one response from the raw (non-DataParallel) model."""
    raw_model.eval()
    prompt = INSTRUCTION_PREFIX + instruction + RESPONSE_PREFIX
    ids    = tokenizer.encode(prompt, add_special_tokens=False).ids
    x      = torch.tensor([ids], dtype=torch.long, device=device)
    eos    = tokenizer.token_to_id("</s>") or 2

    generated = []
    for i in range(max_tokens):
        logits, _ = raw_model(x[:, -CONFIG_100M.max_position_embeddings:])
        logits     = logits[:, -1, :] / max(temperature, 1e-6)
        logits[:, 0] = float("-inf")
        if i < 5:
            logits[:, eos] = float("-inf")
        tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
        x   = torch.cat([x, tok], dim=1)
        if tok.item() == eos:
            break
        generated.append(tok.item())

    return tokenizer.decode(generated).strip()

# ── Preference Dataset Builder ────────────────────────────────────────────────

def build_preference_dataset(raw_model, tokenizer, device,
                              api_key, hf_repo, hf_token, n=200):
    # Try loading cached from local or HF Hub
    if os.path.exists(PREF_DATA_PATH):
        with open(PREF_DATA_PATH) as f:
            data = json.load(f)
        print(f"Loaded {len(data)} cached preference pairs.")
        if len(data) >= n:
            return data
        print(f"Only {len(data)} pairs cached, need {n}. Continuing...")
    else:
        # Try downloading from HF Hub
        if hf_repo and hf_token:
            local = hf_pull_file(hf_repo, hf_token, PREF_DATA_HF_FILE)
            if local:
                import shutil
                shutil.copy(local, PREF_DATA_PATH)
                with open(PREF_DATA_PATH) as f:
                    data = json.load(f)
                print(f"Downloaded {len(data)} preference pairs from HF Hub.")
                if len(data) >= n:
                    return data
            else:
                data = []
        else:
            data = []

    already = len(data)
    print(f"Generating {n - already} more preference pairs with Mimo judge...")

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=99).select(range(min(n * 3, len(ds))))

    for row in ds:
        if len(data) >= n:
            break

        instruction = row["instruction"].strip()
        if row.get("input", "").strip():
            instruction += "\n" + row["input"].strip()

        # Generate two responses with different temperatures
        resp_a = generate_response(raw_model, tokenizer, instruction,
                                    device, temperature=0.7)
        resp_b = generate_response(raw_model, tokenizer, instruction,
                                    device, temperature=1.1)

        # Skip near-identical responses
        if resp_a.strip() == resp_b.strip() or not resp_a or not resp_b:
            continue

        winner = mimo_judge(instruction, resp_a, resp_b, api_key)
        chosen   = resp_a if winner == "A" else resp_b
        rejected = resp_b if winner == "A" else resp_a

        data.append({"instruction": instruction,
                     "chosen":      chosen,
                     "rejected":    rejected})

        if len(data) % 10 == 0:
            print(f"  [{len(data)}/{n}] Winner={winner} | "
                  f"chosen[:60]={chosen[:60]!r}")
            # Save progress
            with open(PREF_DATA_PATH, "w") as f:
                json.dump(data, f, indent=2)
            if hf_repo and hf_token:
                hf_push(PREF_DATA_PATH, hf_repo, hf_token)

    with open(PREF_DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)
    if hf_repo and hf_token:
        hf_push(PREF_DATA_PATH, hf_repo, hf_token)

    print(f"Preference dataset complete: {len(data)} pairs.")
    return data

# ── DPO Dataset ───────────────────────────────────────────────────────────────

class DPODataset(Dataset):
    def __init__(self, prefs, tokenizer, max_length=256):
        self.data       = prefs
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.eos_id     = tokenizer.token_to_id("</s>") or 2

        self.instr_ids = tokenizer.encode(
            INSTRUCTION_PREFIX, add_special_tokens=False).ids
        self.resp_ids  = tokenizer.encode(
            RESPONSE_PREFIX, add_special_tokens=False).ids

    def _encode(self, instruction, response):
        p_ids = (self.instr_ids
                 + self.tokenizer.encode(instruction,
                                         add_special_tokens=False).ids
                 + self.resp_ids)
        r_ids = (self.tokenizer.encode(response,
                                        add_special_tokens=False).ids
                 + [self.eos_id])
        full   = (p_ids + r_ids)[:self.max_length]
        labels = ([-100] * len(p_ids) + r_ids)[:self.max_length]
        pad    = self.max_length - len(full)
        full   += [0]    * pad
        labels += [-100] * pad
        return (torch.tensor(full,   dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))

    def __len__(self):  return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        c_ids, c_lab = self._encode(row["instruction"], row["chosen"])
        r_ids, r_lab = self._encode(row["instruction"], row["rejected"])
        return c_ids, c_lab, r_ids, r_lab

# ── DPO Loss ──────────────────────────────────────────────────────────────────

def _log_probs(m, input_ids, labels, vocab_size):
    """Compute mean per-token log-prob for labelled tokens."""
    logits, _ = m(input_ids)
    shift_logits = logits[:, :-1].contiguous().view(-1, vocab_size)
    shift_labels = labels[:, 1:].contiguous().view(-1)
    mask         = shift_labels != -100
    if mask.sum() == 0:
        return torch.tensor(0.0, device=input_ids.device,
                            requires_grad=m.training)
    log_p     = F.log_softmax(shift_logits, dim=-1)
    token_lp  = log_p.gather(1, shift_labels.clamp(min=0).unsqueeze(1)).squeeze(1)
    return (token_lp * mask).sum() / mask.sum()


def compute_dpo_loss(policy, ref, c_ids, c_lab, r_ids, r_lab, beta, vocab_size):
    with torch.no_grad():
        ref_c = _log_probs(ref, c_ids, c_lab, vocab_size)
        ref_r = _log_probs(ref, r_ids, r_lab, vocab_size)

    pol_c = _log_probs(policy, c_ids, c_lab, vocab_size)
    pol_r = _log_probs(policy, r_ids, r_lab, vocab_size)

    ratio = beta * ((pol_c - ref_c) - (pol_r - ref_r))
    loss  = -F.logsigmoid(ratio)

    with torch.no_grad():
        acc = ((pol_c - ref_c) > (pol_r - ref_r)).float()

    return loss, acc

# ── Main ──────────────────────────────────────────────────────────────────────

def train(args=None):
    hf_repo  = args.hf_repo  if args and hasattr(args, "hf_repo")  else None
    hf_token = os.environ.get("HF_TOKEN", None)
    api_key  = os.environ.get("MIMO_API_KEY", "")

    if not api_key:
        print("ERROR: MIMO_API_KEY not set!")
        return

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    print(f"Device: {device}")

    # Tokenizer
    tokenizer  = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer: {vocab_size} tokens")

    # Build model config
    config            = CONFIG_100M
    config.vocab_size = vocab_size

    # ── Load SFT checkpoint ──
    print("Loading SFT model from HF Hub...")
    ckpt_path, _ = hf_pull_latest(hf_repo, hf_token, SFT_PREFIX)
    if not ckpt_path:
        print("ERROR: No SFT checkpoint found! Run finetune_chat.py first.")
        return

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state.get("model_state_dict", state)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    # Policy model (trained during DPO)
    policy = MiniLlama(config)
    policy.load_state_dict(state, strict=False)
    policy.to(device)

    # Reference model (frozen copy of SFT)
    ref = MiniLlama(config)
    ref.load_state_dict(state, strict=False)
    ref.to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    print(f"Policy: {sum(p.numel() for p in policy.parameters()):,} params | Ref: frozen")

    # raw = policy before DataParallel (used for generation)
    raw = policy

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        policy = nn.DataParallel(policy)

    # ── Phase 3a: Build preference dataset ──
    prefs = build_preference_dataset(
        raw, tokenizer, device, api_key, hf_repo, hf_token,
        n=NUM_JUDGE_SAMPLES
    )

    split    = int(0.95 * len(prefs))
    train_ds = DPODataset(prefs[:split], tokenizer, MAX_LENGTH)
    val_ds   = DPODataset(prefs[split:], tokenizer, MAX_LENGTH)
    print(f"\nDPO dataset — Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  drop_last=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, drop_last=False, num_workers=0)
    train_iter = iter(train_loader)
    val_iter   = iter(val_loader)

    # ── Phase 3b: DPO Training ──
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

    print(f"\nStarting DPO training for {MAX_STEPS} steps (β={BETA})...")
    policy.train()

    for step in range(MAX_STEPS):

        # Eval
        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            policy.eval()
            vl, va, n = 0.0, 0.0, 0
            with torch.no_grad():
                for _ in range(min(len(val_loader), 20)):
                    try:
                        c_ids, c_lab, r_ids, r_lab = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        c_ids, c_lab, r_ids, r_lab = next(val_iter)
                    c_ids, c_lab = c_ids.to(device), c_lab.to(device)
                    r_ids, r_lab = r_ids.to(device), r_lab.to(device)
                    loss, acc = compute_dpo_loss(
                        policy, ref, c_ids, c_lab, r_ids, r_lab,
                        BETA, vocab_size
                    )
                    vl += loss.item(); va += acc.item(); n += 1
            if n > 0:
                print(f"Step {step:4d} | val loss {vl/n:.4f} | "
                      f"preference acc {va/n*100:.1f}%")
            policy.train()

        # Train
        optimizer.zero_grad(set_to_none=True)
        acc_loss, acc_acc = 0.0, 0.0

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
                    policy, ref, c_ids, c_lab, r_ids, r_lab,
                    BETA, vocab_size
                )
                loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()
            acc_loss += loss.item() * GRAD_ACCUM
            acc_acc  += acc.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        lr = scheduler.step(step)

        if step % 10 == 0:
            print(f"Step {step:4d} | loss {acc_loss:.4f} | "
                  f"acc {acc_acc/GRAD_ACCUM*100:.1f}% | lr {lr:.2e}")

        if (step > 0 and step % SAVE_INTERVAL == 0) or step == MAX_STEPS - 1:
            r = policy.module if hasattr(policy, "module") else policy
            name = f"{DPO_PREFIX}_{step}.pt"
            local = os.path.join(_REPO_ROOT, name)
            torch.save({"model_state_dict": r.state_dict(), "step": step},
                       local)
            print(f"  ✓ Saved: {name}")
            if hf_repo and hf_token:
                hf_push(local, hf_repo, hf_token)

    print("\nDPO training complete! 🎉")
    print("Your model is now aligned by AI. Ready for deployment!")


if __name__ == "__main__":
    train()
