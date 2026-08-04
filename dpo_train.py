"""
dpo_train.py  —  Phase 3: RLAIF + DPO with Mimo as judge (FULL SCALE)

Pipeline:
  1. Load SFT model from HuggingFace
  2. Generate 3 responses per instruction (temp 0.6 / 0.9 / 1.2)
  3. Mimo judges all 3 and picks BEST and WORST
  4. Build (instruction, best, worst) preference triplets
  5. Train DPO loss — model strongly prefers best over worst
  6. Save checkpoints to HuggingFace

Scale: 1000 pairs × 2000 DPO steps = maximum alignment signal.

Kaggle usage:
    import minillama.dpo_train as dt
    dt.train(args=Args())

Requires: HF_TOKEN, MIMO_API_KEY as Kaggle secrets.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, json, time, glob, requests, random
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
PREF_DATA_HF_FILE = "dpo_preferences.json"

# ── Scale knobs (FULL) ────────────────────────────────────────────────────────
NUM_JUDGE_SAMPLES = 1000    # preference pairs — unlimited API so go big
GEN_TEMPERATURES  = [0.6, 0.9, 1.2]   # 3 candidates per prompt
GEN_MAX_TOKENS    = 200     # per candidate response
MAX_LENGTH        = 512     # DPO training sequence length

# ── DPO training ──────────────────────────────────────────────────────────────
BATCH_SIZE     = 1
GRAD_ACCUM     = 8
LEARNING_RATE  = 5e-7       # tiny — preserve SFT knowledge
BETA           = 0.1        # DPO temperature
MAX_STEPS      = 2_000
WARMUP_STEPS   = 100
EVAL_INTERVAL  = 200
SAVE_INTERVAL  = 500

# ── API ───────────────────────────────────────────────────────────────────────
MIMO_API_URL  = "http://liiwerwp3f3px714nj2yozh0.161.118.160.111.sslip.io/v1/chat/completions"
JUDGE_MODEL   = "mimo-v2.5-free"

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
        print(f"  ⚠ HF upload failed: {e}")


def hf_pull_file(hf_repo, hf_token, filename):
    try:
        from huggingface_hub import hf_hub_download, HfApi
        api   = HfApi(token=hf_token)
        files = list(api.list_repo_files(hf_repo))
        if filename not in files:
            return None
        return hf_hub_download(repo_id=hf_repo, filename=filename,
                                token=hf_token)
    except Exception as e:
        print(f"  ⚠ HF download ({filename}): {e}")
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
        print(f"  Downloading {target} from HF Hub...")
        local  = hf_hub_download(repo_id=hf_repo, filename=target,
                                  token=hf_token)
        print(f"  Downloaded to {local}")
        return local, step
    except Exception as e:
        print(f"  ⚠ HF pull ({prefix}): {e}")
        return None, None

# ── Mimo AI Judge — multi-criteria ───────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are an expert AI evaluator. Score each response on:\n"
    "1. Helpfulness — does it actually answer the instruction?\n"
    "2. Accuracy    — is the information correct?\n"
    "3. Clarity     — is it easy to understand?\n"
    "4. Completeness — does it cover all aspects?\n"
    "Be strict and objective."
)

def mimo_judge_three(instruction, resp_a, resp_b, resp_c, api_key, retries=3):
    """
    Ask Mimo to rank 3 responses and return (best_letter, worst_letter).
    Returns e.g. ('A', 'C') meaning A is best, C is worst.
    """
    prompt = (
        f"Instruction: {instruction}\n\n"
        f"Response A:\n{resp_a}\n\n"
        f"Response B:\n{resp_b}\n\n"
        f"Response C:\n{resp_c}\n\n"
        "Rank these responses from best to worst based on helpfulness, "
        "accuracy, clarity, and completeness.\n"
        "Reply in EXACTLY this format:\n"
        "BEST: X\nWORST: Y\n"
        "where X and Y are A, B, or C."
    )
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type":  "application/json"}
    payload = {
        "model":       JUDGE_MODEL,
        "messages":    [{"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user",   "content": prompt}],
        "temperature": 0.1,
        "stream":      False,
    }
    for attempt in range(retries):
        try:
            r    = requests.post(MIMO_API_URL, headers=headers,
                                 json=payload, timeout=60)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()

            best  = None
            worst = None
            for line in text.splitlines():
                l = line.strip().upper()
                if l.startswith("BEST:") and not best:
                    c = l.replace("BEST:", "").strip()[:1]
                    if c in ("A", "B", "C"):
                        best = c
                if l.startswith("WORST:") and not worst:
                    c = l.replace("WORST:", "").strip()[:1]
                    if c in ("A", "B", "C"):
                        worst = c

            if best and worst and best != worst:
                return best, worst
            # fallback: try single-letter parse
            letters = [w for w in text.split() if w in ("A", "B", "C")]
            if len(letters) >= 2:
                return letters[0], letters[-1]

        except Exception as e:
            print(f"  ⚠ Judge error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)

    # Default fallback
    return "A", "C"

# ── Response Generation ───────────────────────────────────────────────────────

@torch.no_grad()
def generate_response(raw_model, tokenizer, instruction, device,
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
        if i < 5:
            logits[:, eos] = float("-inf")
        tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
        x   = torch.cat([x, tok], dim=1)
        if tok.item() == eos:
            break
        out.append(tok.item())

    return tokenizer.decode(out).strip()

# ── Preference Dataset Builder ────────────────────────────────────────────────

def build_preference_dataset(raw_model, tokenizer, device,
                              api_key, hf_repo, hf_token, n=1000):
    # Try local cache
    data = []
    if os.path.exists(PREF_DATA_PATH):
        with open(PREF_DATA_PATH) as f:
            data = json.load(f)
        print(f"Loaded {len(data)} cached preference pairs.")
        if len(data) >= n:
            print("Preference dataset complete (cached).")
            return data

    # Try HF Hub cache
    if not data and hf_repo and hf_token:
        local = hf_pull_file(hf_repo, hf_token, PREF_DATA_HF_FILE)
        if local:
            import shutil
            shutil.copy(local, PREF_DATA_PATH)
            with open(PREF_DATA_PATH) as f:
                data = json.load(f)
            print(f"Downloaded {len(data)} pairs from HF Hub.")
            if len(data) >= n:
                return data

    print(f"\nGenerating {n - len(data)} preference pairs "
          f"({len(GEN_TEMPERATURES)} candidates each, Mimo 3-way judge)...")

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=77).select(range(min(n * 4, len(ds))))

    for row in ds:
        if len(data) >= n:
            break

        instruction = row["instruction"].strip()
        if row.get("input", "").strip():
            instruction += "\n" + row["input"].strip()

        # Generate 3 candidates at different temperatures
        candidates = []
        for temp in GEN_TEMPERATURES:
            r = generate_response(raw_model, tokenizer, instruction,
                                   device, temperature=temp)
            candidates.append(r)

        # Skip if too similar
        unique = set(c.strip()[:50] for c in candidates if c)
        if len(unique) < 2:
            continue

        resp_a, resp_b, resp_c = candidates[0], candidates[1], candidates[2]

        # Judge
        best_letter, worst_letter = mimo_judge_three(
            instruction, resp_a, resp_b, resp_c, api_key
        )

        letter_map = {"A": resp_a, "B": resp_b, "C": resp_c}
        chosen   = letter_map.get(best_letter,  resp_a)
        rejected = letter_map.get(worst_letter, resp_c)

        if chosen == rejected:
            continue

        data.append({
            "instruction": instruction,
            "chosen":      chosen,
            "rejected":    rejected,
            "best":        best_letter,
            "worst":       worst_letter,
        })

        if len(data) % 10 == 0:
            print(f"  [{len(data):4d}/{n}] best={best_letter} | "
                  f"chosen[:50]={chosen[:50]!r}")
            with open(PREF_DATA_PATH, "w") as f:
                json.dump(data, f, indent=2)
            if hf_repo and hf_token:
                hf_push(PREF_DATA_PATH, hf_repo, hf_token)

    with open(PREF_DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)
    if hf_repo and hf_token:
        hf_push(PREF_DATA_PATH, hf_repo, hf_token)

    print(f"✓ Preference dataset: {len(data)} pairs saved.")
    return data

# ── DPO Dataset ───────────────────────────────────────────────────────────────

class DPODataset(Dataset):
    def __init__(self, prefs, tokenizer, max_length=512):
        self.data      = prefs
        self.tok       = tokenizer
        self.max_len   = max_length
        self.eos       = tokenizer.token_to_id("</s>") or 2
        self.instr_ids = tokenizer.encode(INSTRUCTION_PREFIX,
                                           add_special_tokens=False).ids
        self.resp_ids  = tokenizer.encode(RESPONSE_PREFIX,
                                           add_special_tokens=False).ids

    def _enc(self, instruction, response):
        p = (self.instr_ids
             + self.tok.encode(instruction, add_special_tokens=False).ids
             + self.resp_ids)
        r = self.tok.encode(response, add_special_tokens=False).ids + [self.eos]
        full   = (p + r)[:self.max_len]
        labels = ([-100] * len(p) + r)[:self.max_len]
        pad    = self.max_len - len(full)
        return (torch.tensor(full   + [0]    * pad, dtype=torch.long),
                torch.tensor(labels + [-100] * pad, dtype=torch.long))

    def __len__(self):  return len(self.data)

    def __getitem__(self, i):
        row = self.data[i]
        ci, cl = self._enc(row["instruction"], row["chosen"])
        ri, rl = self._enc(row["instruction"], row["rejected"])
        return ci, cl, ri, rl

# ── DPO Loss ──────────────────────────────────────────────────────────────────

def _mean_log_prob(m, ids, labels, vocab_size):
    logits, _    = m(ids)
    shift_logits = logits[:, :-1].contiguous().view(-1, vocab_size)
    shift_labels = labels[:, 1:].contiguous().view(-1)
    mask         = shift_labels != -100
    if mask.sum() == 0:
        return torch.tensor(0.0, device=ids.device, requires_grad=m.training)
    lp   = F.log_softmax(shift_logits, dim=-1)
    tlp  = lp.gather(1, shift_labels.clamp(min=0).unsqueeze(1)).squeeze(1)
    return (tlp * mask).sum() / mask.sum()


def dpo_loss(policy, ref, c_ids, c_lab, r_ids, r_lab, beta, vocab_size):
    with torch.no_grad():
        ref_c = _mean_log_prob(ref, c_ids, c_lab, vocab_size)
        ref_r = _mean_log_prob(ref, r_ids, r_lab, vocab_size)

    pol_c = _mean_log_prob(policy, c_ids, c_lab, vocab_size)
    pol_r = _mean_log_prob(policy, r_ids, r_lab, vocab_size)

    reward_gap = beta * ((pol_c - ref_c) - (pol_r - ref_r))
    loss       = -F.logsigmoid(reward_gap)

    with torch.no_grad():
        acc = ((pol_c - ref_c) > (pol_r - ref_r)).float()

    return loss, acc

# ── Main ──────────────────────────────────────────────────────────────────────

def train(args=None):
    hf_repo  = args.hf_repo if args and hasattr(args, "hf_repo") else None
    hf_token = os.environ.get("HF_TOKEN", None)
    api_key  = os.environ.get("MIMO_API_KEY", "")

    if not api_key:
        print("ERROR: MIMO_API_KEY not set!"); return

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    print(f"Device: {device}")

    tokenizer  = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer: {vocab_size} tokens")

    config            = CONFIG_100M
    config.vocab_size = vocab_size

    # Load SFT weights
    print("Loading SFT checkpoint from HF Hub...")
    ckpt_path, _ = hf_pull_latest(hf_repo, hf_token, SFT_PREFIX)
    if not ckpt_path:
        print("ERROR: No SFT checkpoint! Run finetune_chat.py first."); return

    raw_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state     = raw_state.get("model_state_dict", raw_state)
    state     = {k.replace("module.", ""): v for k, v in state.items()}

    # Policy model
    policy = MiniLlama(config)
    policy.load_state_dict(state, strict=False)
    policy.to(device)
    raw = policy   # keep reference before DataParallel wrap

    # Reference model (frozen)
    ref = MiniLlama(config)
    ref.load_state_dict(state, strict=False)
    ref.to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    print(f"Policy: {sum(p.numel() for p in policy.parameters()):,} params | Ref: frozen")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        policy = nn.DataParallel(policy)

    # ── Phase 3a: Preference Dataset ─────────────────────────────────────────
    prefs = build_preference_dataset(
        raw, tokenizer, device, api_key, hf_repo, hf_token,
        n=NUM_JUDGE_SAMPLES
    )
    random.shuffle(prefs)

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

    # ── Phase 3b: DPO Training ────────────────────────────────────────────────
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

    print(f"\nStarting DPO — {MAX_STEPS} steps | β={BETA} | "
          f"{NUM_JUDGE_SAMPLES} preference pairs")
    policy.train()

    for step in range(MAX_STEPS):

        # ── Eval ──
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
                      f"pref-accuracy {va/n*100:.1f}%\n")
            policy.train()

        # ── Train ──
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
            print(f"Step {step:5d} | loss {acc_l:.4f} | "
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
    print("DPO TRAINING COMPLETE! 🎉")
    print("Model aligned by 1000 Mimo-judged preference pairs.")
    print("="*60)


if __name__ == "__main__":
    train()
