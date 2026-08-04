"""
dpo_train.py  —  Phase 3: DPO with Anthropic HH-RLHF Dataset

Uses the same dataset as Llama 2, Vicuna, Mistral, Zephyr etc.
169K human-labeled preference pairs — no API judge needed.

Pipeline:
  1. Load Anthropic/hh-rlhf (human chosen vs rejected pairs)
  2. Parse into (instruction, chosen, rejected) format
  3. Train with DPO loss against frozen SFT reference model
  4. Save checkpoints to HuggingFace

Kaggle usage:
    import minillama.dpo_train as dt
    dt.train(args=Args())

Requires: HF_TOKEN as Kaggle secret.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, json, re, random
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

NUM_PAIRS      = 10_000    # subset of HH-RLHF to use (169K available)
MAX_LENGTH     = 512       # training sequence length
BATCH_SIZE     = 2
GRAD_ACCUM     = 8
LEARNING_RATE  = 5e-7      # tiny — preserve SFT knowledge
BETA           = 0.1       # DPO temperature
MAX_STEPS      = 2_000
WARMUP_STEPS   = 100
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
        print(f"  ⚠ HF upload failed: {e}")


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
        return local, step
    except Exception as e:
        print(f"  ⚠ HF pull failed: {e}")
        return None, None

# ── HH-RLHF Parser ───────────────────────────────────────────────────────────

def parse_hh_pair(chosen_text, rejected_text):
    """
    HH-RLHF format:
      '\n\nHuman: <instruction>\n\nAssistant: <response>'

    Returns (instruction, chosen_response, rejected_response) or None.
    """
    def extract_last_exchange(text):
        # Split on Human turns, take last one
        human_parts = text.split("\n\nHuman: ")
        if len(human_parts) < 2:
            return None, None
        last = human_parts[-1]
        if "\n\nAssistant: " not in last:
            return None, None
        instruction, response = last.split("\n\nAssistant: ", 1)
        return instruction.strip(), response.strip()

    instr_c, resp_c = extract_last_exchange(chosen_text)
    instr_r, resp_r = extract_last_exchange(rejected_text)

    if not (instr_c and resp_c and resp_r):
        return None

    # Sanity: responses must differ and be non-trivial
    if resp_c == resp_r or len(resp_c) < 5 or len(resp_r) < 5:
        return None

    # Use instruction from chosen (they share the same prompt)
    return {"instruction": instr_c,
            "chosen":      resp_c,
            "rejected":    resp_r}


def load_hh_rlhf(n=10_000, seed=42):
    """Load and parse Anthropic HH-RLHF preference pairs."""
    print(f"Loading Anthropic/hh-rlhf (n={n})...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    ds = ds.shuffle(seed=seed)

    pairs = []
    for row in ds:
        if len(pairs) >= n:
            break
        parsed = parse_hh_pair(row["chosen"], row["rejected"])
        if parsed:
            pairs.append(parsed)

    print(f"✓ Parsed {len(pairs)} valid preference pairs from HH-RLHF.")
    return pairs

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
        # Truncate instruction to avoid overflow
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
        print("ERROR: No SFT checkpoint found! Run finetune_chat.py first.")
        return

    raw_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state     = raw_state.get("model_state_dict", raw_state)
    state     = {k.replace("module.", ""): v for k, v in state.items()}

    # Policy model (trained)
    policy = MiniLlama(config)
    policy.load_state_dict(state, strict=False)
    policy.to(device)

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

    # ── Load HH-RLHF dataset ─────────────────────────────────────────────────
    pairs = load_hh_rlhf(n=NUM_PAIRS)
    random.shuffle(pairs)

    split    = int(0.95 * len(pairs))
    train_ds = DPODataset(pairs[:split], tokenizer, MAX_LENGTH)
    val_ds   = DPODataset(pairs[split:], tokenizer, MAX_LENGTH)
    print(f"DPO dataset — Train: {len(train_ds)} | Val: {len(val_ds)}")

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

    print(f"\nStarting DPO — {MAX_STEPS} steps | β={BETA} | "
          f"{len(train_ds)} pairs (Anthropic HH-RLHF)")
    policy.train()

    for step in range(MAX_STEPS):

        # ── Eval ──────────────────────────────────────────────────────────────
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
    print(f"Trained on {len(train_ds)} human preference pairs (HH-RLHF)")
    print("Same training method as Llama 2, Vicuna, Mistral!")
    print("="*60)


if __name__ == "__main__":
    train()
