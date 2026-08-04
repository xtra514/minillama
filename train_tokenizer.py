"""
train_tokenizer.py
Trains a 32K vocab ByteLevel BPE tokenizer on OpenWebText.
Run on Kaggle before pretraining:
    python -m minillama.train_tokenizer

Fast path: if tokenizer already on HF Hub, downloads in seconds (skips 3-min training).
"""
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from datasets import load_dataset

VOCAB_SIZE    = 32000
_REPO_ROOT    = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH     = os.path.join(_REPO_ROOT, "data", "tokenizer_32k.json")
TRAIN_SAMPLES = 200_000
HF_FILENAME   = "tokenizer_32k.json"


def _try_hf_download(hf_repo, hf_token):
    """Download tokenizer from HF Hub if available — saves 3 minutes."""
    try:
        from huggingface_hub import hf_hub_download, HfApi
        api   = HfApi(token=hf_token)
        files = list(api.list_repo_files(hf_repo))
        if HF_FILENAME not in files:
            return False
        print(f"  Downloading {HF_FILENAME} from HF Hub (skipping training)...")
        local = hf_hub_download(repo_id=hf_repo, filename=HF_FILENAME,
                                 token=hf_token)
        import shutil
        os.makedirs(os.path.join(_REPO_ROOT, "data"), exist_ok=True)
        shutil.copy(local, SAVE_PATH)
        print(f"  ✓ Tokenizer ready: {SAVE_PATH}")
        return True
    except Exception as e:
        print(f"  ⚠ HF download failed ({e}), will train locally.")
        return False


def _hf_upload(hf_repo, hf_token):
    """Upload tokenizer to HF Hub so future sessions skip training."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.create_repo(hf_repo, exist_ok=True, private=False)
        api.upload_file(path_or_fileobj=SAVE_PATH,
                        path_in_repo=HF_FILENAME,
                        repo_id=hf_repo)
        print(f"  ✓ Tokenizer uploaded → hf.co/{hf_repo}")
    except Exception as e:
        print(f"  ⚠ HF upload failed: {e}")


def main(hf_repo=None, hf_token=None):
    os.makedirs(os.path.join(_REPO_ROOT, "data"), exist_ok=True)

    hf_token = hf_token or os.environ.get("HF_TOKEN")
    hf_repo  = hf_repo  or os.environ.get("HF_REPO", "galaxy2121/minillama-125m")

    # ── Fast path: download from HF Hub ──────────────────────────────────────
    if hf_token and _try_hf_download(hf_repo, hf_token):
        tok = Tokenizer.from_file(SAVE_PATH)
        enc = tok.encode("Hello! How are you doing today?")
        print(f"Vocab size: {tok.get_vocab_size()}")
        print(f"Test encode: {enc.tokens}")
        print(f"Test decode: {tok.decode(enc.ids)}")
        return

    # ── Slow path: train from scratch (~3 min) ────────────────────────────────
    print(f"Loading {TRAIN_SAMPLES} OpenWebText samples for tokenizer training...")
    ds = load_dataset("openwebtext", split="train", streaming=True)

    def text_iterator():
        for i, sample in enumerate(ds):
            if i >= TRAIN_SAMPLES:
                break
            yield sample["text"]

    print(f"Training ByteLevel BPE tokenizer (vocab={VOCAB_SIZE})...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        show_progress=True,
    )

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer,
                                   length=TRAIN_SAMPLES)

    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B:1 </s>:1",
        special_tokens=[("<s>",  tokenizer.token_to_id("<s>")),
                        ("</s>", tokenizer.token_to_id("</s>"))],
    )

    tokenizer.save(SAVE_PATH)
    print(f"Tokenizer saved to {SAVE_PATH}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    enc = tokenizer.encode("Hello! How are you doing today?")
    print(f"Test encode: {enc.tokens}")
    print(f"Test decode: {tokenizer.decode(enc.ids)}")

    # Upload so future sessions download in seconds
    if hf_token:
        _hf_upload(hf_repo, hf_token)


if __name__ == "__main__":
    main()
