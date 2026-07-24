"""
train_tokenizer.py
Trains a 32K vocab ByteLevel BPE tokenizer on OpenWebText.
Run on Kaggle before pretraining:
    python -m minillama.train_tokenizer
"""
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from datasets import load_dataset

import os

VOCAB_SIZE  = 32000
_REPO_ROOT  = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH   = os.path.join(_REPO_ROOT, "data", "tokenizer_32k.json")
TRAIN_SAMPLES = 200_000

def main():
    os.makedirs(os.path.join(_REPO_ROOT, "data"), exist_ok=True)

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

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer, length=TRAIN_SAMPLES)

    # Add BOS/EOS wrapping
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B:1 </s>:1",
        special_tokens=[("<s>", tokenizer.token_to_id("<s>")),
                        ("</s>", tokenizer.token_to_id("</s>"))],
    )

    tokenizer.save(SAVE_PATH)
    print(f"Tokenizer saved to {SAVE_PATH}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Quick sanity check
    enc = tokenizer.encode("Hello! How are you doing today?")
    print(f"Test encode: {enc.tokens}")
    print(f"Test decode: {tokenizer.decode(enc.ids)}")

if __name__ == "__main__":
    main()
