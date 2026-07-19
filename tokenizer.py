import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset

class MiniLlamaTokenizer:
    def __init__(self, vocab_size=16000):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B:1 </s>:1",
            special_tokens=[
                ("<s>", 1),
                ("</s>", 2),
            ],
        )
        self.trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<unk>", "<s>", "</s>", "<pad>"]
        )

    def train_on_dataset(self, dataset, text_column="text"):
        """Train the tokenizer on a HuggingFace dataset."""
        print(f"Training tokenizer on dataset with vocab size {self.vocab_size}...")
        
        def batch_iterator():
            for i in range(0, len(dataset), 1000):
                yield dataset[i : i + 1000][text_column]

        self.tokenizer.train_from_iterator(batch_iterator(), trainer=self.trainer)
        print("Tokenizer training complete.")

    def save(self, path="tokenizer.json"):
        self.tokenizer.save(path)

    def load(self, path="tokenizer.json"):
        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)
