import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from .tokenizer import MiniLlamaTokenizer
import os
from tqdm import tqdm

class PretokenizedDataset(Dataset):
    def __init__(self, token_ids_path, max_length=1024):
        # Load the pre-tokenized 1D tensor from disk
        self.data = torch.load(token_ids_path)
        self.max_length = max_length
        # Calculate how many full sequences we can extract
        self.num_samples = (len(self.data) - 1) // max_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.max_length
        end_idx = start_idx + self.max_length + 1
        chunk = self.data[start_idx:end_idx]
        
        input_ids = chunk[:-1].clone().detach().long()
        targets = chunk[1:].clone().detach().long()
        
        return input_ids, targets

def prepare_and_get_dataloaders(
    data_dir="data",
    dataset_name="roneneldan/TinyStories",
    batch_size=8,
    max_length=1024,
    vocab_size=16000
):
    """
    Downloads dataset, trains tokenizer, pre-tokenizes the entire dataset, 
    caches it to disk, and returns dataloaders.
    """
    os.makedirs(data_dir, exist_ok=True)
    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    train_bin_path = os.path.join(data_dir, "train.pt")
    val_bin_path = os.path.join(data_dir, "val.pt")
    
    # 1. Load dataset
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = split['train']
    val_ds = split['test']
    
    # 2. Tokenizer
    tokenizer = MiniLlamaTokenizer(vocab_size=vocab_size)
    if not os.path.exists(tokenizer_path):
        print("Training Tokenizer...")
        train_subset = train_ds.select(range(min(100000, len(train_ds))))
        tokenizer.train_on_dataset(train_subset)
        tokenizer.save(tokenizer_path)
    else:
        print("Loading Tokenizer...")
        tokenizer.load(tokenizer_path)

    # 3. Pre-tokenize Data
    def pre_tokenize(ds, path):
        if os.path.exists(path):
            print(f"Pre-tokenized data found at {path}, skipping tokenization.")
            return
            
        print(f"Pre-tokenizing data to {path}...")
        all_tokens = []
        # Tokenize the full dataset
        for row in tqdm(ds, desc="Tokenizing"):
            text = row["text"]
            if not text.strip(): continue
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            
        tensor_data = torch.tensor(all_tokens, dtype=torch.long)
        torch.save(tensor_data, path)
        print(f"Saved {len(all_tokens)} tokens to {path}")

    pre_tokenize(train_ds, train_bin_path)
    pre_tokenize(val_ds, val_bin_path)

    # 4. Create Dataloaders
    train_dataset = PretokenizedDataset(train_bin_path, max_length=max_length)
    val_dataset = PretokenizedDataset(val_bin_path, max_length=max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, tokenizer
