import torch
from .model.transformer import MiniLlama
from .config import CONFIG_10M
from .tokenizer import MiniLlamaTokenizer
import sys

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=50, device="cpu"):
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0) # (1, seq_len)
    
    print(prompt, end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # We crop the context to max_position_embeddings if it gets too long
            x_cond = x if x.size(1) <= model.config.max_position_embeddings else x[:, -model.config.max_position_embeddings:]
            
            # Forward pass (no targets)
            logits, _ = model(x_cond)
            
            # Pluck the logits at the final step
            logits = logits[:, -1, :] # (1, vocab_size)
            
            if temperature > 0.0:
                # Apply temperature
                logits = logits / temperature
                
                # Apply top-k
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                    
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                _, idx_next = torch.topk(logits, k=1, dim=-1)
                
            # Append to sequence
            x = torch.cat((x, idx_next), dim=1)
            
            # Decode the newly generated token and print it
            next_token_str = tokenizer.decode([idx_next.item()])
            print(next_token_str, end="", flush=True)
            
    print("\n")
    return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "minillama_step_1000.pt"
    
    tokenizer = MiniLlamaTokenizer()
    tokenizer.load("tokenizer.json")
    
    config = CONFIG_10M
    config.vocab_size = tokenizer.vocab_size
    model = MiniLlama(config)
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        print("Using randomly initialized model!")
        
    model.to(device)
    
    prompt = "Once upon a time, there was a little girl who"
    generate(model, tokenizer, prompt, max_new_tokens=200, device=device)
