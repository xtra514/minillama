import torch
from model.transformer import MiniLlama
from config import CONFIG_10M
from transformers import PreTrainedTokenizerFast
import os

def load_model(checkpoint_path, tokenizer_path):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    
    config = CONFIG_10M
    config.vocab_size = tokenizer.vocab_size
    
    model = MiniLlama(config)
    
    # Load state dict (handling DataParallel prefixes if they exist)
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(x)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            x = torch.cat((x, next_token), dim=1)
            
            # Stop if EOS token generated
            if tokenizer.eos_token_id and next_token.item() == tokenizer.eos_token_id:
                break
                
    generated_text = tokenizer.decode(x[0].tolist(), skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    # Choose your highest checkpoint number! (e.g. minillama_step_1900.pt)
    CHECKPOINT = "minillama_step_100.pt" 
    TOKENIZER = "data/tokenizer.json"
    
    if not os.path.exists(CHECKPOINT):
        print(f"Error: Could not find {CHECKPOINT}. Make sure you typed the step number correctly!")
        exit(1)
        
    print(f"Loading {CHECKPOINT}...")
    model, tokenizer = load_model(CHECKPOINT, TOKENIZER)
    
    print("\n--- AI Generation ---")
    prompt = "Once upon a time, there was a little girl named Lily. She"
    print(f"Prompt: {prompt}")
    
    output = generate_text(model, tokenizer, prompt, max_new_tokens=150)
    print(f"\nGenerated: {output}\n---------------------")
