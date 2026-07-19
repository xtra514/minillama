import torch
from minillama.model.transformer import MiniLlama
from minillama.config import CONFIG_10M
from minillama.utils import apply_weight_init, count_parameters
import time

def run_sanity_check():
    # 1. Initialize
    config = CONFIG_10M
    # Pretend vocab size is 16000 for this test
    config.vocab_size = 16000
    
    model = MiniLlama(config)
    apply_weight_init(model, config)
    
    # 2. Print parameter count
    params = count_parameters(model)
    print(f"Total Parameters: {params / 1e6:.2f} M")
    assert 10 < (params / 1e6) < 13, f"Expected around 10-12M parameters, got {params / 1e6:.2f}M"
    
    # 3. Test forward pass
    batch_size = 4
    seq_len = 256
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    dummy_targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    try:
        logits, loss = model(dummy_input, dummy_targets)
        print("Forward pass successful!")
        print(f"Logits shape: {logits.shape} (Expected: {batch_size, seq_len, config.vocab_size})")
        print(f"Initial loss: {loss.item():.4f}")
    except Exception as e:
        print(f"Forward pass failed! {e}")
        return

    # 4. Overfit on a tiny batch
    print("\n--- Starting Overfit Test (200 steps) ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    
    # We will just overfit on the dummy input
    start_time = time.time()
    for step in range(200):
        logits, loss = model(dummy_input, dummy_targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0 or step == 199:
            print(f"Step {step:3d} | Loss: {loss.item():.4f}")
            
    print(f"Time taken: {time.time() - start_time:.2f}s")
    
    if loss.item() < 0.1:
        print("✅ Sanity check passed: Model successfully overfit a single batch.")
    else:
        print("❌ Sanity check failed: Model did not overfit effectively.")

if __name__ == "__main__":
    run_sanity_check()
