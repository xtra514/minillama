"""
chat.py — Interactive CLI chat with MiniLlama-125M

Usage:
    python chat.py
    python chat.py --model minillama_125m_chat_step_2999.pt
    python chat.py --temp 0.7 --top_k 40
"""
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from minillama.model.transformer import MiniLlama
from minillama.config import CONFIG_100M
import argparse, os, glob

INSTRUCTION_PREFIX = "### Instruction:\n"
RESPONSE_PREFIX    = "\n### Response:\n"

def find_latest_chat_model():
    ckpts = sorted(glob.glob("minillama_125m_chat_step_*.pt"),
                   key=lambda p: int(p.split("_")[-1].replace(".pt", "")))
    return ckpts[-1] if ckpts else None

@torch.no_grad()
def respond(model, tokenizer, instruction, device,
            max_new_tokens=300, temperature=0.8, top_k=50):
    prompt = INSTRUCTION_PREFIX + instruction + RESPONSE_PREFIX
    ids    = tokenizer.encode(prompt, add_special_tokens=False).ids
    x      = torch.tensor([ids], dtype=torch.long, device=device)
    eos    = tokenizer.token_to_id("</s>") or 2

    generated = []
    for i in range(max_new_tokens):
        x_cond = x[:, -model.config.max_position_embeddings:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / temperature

        # Ban EOS for first 10 tokens
        if i < 10:
            logits[:, eos] = float("-inf")

        # Top-k filter
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
        x = torch.cat([x, next_tok], dim=1)
        generated.append(next_tok.item())
        if next_tok.item() == eos:
            break

    return tokenizer.decode(generated).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default=None,  help="Path to chat model checkpoint")
    parser.add_argument("--tokenizer", default="minillama/data/tokenizer_32k.json")
    parser.add_argument("--temp",      type=float, default=0.8)
    parser.add_argument("--top_k",     type=int,   default=50)
    parser.add_argument("--max_tokens",type=int,   default=300)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer  = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()

    # Find model
    model_path = args.model or find_latest_chat_model()
    if not model_path:
        print("No chat model found. Train one with finetune_chat.py first!")
        return
    print(f"Loading model: {model_path}")

    config = CONFIG_100M
    config.vocab_size = vocab_size
    model = MiniLlama(config)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)\n")

    print("=" * 60)
    print("MiniLlama Chat  |  type 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("MiniLlama: Goodbye!")
            break

        response = respond(
            model, tokenizer, user_input, device,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k
        )
        print(f"MiniLlama: {response}")


if __name__ == "__main__":
    main()
