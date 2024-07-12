import torch
import time
from transformers import AutoTokenizer

from gpt import GPT, GPTConfig
from utils import get_device
from data_loader import DataLoaderLite

device = get_device()

if __name__ == "__main__":
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    train_loader = DataLoaderLite(B=4, T=1024, encoder=gpt2_tokenizer)

    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)

    torch.set_float32_matmul_precision('high')

    # optimize!
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0  # time difference in seconds
        tokens_processed = train_loader.B * train_loader.T
        tokens_per_sec = tokens_processed / dt
        print(f"step {i:4d} | loss: {loss.item():.6f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
