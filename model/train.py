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

    model = GPT(GPTConfig())

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
        dt = (t1 - t0) * 1000  # time difference in milliseconds
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
        print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")
