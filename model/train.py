import time

import torch
from transformers import AutoTokenizer

from data_loader import DataLoaderLite
from gpt import GPT, GPTConfig
from lr import LRScheduler
from utils import get_device

device = get_device()

if __name__ == "__main__":
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    train_loader = DataLoaderLite(B=4, T=1024, encoder=gpt2_tokenizer)

    torch.set_float32_matmul_precision('high')

    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)

    lr_scheduler = LRScheduler()

    # optimize!
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    for step in range(lr_scheduler.max_steps):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = lr_scheduler.get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0  # time difference in seconds
        tokens_processed = train_loader.B * train_loader.T
        tokens_per_sec = tokens_processed / dt
        print(f"step {step:4d} | loss: {loss.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
