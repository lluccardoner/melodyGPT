import torch
from datasets import load_dataset
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

    train_loader = DataLoaderLite(B=4, T=32, encoder=gpt2_tokenizer)

    model = GPT(GPTConfig())

    # optimize!
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"step {i}, loss: {loss.item()}")
