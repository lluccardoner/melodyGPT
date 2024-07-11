import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from gpt import GPT, GPTConfig
from utils import get_device

device = get_device()

if __name__ == "__main__":
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("lluccardoner/melodyGPT-song-chords-text-1", split="train")

    df = dataset.to_pandas()
    chords = " ".join(df["chords_str"].dropna())[:1000]
    tokens = gpt2_tokenizer.encode(chords)
    B, T = 4, 32
    buf = torch.tensor(tokens[:B * T + 1])
    buf = buf.to(device)
    x = buf[:-1].view(B, T)
    y = buf[1:].view(B, T)

    model = GPT(GPTConfig())

    # optimize!
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"step {i}, loss: {loss.item()}")
