"""Train melodyGPT v1 with the original GPT2 tokenizer"""
import os
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

    total_batch_size = 524288  # 2**19, ~0.5M, in number of tokens (from GPT2)
    B = 16  # micro batch size
    T = 1024  # sequence length
    assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
    grad_accum_steps = total_batch_size // (B * T)
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    data_loader = DataLoaderLite(B=B, T=T, encoder=gpt2_tokenizer)

    torch.set_float32_matmul_precision('high')

    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)

    max_steps = int(len(data_loader.train_tokens) / total_batch_size)
    warmup_steps = int(0.05 * len(data_loader.train_tokens) / total_batch_size)  # GPT-2 warmup over 375 million tokens but we have less
    print(f"{max_steps=}, {warmup_steps=}")
    lr_scheduler = LRScheduler(max_steps=max_steps, warmup_steps=warmup_steps)

    # Loggin
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f:  # open for writing to clear the file
        pass

    # optimize!
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
    for step in range(lr_scheduler.max_steps):
        t0 = time.time()
        is_last_step = (step == max_steps - 1)

        # Evaluation
        if step % 5 == 0 or is_last_step:
            model.eval()
            data_loader.reset_val()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 5
                for _ in range(val_loss_steps):
                    x, y = data_loader.next_batch_val()
                    x, y = x.to(device), y.to(device)
                    logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()

                # Model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': model.state_dict(),
                    'config': model.config,
                    'step': step,
                    'val_loss': val_loss_accum
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

        # Training
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = data_loader.next_batch_train()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
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
        tokens_processed = data_loader.B * data_loader.T * grad_accum_steps
        tokens_per_sec = tokens_processed / dt
        with open(log_file, "a") as f:
            log_msg = f"step {step:5d} | train_loss: {loss_accum:.6f}|  val_loss: {val_loss_accum:.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            print(log_msg)
            f.write(log_msg + "\n")
