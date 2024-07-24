import os

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, PreTrainedTokenizer

from gpt import GPT
from utils import get_device

device = get_device()


def get_tokens(input_str: str, encoder: PreTrainedTokenizer, num_sequences: int = 5) -> torch.Tensor:
    tokens = encoder.encode(input_str)
    tokens = torch.tensor(tokens, dtype=torch.long)  # (num_tokens,)
    tokens = tokens.unsqueeze(0).repeat(num_sequences, 1)  # (num_sequences, num_tokens)
    output = tokens.to(device)
    return output


def generate_predictions(x: torch.Tensor, model: GPT, max_length: int = 30,
                         seed: int = 42) -> torch.Tensor:
    model.eval()
    model.to(device)
    # generate! right now x is (B, T) where B = num_sequences, T = num_tokens
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            logits, _ = model(x)  # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input_str to sum to 1
            ix = torch.multinomial(topk_probs, 1)  # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)
    return x


def decode_and_print_tokens(tokens: torch.Tensor, encoder):
    for i in range(tokens.size(0)):
        sequence_tokens = tokens[i, :].tolist()
        decoded = encoder.decode(sequence_tokens)
        print(">", decoded)


def generate(input_str: str, encoder: PreTrainedTokenizer, model: GPT):
    initial_tokens = get_tokens(input_str, encoder)
    generated_tokens = generate_predictions(initial_tokens, model)
    decode_and_print_tokens(generated_tokens, encoder)


if __name__ == "__main__":
    # Pachelbel Canon in D chords: "D A Bm F#m G D G A"
    chords = "D A Bm F#m"

    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2 = GPT.from_pretrained('gpt2')
    print("Generating with GPT2")
    generate(chords, gpt2_tokenizer, gpt2)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    melodyGPT1_checkpoint_path = os.path.join(dir_path, "../logs/melodyGPT1/model_00030.pt")
    melodyGPT1_checkpoint = torch.load(melodyGPT1_checkpoint_path, map_location=device)
    melodyGPT1 = GPT(melodyGPT1_checkpoint["config"])
    state_dict = melodyGPT1_checkpoint["model"]
    melodyGPT1_state_dict = melodyGPT1.state_dict()
    for key in state_dict:
        if not key.endswith("attn.bias"):  # TODO model was trained wit it but it is not used
            key_name = key.replace("_orig_mod.", "")
            assert state_dict[key].shape == melodyGPT1_state_dict[key_name].shape
            with torch.no_grad():
                melodyGPT1_state_dict[key_name].copy_(state_dict[key])
    print("Generating with MelodyGPT v1")
    generate(chords, gpt2_tokenizer, melodyGPT1)

    chords_gpt2_tokenizer = AutoTokenizer.from_pretrained("lluccardoner/melodyGPT-song-chords-tokenizer-1")

    melodyGPT2_checkpoint_path = os.path.join(dir_path, "../logs/melodyGPT2/model_00026.pt")
    melodyGPT2_checkpoint = torch.load(melodyGPT2_checkpoint_path, map_location=device)
    melodyGPT2 = GPT(melodyGPT2_checkpoint["config"])
    state_dict = melodyGPT2_checkpoint["model"]
    melodyGPT2_state_dict = melodyGPT2.state_dict()
    for key in state_dict:
        if not key.endswith("attn.bias"):  # TODO model was trained wit it but it is not used
            key_name = key.replace("_orig_mod.", "")
            assert state_dict[key].shape == melodyGPT2_state_dict[key_name].shape
            with torch.no_grad():
                melodyGPT2_state_dict[key_name].copy_(state_dict[key])
    print("Generating with MelodyGPT v2")
    generate(chords, chords_gpt2_tokenizer, melodyGPT2)
