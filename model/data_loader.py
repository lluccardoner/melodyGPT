from typing import List

import torch
from torch.utils.data import random_split
from datasets import load_dataset
from transformers import PreTrainedTokenizer


class DataLoaderLite:
    def __init__(self, B, T, encoder: PreTrainedTokenizer, val_split: float = 0.1, seed: int = 42):
        self.B = B
        self.T = T
        self.encoder = encoder

        # at init load tokens from disk and store them in memory
        dataset = load_dataset("lluccardoner/melodyGPT-song-chords-text-1", split="train")

        all_tokens = []
        for doc in dataset:
            if doc_chords := doc["chords_str"]:
                doc_tokens = self._tokenize_document(doc_chords)
                all_tokens.extend(doc_tokens)

        generator = torch.Generator().manual_seed(seed)
        train_split, val_split = random_split(all_tokens, [1-val_split, val_split], generator=generator)
        self.train_tokens = torch.tensor(train_split)
        self.val_tokens = torch.tensor(val_split)
        print(f"loaded {len(self.train_tokens)} train tokens and {len(self.val_tokens)} validation tokens")

        # state
        self.reset_train()
        self.reset_val()

    def reset_train(self):
        self.current_position_train = 0

    def reset_val(self):
        self.current_position_val = 0

    def _tokenize_document(self, doc) -> List:
        # tokenizes a single document and returns a numpy array of uint16 tokens
        eot = self.encoder.eos_token_id
        tokens = [eot]  # the special <|endoftext|> token delimits all documents
        tokens.extend(self.encoder.encode(doc))
        return tokens

    def next_batch_train(self):
        B, T = self.B, self.T
        buf = self.train_tokens[self.current_position_train: self.current_position_train + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position_train += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position_train + (B * T + 1) > len(self.train_tokens):
            self.current_position_train = 0
        return x, y

    def next_batch_val(self):
        B, T = self.B, self.T
        buf = self.val_tokens[self.current_position_val: self.current_position_val + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position_val += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position_val + (B * T + 1) > len(self.val_tokens):
            self.current_position_val = 0
        return x, y
