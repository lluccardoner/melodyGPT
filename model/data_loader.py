from typing import List

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer


class DataLoaderLite:
    def __init__(self, B, T, encoder: PreTrainedTokenizer):
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

        self.tokens = torch.tensor(all_tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def _tokenize_document(self, doc) -> List:
        # tokenizes a single document and returns a numpy array of uint16 tokens
        eot = self.encoder.eos_token_id
        tokens = [eot]  # the special <|endoftext|> token delimits all documents
        tokens.extend(self.encoder.encode(doc))
        return tokens

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
