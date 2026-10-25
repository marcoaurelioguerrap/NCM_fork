"""Character-level tokenizer for the target side (one order per line).

Orders are a tiny, closed alphabet: a letter mnemonic, unit ids, digits, commas,
minus signs and unit-type underscores.  A character tokenizer makes exact
reconstruction the only acceptable result, unlike GPT-2 subwords which split
coordinates and ids in ways a small VQ-VAE cannot reliably invert.
"""

from __future__ import annotations

__all__ = ["ALPHABET", "CharTokenizer"]

# Space, comma, minus, underscore, digits, upper and lower case letters.
ALPHABET = " ,-_.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
PAD_ID = 0
EOS_ID = 1


class CharTokenizer:
    """A fixed 68-symbol alphabet with explicit pad and end-of-sequence ids."""

    def __init__(self):
        self.itos = ["<pad>", "<eos>", *ALPHABET]
        self.stoi = {char: index for index, char in enumerate(self.itos)}
        self.pad_token_id = PAD_ID
        self.eos_token_id = EOS_ID
        self.vocab_size = len(self.itos)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[char] for char in text]
        except KeyError as exc:
            raise ValueError(f"character {exc.args[0]!r} is outside the order alphabet") from None

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        chars = []
        for index in ids:
            index = int(index)
            if index in (PAD_ID, EOS_ID):
                if skip_special_tokens:
                    break
                continue
            chars.append(self.itos[index])
        return "".join(chars)

    def encode_padded(self, text: str, context_length: int) -> list[int]:
        ids = self.encode(text) + [EOS_ID]
        if len(ids) > context_length:
            raise ValueError(
                f"order {text!r} needs {len(ids)} characters with EOS, "
                f"context is {context_length}"
            )
        return ids + [PAD_ID] * (context_length - len(ids))
