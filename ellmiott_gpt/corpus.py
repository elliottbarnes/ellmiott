"""Character tokenization and a contiguous, held-out corpus split."""

from hashlib import sha256
from pathlib import Path

import torch

CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class CharacterTokenizer:
    def __init__(self, vocab):
        self.vocab = list(vocab)
        if not self.vocab or len(set(self.vocab)) != len(self.vocab) or any(len(c) != 1 for c in self.vocab):
            raise ValueError("Vocabulary must contain distinct single characters")
        self.char_to_id = {character: index for index, character in enumerate(self.vocab)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    def encode(self, text):
        # Training is deliberately strict; the browser reports any replacement.
        return [self.char_to_id[c] for c in text]

    def decode(self, ids):
        return "".join(self.vocab[int(i)] for i in ids)


def load_corpus(path=Path("data/tinyshakespeare.txt")):
    content = Path(path).read_bytes()
    text = content.decode("utf-8")
    if len(text) < 650:
        raise ValueError("Use at least 650 characters so both splits exceed the 64-character context")
    if "\0" in text:
        raise ValueError("Choose plain UTF-8 text, not a binary file")
    tokenizer = CharacterTokenizer.from_text(text)
    if not 2 <= len(tokenizer.vocab) <= 256:
        raise ValueError("Browser-compatible training requires 2 to 256 distinct characters")
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(0.9 * len(encoded))
    metadata = {
        "title": "Tiny Shakespeare", "author": "William Shakespeare",
        "compiled_by": "Andrej Karpathy", "url": CORPUS_URL,
        "sha256": sha256(content).hexdigest(), "characters": len(text),
        "train_characters": split, "validation_characters": len(text) - split,
        "split": "First 90% training; final 10% held out, by character position",
        "license_note": "Underlying Shakespeare text is public domain. Tiny Shakespeare compilation attributed to Andrej Karpathy; no separate data license is asserted.",
    }
    bundled = Path(__file__).resolve().parents[1] / "data" / "tinyshakespeare.txt"
    if sha256(content).hexdigest() != sha256(bundled.read_bytes()).hexdigest():
        metadata.update(title="Custom training text", author="User supplied", compiled_by="User supplied",
                        url="", license_note="User-supplied text; no rights or license are inferred.")
    return encoded[:split], encoded[split:], tokenizer, metadata


def make_batch(data, batch_size, block_size, generator):
    if len(data) <= block_size:
        raise ValueError("Corpus split must be longer than the context window")
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    positions = starts[:, None] + torch.arange(block_size)[None, :]
    return data[positions], data[positions + 1]
