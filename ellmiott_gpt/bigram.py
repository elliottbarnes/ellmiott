"""A count-based baseline: predict a character using only the previous one.

python -m ellmiott_gpt.bigram --count 200 --seed 42
This deliberately tiny model is useful before adding a transformer. It uses
Laplace smoothing (one extra observation per pair), not gradient descent.
"""

import argparse
import json

import torch

from .corpus import load_corpus


def fit_bigram(ids, vocab_size):
    pairs = ids[:-1] * vocab_size + ids[1:]
    counts = torch.bincount(pairs, minlength=vocab_size * vocab_size).reshape(vocab_size, vocab_size).float() + 1
    return counts / counts.sum(dim=-1, keepdim=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < 0:
        parser.error("Count must not be negative")
    torch.set_num_threads(1)
    train, validation, tokenizer, _ = load_corpus()
    probabilities = fit_bigram(train, len(tokenizer.vocab))
    loss = -probabilities[validation[:-1], validation[1:]].log().mean().item()
    print(json.dumps({"model": "smoothed character bigram", "validation_loss": loss,
                      "evaluation": "Every adjacent pair in the held-out final 10%; not the GPT's sampled batches",
                      "conditional_probabilities": probabilities.numel()}))
    generator = torch.Generator().manual_seed(args.seed)
    ids = tokenizer.encode("\n")
    for _ in range(args.count):
        ids.append(torch.multinomial(probabilities[ids[-1]], 1, generator=generator).item())
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
