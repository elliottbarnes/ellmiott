"""Small, editable experiments accompanying the six lessons.

Run from the repository root: python docs/experiments.py all
These isolate concepts; they do not create the published GPT checkpoints.
"""

import argparse
import json
import math

import torch
from torch import nn
from torch.nn import functional as F


def show(label, value):
    if isinstance(value, torch.Tensor):
        value = value.detach().tolist()
    print(f"{label}: {json.dumps(value, ensure_ascii=False)}")


def tokens():
    # EDIT: change the corpus or smoothing; predict a transition first.
    text = "aba cab aba cab aba cab "
    smoothing = 1.0
    vocab = sorted(set(text))
    stoi = {character: i for i, character in enumerate(vocab)}
    ids = torch.tensor([stoi[character] for character in text])
    reconstructed = "".join(vocab[i] for i in ids.tolist())
    assert reconstructed == text
    counts = torch.full((len(vocab), len(vocab)), smoothing)
    for previous, following in zip(ids[:-1], ids[1:]):
        counts[previous, following] += 1
    probabilities = counts / counts.sum(dim=1, keepdim=True)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(len(vocab)))
    loss = -probabilities[ids[:-1], ids[1:]].log().mean()
    show("vocabulary", vocab)
    show("round trip", reconstructed)
    show("first shifted pairs", list(zip(text[:8], text[1:9])))
    focus = "a" if "a" in stoi else vocab[0]
    show(f"P(next | {focus!r})", dict(zip(vocab, probabilities[stoi[focus]].tolist())))
    show("training loss, nats per character", loss)
    show("uniform baseline loss", math.log(len(vocab)))


def embeddings():
    # EDIT: use a wider representation, or zero the position table.
    text = "abac"
    channels = 4
    vocab = sorted(set(text))
    ids = torch.tensor([vocab.index(character) for character in text])
    token_table = torch.randn(len(vocab), channels)
    position_table = torch.randn(len(text), channels)
    token_vectors = token_table[ids]
    input_vectors = token_vectors + position_table
    repeated_pair = next(
        ((i, j) for i in range(len(text)) for j in range(i + 1, len(text))
         if text[i] == text[j]), None
    )
    show("input shape", list(input_vectors.shape))
    show("token vectors", token_vectors)
    show("token + position vectors", input_vectors)
    if repeated_pair is not None:
        first, second = repeated_pair
        assert torch.equal(token_vectors[first], token_vectors[second])
        show("repeated character positions", [first, second])
        show("combined vectors equal", torch.equal(input_vectors[first], input_vectors[second]))
    else:
        show("repeated character positions", "none; add a repeated character to compare")


def attention():
    # EDIT: perturb a different position, or change the head dimension.
    length, channels = 4, 3
    x = torch.randn(length, channels)
    wq, wk, wv = (torch.randn(channels, channels) for _ in range(3))

    def forward(inputs, causal):
        queries, keys, values = inputs @ wq, inputs @ wk, inputs @ wv
        scores = queries @ keys.T / math.sqrt(channels)
        if causal:
            mask = torch.ones(length, length, dtype=torch.bool).tril()
            scores = scores.masked_fill(~mask, -torch.inf)
        weights = scores.softmax(dim=-1)
        return weights @ values, weights

    original, weights = forward(x, causal=True)
    altered = x.clone()
    altered[-1] += 3
    causal_result, _ = forward(altered, causal=True)
    unmasked, _ = forward(x, causal=False)
    unmasked_result, _ = forward(altered, causal=False)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(length))
    assert torch.equal(weights.triu(diagonal=1), torch.zeros_like(weights))
    assert torch.allclose(original[:-1], causal_result[:-1])
    assert not torch.allclose(unmasked[:-1], unmasked_result[:-1])
    show("causal attention matrix", weights)
    show("earlier causal outputs unchanged", True)
    show("largest earlier change without mask", (unmasked[:-1] - unmasked_result[:-1]).abs().max())


def block():
    # EDIT: change the expansion factor, or replace GELU with nn.Identity().
    channels = 4
    expansion = 4
    x = torch.randn(1, 3, channels)
    norm = nn.LayerNorm(channels, eps=1e-5)
    branch = nn.Sequential(
        nn.Linear(channels, expansion * channels),
        nn.GELU(approximate="tanh"),
        nn.Linear(expansion * channels, channels),
    )
    normalized = norm(x)
    update = branch(normalized)
    result = x + update
    show("input/branch/output shapes", [list(t.shape) for t in (x, update, result)])
    show("normalized means by position", normalized.mean(dim=-1))
    show("normalized variances by position", normalized.var(dim=-1, unbiased=False))
    assert result.shape == x.shape
    with torch.no_grad():
        for parameter in branch.parameters():
            parameter.zero_()
    assert torch.equal(x + branch(norm(x)), x)
    show("zero branch preserves residual stream", True)


def training():
    # EDIT: learning rate, steps, or the repeating pattern.
    text = "abc" * 30
    learning_rate = 0.1
    steps = 80
    vocab = sorted(set(text))
    ids = torch.tensor([vocab.index(character) for character in text])
    # A V-by-V neural lookup table: each row holds next-character logits.
    logits_table = nn.Parameter(torch.zeros(len(vocab), len(vocab)))
    optimizer = torch.optim.AdamW([logits_table], lr=learning_rate)

    def loss():
        return F.cross_entropy(logits_table[ids[:-1]], ids[1:])

    start_loss = loss().item()
    start_weights = logits_table.detach().clone()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        objective = loss()
        objective.backward()
        optimizer.step()
    end_loss = loss().item()
    assert not torch.equal(start_weights, logits_table.detach())
    assert end_loss < start_loss
    show("training loss before", start_loss)
    show("training loss after", end_loss)
    show("learned transition probabilities", logits_table.softmax(dim=-1))
    show("scope", "toy training data only; no held-out evaluation in this isolated example")


def sampling():
    # EDIT: logits, temperatures, top_k, and seed. Hold other factors fixed.
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    temperatures = [0.5, 1.0, 1.5]
    top_k = 2
    seed = 42
    for temperature in temperatures:
        probabilities = (logits / temperature).softmax(dim=-1)
        entropy = -(probabilities * probabilities.log()).sum()
        show(f"temperature {temperature}", probabilities)
        show("entropy", entropy)
    # Stable descending order gives lower token IDs precedence in a tie.
    selected = torch.argsort(logits, descending=True, stable=True)[:top_k]
    filtered = torch.full_like(logits, -torch.inf)
    filtered[selected] = logits[selected]
    probabilities = filtered.softmax(dim=-1)
    first = torch.multinomial(probabilities, 16, replacement=True,
                              generator=torch.Generator().manual_seed(seed))
    repeat = torch.multinomial(probabilities, 16, replacement=True,
                               generator=torch.Generator().manual_seed(seed))
    assert torch.equal(first, repeat)
    assert len(probabilities.nonzero()) == top_k
    show("top-k probabilities", probabilities)
    show("sampled token IDs", first)
    show("same seed reproduces samples in this implementation", True)


EXPERIMENTS = {
    "tokens": tokens,
    "embeddings": embeddings,
    "attention": attention,
    "block": block,
    "training": training,
    "sampling": sampling,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=[*EXPERIMENTS, "all"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    chosen = EXPERIMENTS if args.experiment == "all" else {args.experiment: EXPERIMENTS[args.experiment]}
    for name, experiment in chosen.items():
        print(f"\n{name}")
        torch.manual_seed(42)
        experiment()


if __name__ == "__main__":
    main()
