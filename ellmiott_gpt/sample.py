"""Generate locally from exactly the JSON weights downloaded by the browser."""

import argparse
import math
from pathlib import Path

import torch

from .corpus import CharacterTokenizer
from .export import load_export


@torch.inference_mode()
def sample(model, tokenizer, prompt, count=200, temperature=0.8, top_k=0, seed=42, disabled_head=None):
    if count < 0 or not math.isfinite(temperature) or temperature < 0 or top_k < 0:
        raise ValueError("Count/top-k and finite temperature must be nonnegative")
    context = tokenizer.encode(prompt or "\n")
    generator = torch.Generator().manual_seed(seed)
    generated = []
    for _ in range(count):
        ids = torch.tensor([context[-model.config.block_size:]])
        logits, _ = model(ids, disabled_head=disabled_head)
        logits = logits[0, -1].double()
        if temperature == 0:
            next_id = logits.argmax().item()
            context.append(next_id)
            generated.append(next_id)
            continue
        # Center before scaling, so even tiny positive temperatures cannot turn
        # the largest logit into +infinity and produce an undefined softmax.
        logits = (logits - logits.max()) / temperature
        if 0 < top_k < len(logits):
            # Keep exactly k candidates, including a deterministic choice at ties.
            candidates = torch.argsort(logits, descending=True, stable=True)[:top_k]
            filtered = torch.full_like(logits, float("-inf"))
            filtered[candidates] = logits[candidates]
            logits = filtered
        next_id = torch.multinomial(logits.softmax(-1), 1, generator=generator).item()
        context.append(next_id)
        generated.append(next_id)
    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("site/models/trained.json"))
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-head", help="Zero an attention head, e.g. 0:1")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, payload = load_export(args.checkpoint)
    tokenizer = CharacterTokenizer(payload["vocab"])
    disabled = None
    if args.disable_head:
        try:
            disabled = tuple(int(i) for i in args.disable_head.split(":"))
            if len(disabled) != 2:
                raise ValueError()
        except ValueError:
            parser.error("--disable-head must be LAYER:HEAD")
    try:
        result = sample(model, tokenizer, args.prompt, args.count, args.temperature, args.top_k, args.seed, disabled)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(args.prompt + result)


if __name__ == "__main__":
    main()
