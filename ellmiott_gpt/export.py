"""Portable, finite JSON weights and browser/PyTorch comparison fixtures."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import torch

from .model import GPT, GPTConfig

FORMAT = "ellmiott-gpt-v1"


def rounded(value):
    if isinstance(value, float):
        return float(format(value, ".8g"))
    if isinstance(value, list):
        return [rounded(x) for x in value]
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")


def export_checkpoint(model, tokenizer, path, step):
    payload = {
        "format": FORMAT, "config": asdict(model.config), "vocab": tokenizer.vocab, "step": step,
        "tensors": {name: {"shape": list(tensor.shape), "data": rounded(tensor.detach().cpu().reshape(-1).tolist())}
                    for name, tensor in model.state_dict().items()},
    }
    write_json(path, payload)
    content = Path(path).read_bytes()
    return {"file": Path(path).name, "step": step, "bytes": len(content), "sha256": sha256(content).hexdigest()}


def load_export(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT:
        raise ValueError("Unsupported checkpoint format")
    model = GPT(GPTConfig(**payload["config"]))
    tensors = {name: torch.tensor(spec["data"], dtype=torch.float32).reshape(spec["shape"])
               for name, spec in payload["tensors"].items()}
    model.load_state_dict(tensors, strict=True)
    return model.eval(), payload


@torch.inference_mode()
def parity_cases(path, checkpoint):
    # Re-load the rounded exported values: the fixture describes exactly the
    # downloadable weights, rather than higher precision hidden originals.
    model, payload = load_export(path)
    vocab = payload["vocab"]
    cases = []
    for prompt, disabled in [("To be, or not to be", None), ("ROMEO:\n", (0, 1))]:
        if not all(c in vocab for c in prompt):
            prompt = "".join(vocab[:min(16, model.config.block_size)])
        prompt = prompt[:model.config.block_size]
        disabled = (0, min(1, model.config.n_head - 1)) if disabled else None
        ids = [vocab.index(c) for c in prompt]
        logits, attention = model(torch.tensor([ids]), disabled_head=disabled, return_attention=True)
        layer, head = (0, min(1, model.config.n_head - 1)) if disabled else (min(1, model.config.n_layer - 1), min(2, model.config.n_head - 1))
        cases.append({
            "checkpoint": checkpoint, "prompt": prompt, "ids": ids,
            "logits": rounded(logits[0, -1].tolist()),
            "attention": rounded(attention[layer][0, head].tolist()),
            "layer": layer, "head": head,
            "disabledHead": {"layer": disabled[0], "head": disabled[1]} if disabled else None,
        })
    return cases
