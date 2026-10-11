"""Train locally and export three genuine stages of a small GPT.

Run from the repository root: python -m ellmiott_gpt.train --steps 4000
Validation uses the same fixed held-out batches at every evaluation. A separate
random generator samples training batches; evaluating never changes training.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F

from .corpus import load_corpus, make_batch
from .export import FORMAT, export_checkpoint, parity_cases, write_json
from .model import GPT, GPTConfig


def evaluate(model, batches):
    model.eval()
    with torch.inference_mode():
        losses = [F.cross_entropy(model(x)[0].reshape(-1, model.config.vocab_size), y.reshape(-1)).item()
                  for x, y in batches]
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4000, help="Total optimizer steps, including resumed steps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--early-step", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data", type=Path, default=Path("data/tinyshakespeare.txt"))
    parser.add_argument("--output", type=Path, default=Path("site/models"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures/parity.json"))
    parser.add_argument("--resume", type=Path, help="Resume a local training.pt file; keep all other training flags the same")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.eval_batches, args.threads) < 1:
        parser.error("Steps, batch size, evaluation intervals and threads must be positive")
    if not 0 < args.early_step <= args.steps or not args.learning_rate > 0:
        parser.error("Early step must be in 1..steps and learning rate must be positive")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    train, validation, tokenizer, corpus = load_corpus(args.data)
    config = GPTConfig(vocab_size=len(tokenizer.vocab))
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=0.1)
    training_generator = torch.Generator().manual_seed(args.seed + 1)
    train_eval_generator = torch.Generator().manual_seed(args.seed + 2)
    val_eval_generator = torch.Generator().manual_seed(args.seed + 3)
    train_eval = [make_batch(train, args.batch_size, config.block_size, train_eval_generator) for _ in range(args.eval_batches)]
    val_eval = [make_batch(validation, args.batch_size, config.block_size, val_eval_generator) for _ in range(args.eval_batches)]
    args.output.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    history, checkpoints, fixtures = [], [], []
    start_step, previous_seconds = 0, 0.0
    settings = {"seed": args.seed, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                "eval_batches": args.eval_batches, "early_step": args.early_step, "threads": args.threads}
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved["settings"] != settings or saved["corpus_sha256"] != corpus["sha256"]:
            parser.error("Resume requires identical settings and corpus")
        if saved["config"] != asdict(config):
            parser.error("Resume architecture differs from the checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        training_generator.set_state(saved["generator"])
        start_step, previous_seconds = saved["step"], saved["seconds"]
        if start_step >= args.steps:
            parser.error("Total steps must be larger than the resumed step")
        history, checkpoints, fixtures = saved["history"], saved["checkpoints"], saved["fixtures"]
        checkpoints = [c for c in checkpoints if c["id"] != "trained"]
        fixtures = [case for case in fixtures if case["checkpoint"] != "trained"]
        # Preserve initial/early exports when resuming into another directory.
        for item in checkpoints:
            source = args.resume.parent / item["file"]
            (args.output / item["file"]).write_bytes(source.read_bytes())

    started = time.perf_counter()

    def record(step):
        item = {"step": step, "train_loss": evaluate(model, train_eval), "validation_loss": evaluate(model, val_eval)}
        history.append(item)
        print(json.dumps({**item, "seconds": round(previous_seconds + time.perf_counter() - started, 2)}), flush=True)
        return item

    def checkpoint(identifier, label, step, metrics):
        path = args.output / f"{identifier}.json"
        metadata = export_checkpoint(model, tokenizer, path, step)
        checkpoints.append({"id": identifier, "label": label, **metadata, "validation_loss": metrics["validation_loss"]})
        (args.run_dir / path.name).write_bytes(path.read_bytes())
        if identifier != "early":
            fixtures.extend(parity_cases(path, identifier))

    if start_step == 0:
        metrics = record(0)
        checkpoint("initial", "Untrained", 0, metrics)

    for step in range(start_step + 1, args.steps + 1):
        x, y = make_batch(train, args.batch_size, config.block_size, training_generator)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step in (args.early_step, args.steps):
            metrics = record(step)
            if step == args.early_step:
                checkpoint("early", "Early training", step, metrics)
            if step == args.steps:
                checkpoint("trained", "Trained", step, metrics)
            elapsed = previous_seconds + time.perf_counter() - started
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "generator": training_generator.get_state(), "step": step, "config": asdict(config),
                        "settings": settings, "corpus_sha256": corpus["sha256"], "seconds": elapsed,
                        "history": history, "checkpoints": checkpoints, "fixtures": fixtures}, args.run_dir / "training.pt")

    manifest = {
        "format": FORMAT, "config": asdict(config), "vocab": tokenizer.vocab,
        "parameter_count": model.parameter_count, "checkpoints": checkpoints, "history": history,
        "training": {**settings, "steps": args.steps, "device": "cpu", "seconds": round(elapsed, 3),
                     "torch_version": str(torch.__version__), "python_version": platform.python_version(),
                     "platform": platform.system(), "architecture": platform.machine(), "optimizer": "AdamW",
                     "betas": [0.9, 0.99], "weight_decay": 0.1, "gradient_clip_norm": 1.0,
                     "evaluation": "Mean cross-entropy (natural-log nats per character) on fixed seeded batches",
                     "eval_tokens": args.eval_batches * args.batch_size * config.block_size,
                     "training_token_presentations": args.steps * args.batch_size * config.block_size,
                     "timing_note": "Measured wall time includes evaluation and JSON export; excludes setup/dependency installation. Resumed runs sum active process time."},
        "corpus": corpus,
    }
    write_json(args.output / "manifest.json", manifest)
    write_json(args.fixtures, {"format": FORMAT, "cases": fixtures})
    print(f"Exported {model.parameter_count:,} parameters; {elapsed:.1f}s; validation loss {metrics['validation_loss']:.4f}", flush=True)


if __name__ == "__main__":
    main()
