# Train, inspect, and reproduce

Install the Python environment from the [README](../README.md), and run these commands from the repository root. The training path is CPU-only by default and reads the committed corpus. Set aside disk space for both the Python/PyTorch environment and local optimizer checkpoints; the portable browser weights are much smaller than a general-purpose language model.

## First check the training path

This short run exercises actual forward/backward passes, evaluation, and export. Separate output paths preserve the committed demo weights and parity fixtures.

```sh
python -m ellmiott_gpt.train \
  --steps 10 --early-step 1 --eval-every 5 --eval-batches 2 \
  --batch-size 4 --threads 2 \
  --output runs/smoke/models \
  --run-dir runs/smoke \
  --fixtures runs/smoke/parity.json
```

Ten updates check that the pipeline runs; they do not train a useful language model. Inspect `runs/smoke/models/manifest.json` for its settings and measured results. Sample its actual exported checkpoint with:

```sh
python -m ellmiott_gpt.sample \
  --checkpoint runs/smoke/models/trained.json \
  --prompt 'ROMEO:' --count 80 --seed 42
```

## Reproduce the published configuration

```sh
python -m ellmiott_gpt.train \
  --steps 8000 --batch-size 16 --eval-every 100 --eval-batches 16 \
  --early-step 100 --learning-rate 0.001 --seed 42 --threads 4 \
  --output runs/reproduction/models \
  --run-dir runs/reproduction \
  --fixtures runs/reproduction/parity.json
```

This uses two blocks, four heads, 64 embedding channels, 64-character windows, AdamW, and the same data split as the shipped model. The released run was performed in two segments: 4,000 updates, then a continuation to 8,000. Training uses a constant learning rate; evaluation has separate random generators, so evaluating does not consume the training batch sequence. The saved optimizer and training generator states allow continuation.

The [model card](../MODEL_CARD.md) records the measured run. Deterministic operations and a fixed seed aid reproducibility within the same software and hardware environment. Bit-for-bit equality across operating systems, PyTorch versions, or processors is not promised. Check the corpus hash, version information, configuration, and comparable loss before explaining a discrepancy.

## Continue a local run

For example, continue the reproduction above to 9,000 **total** updates:

```sh
python -m ellmiott_gpt.train \
  --resume runs/reproduction/training.pt --steps 9000 \
  --batch-size 16 --eval-every 100 --eval-batches 16 \
  --early-step 100 --learning-rate 0.001 --seed 42 --threads 4 \
  --output runs/reproduction/models \
  --run-dir runs/reproduction \
  --fixtures runs/reproduction/parity.json
```

Keep the training settings and corpus unchanged; the loader checks saved settings and the corpus digest. Local resumable state is written at evaluation intervals. If interrupted between saves, restart from the last saved step. The JSON inference checkpoint contains weights but not optimizer state; use `training.pt` to resume optimization.

## Sampling and controlled ablation

```sh
python -m ellmiott_gpt.sample --prompt 'ROMEO:' \
  --count 200 --temperature 0.8 --top-k 0 --seed 42

python -m ellmiott_gpt.sample --prompt 'ROMEO:' \
  --count 200 --temperature 0.8 --top-k 0 --seed 42 \
  --disable-head 0:1
```

`--disable-head 0:1` disables head 1 in layer 0; indices start at zero. `--temperature 0` selects the highest-scoring token deterministically. Use supported corpus characters in Python prompts: the Python tokenizer is strict and reports an unknown character. The browser makes unsupported-character replacement explicit for interactive use. For causal comparisons, inspect a fixed prompt's logits before comparing whole generated passages.

The count baseline uses training transitions with additive smoothing:

```sh
python -m ellmiott_gpt.bigram --count 200 --seed 42
```

Its reported validation loss covers every adjacent pair in the held-out split. The GPT loss averages fixed sampled windows. Both are character cross-entropies on the same split, but the evaluation positions differ, so avoid treating a tiny difference as a precise paired comparison.

## Publish a deliberately chosen new checkpoint

The default training output paths are `site/models`, `runs`, and `tests/fixtures/parity.json`. Running without the separate paths shown above replaces the committed browser checkpoint files and reference fixtures in your working tree. Make that an intentional model update, inspect the diff, update the model card, and run:

```sh
python -m unittest discover -s tests -p 'test_*.py'
python docs/experiments.py all
npm test
npm run build
```

Commit the JSON weights, manifest, and matching parity fixtures together. Keep temporary optimizer checkpoints under ignored `runs/`. A new published metric should be traceable to a checkpoint and corpus hash, not copied from a previous experiment.
