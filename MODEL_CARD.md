# Ellmiott GPT · model card

## Purpose

Ellmiott GPT is an inspectable character-level language model for learning how a small decoder-only transformer works. Its intended uses are local experiments, teaching, numerical inspection, and reproducing a modest training run. It predicts text continuations in the style of its narrow corpus. It has no instruction tuning, conversation training, retrieval, external tools, or claim to factual reliability.

The browser and Python sampler both load the committed JSON weights. Checkpoints are real snapshots from random initialization and training. The browser does not substitute scripted responses or call a hosted language-model service.

## Architecture

| Property | Shipped configuration |
| --- | --- |
| Model type | Pre-norm, decoder-only causal transformer |
| Parameters | 112,512 |
| Vocabulary | 65 distinct characters |
| Context | Last 64 supported characters |
| Blocks / heads per block | 2 / 4 |
| Embedding / head width | 64 / 16 |
| Feedforward width | 256, with tanh-approximate GELU |
| Position encoding | Learned absolute embeddings within the retained window |
| Layer normalization epsilon | `1e-5` |
| Output projection | Untied, no bias |
| Dropout | None |

See [architecture](docs/architecture.md) and [model.py](ellmiott_gpt/model.py) for the exact operations. The 64-character limit excludes earlier text from each forward pass. Unsupported characters are rejected by the Python tokenizer; the browser visibly replaces them with `?` (or a space if that fallback is required by a different vocabulary). Empty browser input uses a newline.

## Data and split

Training uses [Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt), compiled by Andrej Karpathy from works by William Shakespeare. The underlying Shakespeare text is public domain; no separate license is asserted for the compilation. Its source and the project MIT license are documented separately in [data provenance](data/README.md) and [citations](CITATIONS.md).

The bundled file preserves the downloaded bytes:

```text
SHA-256: 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed
```

There are **1,115,394 characters**. The first **1,003,854** form the training split; the final **111,540** form the validation split. Windows are sampled within a split, never across the boundary. The character vocabulary is derived from the whole corpus, so the inventory of validation symbols is known; validation sequences are not used for gradient updates. There is no separate test split, no broad benchmark evaluation, and no claim that repeated phrases cannot appear across the split.

This is a single literary corpus with historical language, stereotypes, violence, and other themes from its source plays. Outputs can reproduce source fragments, invent words, or generate objectionable text. The dataset does not represent contemporary language use or a broad population of speakers.

## Training run

The shipped run used Python 3.12.14 and PyTorch 2.6.0 on a Darwin arm64 CPU with four PyTorch threads. It started with seed 42 and ran 4,000 updates, then resumed the saved optimizer and random-generator state to reach 8,000 total updates. The manifest records **61.902 seconds** of active measured wall time, including evaluation and export and excluding environment setup. That is one measurement on the recorded environment, not a runtime guarantee for another computer.

| Setting | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | Constant `0.001` |
| Betas | `(0.9, 0.99)` |
| Weight decay | `0.1` |
| Gradient clipping | Maximum global norm `1.0` |
| Batch size / sequence length | 16 / 64 |
| Updates | 8,000 |
| Training token presentations | 8,192,000, including repeat sampling |
| Evaluation interval | Every 100 updates, plus initialization |
| Evaluation per split | 16 fixed seeded batches; 16,384 target positions |

Training batches, training-evaluation batches, and validation batches use separate generators with seeds 43, 44, and 45 respectively. Evaluation uses the same sampled windows at each checkpoint; it does not advance the training generator. Measured losses use natural logarithms and are **nats per character**. The fixed validation sample supports comparisons along this run but is not every character in the validation split.

## Checkpoints and measured results

| Checkpoint | Updates | Validation cross-entropy | Character perplexity | JSON bytes |
| --- | ---: | ---: | ---: | ---: |
| Untrained | 0 | 4.1831 | 65.569 | 1,427,441 |
| Early | 100 | 2.6188 | 13.719 | 1,437,311 |
| Trained | 8,000 | 1.7833 | 5.950 | 1,391,307 |

Perplexity is `exp(cross-entropy)`. It is specific to the tokenization and evaluated data, so it is not directly comparable with word- or subword-level model perplexity. The final training-evaluation loss is **1.6057**. These measurements demonstrate improved next-character prediction on the sampled held-out text; they do not establish coherent long-form writing, factual understanding, or conversational ability.

The released trained checkpoint is the fixed 8,000-update endpoint. It is not selected as the minimum validation-loss checkpoint: the recorded 7,900-update measurement was lower. The full curve is retained rather than reporting only the most favorable point.

The count-based bigram baseline reports **2.4819** nats per character across all adjacent pairs in the held-out split. It uses the same corpus boundary, but its full-pair evaluation differs from the GPT's fixed sampled windows. Treat it as a baseline with that explicit measurement difference.

The authoritative machine-readable record is [manifest.json](site/models/manifest.json), including complete history, settings, source digest, and every checkpoint's digest. The trained checkpoint SHA-256 is:

```text
61c1869cf0d2bccbaff4884b7600100ecef046420cf27aeb2adba6f00035296f
```

## Inspection, intervention, and verification

The browser displays token-embedding values before the position sum, a selected causal-attention matrix, and a sampling distribution after temperature/top-k adjustment. A displayed heatmap is one intermediate calculation, not a complete explanation of the model's behavior.

Disabling a head zeros its weighted value output before projection. The interface can still display that head's calculated attention weights. Compare fixed-prompt distributions before generated text: after a different token is sampled, later steps have different contexts as well as the original intervention.

The test suites check model behavior and Python/JavaScript numerical agreement for initial and trained checkpoints, with ordinary and disabled-head cases. Export reference values are computed by reloading the same rounded JSON weights that browsers download. The JavaScript tests use absolute tolerances of `2e-4` for logits and `2e-5` for attention. Fixed test cases provide evidence, not an exhaustive proof of correctness on all prompts. Python and JavaScript use different random generators; matching seeds are not a promise of identical cross-runtime generated passages.

## Practical limitations

- Character-level tokens and a short context make long-range coherence difficult.
- The small architecture, narrow corpus, and short training run often produce nonsense, misspellings, repetition, and incomplete dialogue.
- Training and validation loss do not measure truthfulness, safety, fairness, or suitability for consequential decisions.
- Browser generation recomputes the context window for each token. Device speed affects latency; it has no GPU path or key/value cache.
- This version has no instruction fine-tuning, preference optimization, moderation model, multilingual evaluation, or formal memorization audit.

The project is designed for inspection and experimentation. [Reproduce the run](docs/training.md), keep any changed configuration explicit, and update this card when publishing new weights. Original code, documentation, and generated weights are offered under the [MIT license](LICENSE); third-party corpus attribution is separate.
