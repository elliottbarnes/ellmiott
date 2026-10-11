# Ellmiott

**Build a language model. See every prediction.**

A small GPT built for learning: readable PyTorch training code, a separate JavaScript inference engine, and an interactive lab that runs the actual trained weights in your browser. Follow a character through tokenization, embeddings, causal attention, and the next-token distribution. Change sampling settings, switch training checkpoints, or disable an attention head and investigate the effect.

[Open the interactive lab](https://elliottbarnes.github.io/ellmiott/) · [Start the six lessons](docs/README.md) · [Model card](MODEL_CARD.md) · [Reproduce training](docs/training.md)

## What is here

- **A complete small decoder-only transformer:** two blocks, four attention heads per block, 64 channels, a 64-character context, and 112,512 learned parameters.
- **Three real checkpoints:** untrained, early training, and trained. Their weights and measured losses are committed with hashes and data provenance.
- **An interactive model inspection lab:** tokens, input embedding strips, causal attention matrices, next-token probabilities, temperature, top-k, a seeded generator, head ablation, and the recorded learning curve.
- **Six guided lessons:** equations explained through runnable experiments, exercises, worked answers, and a capstone based on a testable claim.
- **Two comparable forward passes:** Python exports numerical reference cases; JavaScript tests check logits and attention from the same rounded weights.
- **A count-based bigram baseline:** establish what one previous character can predict before adding a transformer.

The checkpoint is a small character-level text-completion model trained on Tiny Shakespeare. It produces fragments, invented words, and inconsistent prose. It has no instruction or conversation training. Its educational value is in the visible calculations and reproducible experiments; see the [model card](MODEL_CARD.md) for the scope of the measurements.

## Run the browser lab locally

Use Node.js 22 or newer. The site has no runtime packages to install.

```sh
git clone https://github.com/elliottbarnes/ellmiott.git
cd ellmiott
npm test
npm run build
npm run dev
```

Open the local URL printed by the development server. A server is needed because the browser loads JavaScript modules, a worker, and checkpoint files. The static production site is written to `dist/`; GitHub Actions builds and publishes that directory to GitHub Pages.

The page downloads a checkpoint when selected, then computes generation locally in a Web Worker. There are no model API keys, analytics, remote fonts, or paid services. Prompts are passed to the local worker, not sent to an inference server. Ordinary static-file requests still go to the site's hosting provider.

## Run the Python lessons and model

Use Python 3.11 or 3.12 and create a virtual environment, which keeps this project's packages separate from other Python projects:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python docs/experiments.py all
python -m unittest discover -s tests -p 'test_*.py'
python -m ellmiott_gpt.bigram --count 200 --seed 42
python -m ellmiott_gpt.sample --prompt 'ROMEO:' --count 200 --seed 42
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell instead. PyTorch provides tensor operations, automatic differentiation, and AdamW; this project explicitly implements the transformer forward pass. Training uses the CPU and the bundled corpus, with no corpus download required.

Start with the [small training smoke run](docs/training.md#first-check-the-training-path), then reproduce the published configuration. The committed JSON checkpoints let you use the browser or Python sampler without training first.

## A first experiment

1. Load the **untrained** checkpoint and inspect `ROMEO:`. Record the highest next-character probability.
2. Switch to **trained**, preserving the prompt. Compare the distribution and inspect a head's attention matrix.
3. Disable that head. Record the probability changes before generating a passage, so the input stays identical.
4. Restore the head, then compare temperature 0.5 and 1.5 with the same seed. Temperature changes sampling; it does not change the model's learned weights.

The [course](docs/README.md) turns these observations into increasingly precise questions. Attention pictures are views of a calculation, not complete explanations of a model's reasoning.

## Find your way around

| Path | Purpose |
| --- | --- |
| [`ellmiott_gpt/model.py`](ellmiott_gpt/model.py) | Transformer layers, tensor shapes, and head intervention |
| [`ellmiott_gpt/train.py`](ellmiott_gpt/train.py) | Seeded CPU training, held-out evaluation, checkpoint/resume |
| [`ellmiott_gpt/export.py`](ellmiott_gpt/export.py) | Portable weights and numerical parity fixtures |
| [`ellmiott_gpt/bigram.py`](ellmiott_gpt/bigram.py) | Smoothed count-based baseline |
| [`site/engine.js`](site/engine.js) | Browser inference and sampling calculations |
| [`site/worker.js`](site/worker.js) | Background computation and streamed generation |
| [`site/models/manifest.json`](site/models/manifest.json) | Actual run settings, learning curve, file hashes, corpus split |
| [`docs/`](docs/README.md) | Six lessons, editable experiments, architecture, reproduction |
| [`tests/`](tests/) | Python behavior tests and JavaScript numerical checks |

See [architecture](docs/architecture.md) for the tensor layout and runtime agreement. See [citations](CITATIONS.md) for the research and educational references.

## License and attribution

Original project code, documentation, and generated checkpoints are provided under the [MIT license](LICENSE). The bundled Shakespeare text is separate third-party source material: the underlying works are public domain, and the Tiny Shakespeare compilation is attributed to Andrej Karpathy. See [data provenance](data/README.md) and [citations](CITATIONS.md); the project license does not relicense third-party material.

The transformer architecture and GPT training objective build on established research. minGPT and nanoGPT are credited educational references. Ellmiott's teaching interface, runnable experiments, explicit model implementation, and browser runtime form this project's implementation.
