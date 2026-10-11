# Learn by changing one thing

The course follows a character from the input string to the next-character probability distribution. Each lesson has a small experiment, a prediction to make before running it, and exercises with worked answers. The browser lab uses the actual exported model weights; the small Python experiments isolate the same mathematical ideas so you can edit them without retraining the full model.

Prerequisites: basic Python, lists, loops, and functions. A **tensor** is an array with a shape, such as `[tokens, channels]`. A **parameter** is a number adjusted during training. You do not need to know calculus to run the experiments; the lessons explain what the gradients are used for.

| Lesson | Question | Local experiment |
| --- | --- | --- |
| [1. Characters and a first predictor](lessons/01-tokenization.md) | What exactly does the model predict? | `python docs/experiments.py tokens` |
| [2. Embeddings and positions](lessons/02-embeddings.md) | How do characters become vectors? | `python docs/experiments.py embeddings` |
| [3. Causal attention](lessons/03-attention.md) | Which earlier positions can influence this prediction? | `python docs/experiments.py attention` |
| [4. A transformer block](lessons/04-transformer-block.md) | How does the network combine information? | `python docs/experiments.py block` |
| [5. Training and honest evaluation](lessons/05-training.md) | Is it learning patterns that transfer? | `python docs/experiments.py training` |
| [6. Sampling and intervention](lessons/06-sampling-and-ablation.md) | Why do the same weights produce different text? | `python docs/experiments.py sampling` |

Run the commands from the repository root with the Python environment in the [README](../README.md) activated. You can run all six with `python docs/experiments.py all`. The experiments use fixed seeds and include assertions for their central claims. Edit [experiments.py](experiments.py) to test a hypothesis, then explain the result before looking at the lesson's answer.

## A useful experiment log

Copy this table for each change. Keep the seed, prompt, model checkpoint, and every other setting fixed when comparing two conditions.

| Hypothesis | One changed setting | Observation | What this supports | What it does not prove |
| --- | --- | --- | --- | --- |
| Higher temperature spreads probability more widely | Temperature 0.5 → 1.5 | Record top probability and entropy | A change to sampling uncertainty | Better prose or better factual knowledge |

## Capstone: make a claim you can check

Choose one experiment: shorten context, change the smoothing of the bigram baseline, or disable one attention head. Write down a prediction. Collect a result on a fixed prompt set and at least one counterexample. Include your settings, evidence, and a limitation. If you train a new architecture, keep a separate output folder and report validation loss on the same split; a pleasing sample alone is weak evidence.

For model details, data provenance, and the shipped run, see the [model card](../MODEL_CARD.md). For tensor shapes and Python/browser agreement, see [architecture](architecture.md).
