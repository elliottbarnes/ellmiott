# 5 · Training and honest evaluation

[Previous: transformer blocks](04-transformer-block.md) · [Course index](../README.md) · [Next: sampling and ablation](06-sampling-and-ablation.md)

**Goal:** connect the next-character objective to an optimizer step, then interpret a real learning curve without overstating it.

## What the loss measures

For each position, the model assigns a probability to the true next character. Cross-entropy penalizes a low probability for that target. Over `N` target positions:

$$L=-\frac{1}{N}\sum_{i=1}^{N}\log p_i(\text{target}_i).$$

The logarithm is natural, so this loss is measured in nats per character. Predicting the true target with probability `0.5` contributes about `0.693`; probability `0.1` contributes about `2.303`. Confident wrong predictions are expensive. PyTorch's cross-entropy accepts raw logits and performs the numerically stable log-softmax internally.

**Perplexity** is `exp(L)`. It summarizes average predictive uncertainty on the evaluated data. Character perplexity cannot be compared directly with word- or subword-token perplexity: the units and vocabularies differ.

## One update

```python
optimizer.zero_grad(set_to_none=True)
logits, _ = model(inputs)
loss = cross_entropy(logits.flatten(0, 1), targets.flatten())
loss.backward()
optimizer.step()
```

This is an abbreviated training loop; the complete implementation is in [train.py](../../ellmiott_gpt/train.py). `backward()` computes how each parameter affects loss. The optimizer uses those gradients to adjust parameters. Clearing gradients avoids unintentionally adding this batch's gradients to the previous batch's. Evaluating without gradients saves work and leaves parameters unchanged.

Run:

```sh
python docs/experiments.py training
```

The isolated experiment fits a neural bigram table to a repeated toy sequence and prints loss before and after optimization. It checks that an update actually changes weights. Edit the learning rate or the number of steps. This small demonstration is not the published GPT training run.

## Split before sampling windows

The corpus is divided into contiguous training and validation sections. Training batches are drawn only from the training section; validation batches only from the held-out section. No sampled window crosses that boundary. The vocabulary is an inventory of symbols, while transition frequencies and model weights must be learned from training text.

Randomly distributing overlapping windows across splits could place nearly identical passages in both sets. A contiguous split avoids that particular leak, although a single book-like corpus still has repeated phrases, genre overlap, and limited diversity. This project's validation curve is evidence about this corpus, not a general language benchmark. There is no separate held-out test benchmark.

## Read the actual run

Use the [reproduction guide](../training.md) to train locally. The browser's learning chart is loaded from the measured [manifest](../../site/models/manifest.json), and the [model card](../../MODEL_CARD.md) records the shipped run. Compare training and validation curves:

- Both falling suggests progress on the measured objective.
- Training falling while validation rises can indicate overfitting.
- A large spike can indicate an unstable update, unusual batch, or bug; it is a reason to investigate.

The displayed losses are averages over sampled evaluation windows. They can fluctuate. Do not infer a meaningful difference from the last decimal place, or compare runs with changed splits as if only the architecture changed.

## Exercises

1. What is the expected loss of a uniform predictor over a vocabulary of size `V`?
2. If training loss improves but a generated sample looks worse, which result is wrong?
3. What must you preserve for a fair comparison between two training runs?

<details>
<summary>Worked answers</summary>

1. `log(V)`, with perplexity `V`.
2. Neither necessarily. Loss aggregates many predictions, while a sample depends on the prompt and random choices. Compare validation loss and a fixed collection of seeded samples; neither is a complete quality measure alone.
3. At minimum, the corpus bytes, split, tokenizer, evaluation procedure, and declared compute/training budget. Record seed and hyperparameters. Changing only one factor makes your conclusion easier to defend, but repeated seeds are needed for stronger evidence than one run.

</details>

**Checkpoint:** you can describe what was optimized, what was held out, and exactly how far the measurements support a claim.
