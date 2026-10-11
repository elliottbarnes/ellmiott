# 3 · Causal attention

[Previous: embeddings](02-embeddings.md) · [Course index](../README.md) · [Next: transformer blocks](04-transformer-block.md)

**Goal:** compute a tiny attention matrix and explain why its upper triangle must be zero.

## Queries, keys, and values

Each head projects the current token representations into three arrays: queries `Q`, keys `K`, and values `V`. A query at one position is compared with keys at allowed positions. The resulting weights mix the value vectors. These are learned linear projections, not literal searches through a database.

For one head:

$$S = \frac{QK^\top}{\sqrt{d}},\qquad A = \operatorname{softmax}(S+M),\qquad O = AV.$$

`d` is the head's channel count. The shipped model divides 64 channels among four heads, so `d = 16`. Dividing by `√d` controls the score scale. Softmax operates across the keys in each query row. Every row of `A` is nonnegative and sums to one.

The mask `M[i,j]` is zero when `j ≤ i` and negative infinity when `j > i`. Softmax turns masked entries into zero. A token can attend to itself because its own character is part of the input; its target is the **next** character.

```text
            key positions
              0  1  2  3
query 0       ✓  ·  ·  ·
query 1       ✓  ✓  ·  ·
query 2       ✓  ✓  ✓  ·
query 3       ✓  ✓  ✓  ✓
```

Without this restriction, training could read future input characters containing the answers. That would make training loss look impressive while teaching an unusable generation procedure.

## Test causality by changing the future

Run:

```sh
python docs/experiments.py attention
```

The experiment computes a four-token head, checks row sums and the mask, then modifies the final input vector. With causal attention, outputs at earlier positions remain unchanged. The unmasked comparison does change earlier outputs. Edit the input length or perturb a different position, then predict exactly which output rows are allowed to change.

Multiple heads run these operations with different projections. Their outputs are concatenated and projected back to 64 channels. Four heads do not mean four named linguistic roles; any interpretation needs evidence.

## Try it in the lab

Use `"To be, or not to be"`. Select a layer and head, then inspect a row of the heatmap. Check that future columns have zero weight. Compare a different head on the same checkpoint and prompt. Treat the picture as a record of a calculation: attention weight alone is not a complete explanation of a prediction because values, residual paths, later layers, and the output projection also matter.

## Exercises

1. What should the first attention row be for a nonempty prompt?
2. Does a high weight on a space prove that the head understands word boundaries?
3. Why can training process all positions together, although generation adds one token at a time?

<details>
<summary>Worked answers</summary>

1. `[1, 0, …, 0]`: position zero has only itself available.
2. No. It shows a strong weight for that input under those settings. To investigate a hypothesis, compare many examples, alternative explanations, and interventions. One attractive heatmap is insufficient.
3. Training already has the full text and shifted targets. The mask independently restricts each row to its prefix. During generation, the next input character does not exist until it has been sampled, so future steps depend on previous results.

</details>

**Checkpoint:** you can identify the query and key axes, compute allowed positions, and design a test for accidental future-token leakage.
