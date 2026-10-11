# 2 · Embeddings and positions

[Previous: tokenization](01-tokenization.md) · [Course index](../README.md) · [Next: attention](03-attention.md)

**Goal:** trace one token ID into the vector that enters the first transformer block.

## A learned lookup table

An embedding table has shape `[V, C]`: one row for each vocabulary item and `C` numerical channels per row. Looking up token `t` returns row `E[t]`. The shipped model uses 64 channels. The lookup is equivalent to multiplying a one-hot vector by the table, but the implementation avoids allocating that mostly-zero vector.

Embedding values start random and change during training. A channel does not come with a human label such as “royalty” or “past tense.” A colorful embedding strip is a view of numbers, not a dictionary of meanings.

The same character uses the same token-embedding row wherever it occurs. To expose order, the model adds a second learned lookup table indexed by position:

$$x_i = E[t_i] + P[i].$$

Here `i` is a position within the current context window, `tᵢ` is its character ID, and both vectors have shape `[C]`. Their sum also has shape `[C]`. This model has a finite learned table of 64 positions. When generation moves beyond the window, the retained suffix is assigned positions 0 through 63 again.

## Follow the shapes

For a batch of `B` examples, each containing `T` tokens:

```text
token IDs                  [B, T]
token embedding lookup     [B, T, C]
position embedding lookup     [T, C]
broadcast sum              [B, T, C]
```

**Broadcasting** reuses the same position vectors for every example in the batch. It does not combine different examples. The two axes `T` and `C` mean different things: one moves along the text, the other across a token's features.

Run:

```sh
python docs/experiments.py embeddings
```

The input contains repeated characters. The experiment checks that their token embeddings match, while their token-plus-position vectors differ. Edit `channels` and `text`; predict the output shape before running. Set the position table to zero and repeat the comparison.

## Try it in the lab

Inspect `"aaaa"` and compare the embedding strips for repeated tokens. These strips show only the first eight channels of the token embedding, before position embeddings are added. Repeated characters therefore display the same values in the same checkpoint. The position sum is visible in the local experiment above; it is not shown in the browser strip. Switch checkpoints to observe how learned token values change. A different color alone does not establish that an embedding improved.

## Exercises

1. A vocabulary has 65 characters and the model has 64 channels. How many token-embedding parameters are there?
2. How many additional parameters does a learned table of 64 positions add?
3. Would a wider embedding necessarily improve a fixed training run?

<details>
<summary>Worked answers</summary>

1. `65 × 64 = 4,160`. It is a table with no separate bias.
2. `64 × 64 = 4,096`. Together, these two tables contain 8,256 parameters. Check the manifest vocabulary size before applying these numbers to a different corpus.
3. No. More channels increase capacity and computation, but the data, optimizer, number of updates, and risk of overfitting still matter. Compare runs using the same validation split and record their costs.

</details>

**Checkpoint:** you can distinguish a token ID, a token embedding, a position embedding, and their sum.
