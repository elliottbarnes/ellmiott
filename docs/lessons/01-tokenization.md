# 1 · Characters and a first predictor

[Course index](../README.md) · [Next: embeddings](02-embeddings.md)

**Goal:** explain the training example hidden inside a piece of text and build a useful baseline without a neural network.

## Text becomes a sequence

This project gives every distinct corpus character a fixed integer ID. A token is one character, including spaces, punctuation, and newlines. IDs identify symbols; the numerical distance between two IDs has no linguistic meaning. A tokenizer with `a = 0` and `z = 25` is not saying that `z` is twenty-five times more meaningful.

For a toy vocabulary `[' ', 'a', 'b']`, `"aba "` becomes `[1, 2, 1, 0]`. Decoding reverses that lookup. The published vocabulary is stored inside every checkpoint so the browser and Python use the same mapping. The browser reports unsupported characters and replaces them with `?` when available, otherwise a space. That replacement loses information; it is not a new learned token.

The next-character task shifts a sequence by one position:

```text
text:     h e l l o
input:    h e l l
target:   e l l o
```

There are four training predictions here, not one. Using zero-based indices, position 2 receives the prefix `hel` and is asked to predict the second `l`. The causal mask in lesson 3 ensures it cannot read that answer from the input at position 3.

## A count-based bigram baseline

A bigram model uses just the immediately previous character. Count each transition `a → b`, then normalize the row:

$$P(b\mid a)=\frac{N(a,b)+\alpha}{\sum_j N(a,j)+\alpha V}.$$

`N(a,b)` is how often the pair occurred, `V` is the vocabulary size, and `α` is an added count. Positive `α` gives unseen transitions a nonzero probability. Setting `α` extremely large makes every row approach a uniform distribution. This is **smoothing**, not training a hidden layer.

Run:

```sh
python docs/experiments.py tokens
```

The experiment prints the vocabulary, a round trip, shifted pairs, the next-character distribution after `a`, and the baseline's average loss. Change its `text` string and smoothing value. Before running, predict which transition should become most likely.

## Try it in the lab

Open [Ellmiott GPT](https://elliottbarnes.github.io/ellmiott-gpt/), type a short prompt, and inspect its tokens. Add a space, newline, then an emoji. Which changes add a token? Which trigger the unsupported-character notice? Try a prompt longer than the context window and inspect which suffix is retained. The model can use only that visible suffix.

## Exercises

1. How many adjacent training pairs are in a string of `n` characters?
2. Why must counts come from the training split rather than the whole dataset?
3. Give two contexts where a bigram model must make the same prediction although a longer-context model could distinguish them.

<details>
<summary>Worked answers</summary>

1. `n - 1`, assuming one continuous sequence with no extra boundary token. Four characters produce three adjacent pairs.
2. Using validation transitions to construct counts lets the baseline learn from the data used to judge it. This is leakage, even though no gradient descent is involved.
3. `"red ca"` and `"a cup of tea"` both end in `a`. A bigram predictor sees the same final character in both. A transformer can assign different distributions because earlier context differs. That ability does not guarantee this small trained model will do so usefully.

</details>

**Checkpoint:** you can encode a string, construct shifted targets, and explain why a bigram model is a meaningful but limited baseline.
