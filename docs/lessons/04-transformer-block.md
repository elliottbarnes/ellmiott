# 4 · Building a transformer block

[Previous: attention](03-attention.md) · [Course index](../README.md) · [Next: training](05-training.md)

**Goal:** explain every step between the input embeddings and output logits.

## Two updates to the representation

This model uses a **pre-norm** decoder block: normalize the input before each transformation, then add the transformation's result back to the original stream.

$$u = x + \operatorname{Attention}(\operatorname{LN}_1(x))$$
$$y = u + \operatorname{MLP}(\operatorname{LN}_2(u)).$$

The additions are residual connections. They let a branch learn a change to an existing representation. Both sides of each addition must have identical shapes `[B, T, C]`. If the branch outputs zero, the residual path passes its input through unchanged.

**Layer normalization** treats one token at a time. It subtracts that token vector's channel mean and divides by the square root of its channel variance plus `ε`. Then learned scale and bias parameters adjust the result:

$$\operatorname{LN}(x)=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta.$$

Here `ε = 10⁻⁵`; it prevents division by zero for a constant vector. Normalization does not average across time, so it introduces no access to future tokens.

## The feedforward branch

The MLP applies the same transformation separately to every position:

```text
64 channels → linear → 256 channels → GELU → linear → 64 channels
```

Attention mixes information between positions. The MLP transforms each position's accumulated features. GELU is a smooth nonlinear activation. Without nonlinear operations, stacked linear layers could collapse into a single linear map. The Python model and JavaScript engine use the same tanh approximation of GELU to make their numerical outputs comparable.

Run:

```sh
python docs/experiments.py block
```

The experiment applies a small normalized MLP residual branch, prints shapes and channel statistics, and checks that zeroing the branch preserves its input. Change the hidden width. Then replace GELU with the identity and consider which expressive behavior disappears.

## From hidden states to predictions

After two blocks, a final layer normalization feeds a linear output head with shape `[V, C]`. This produces `V` **logits** per position: unrestricted scores, not probabilities. Softmax converts those scores into a distribution. The final input position's logits predict the next character.

The model uses an untied output matrix: it is a separate parameter from the token embedding table. This is a design choice, not a requirement of all GPT models. See the [architecture reference](../architecture.md) for the exact tensor contract.

## Exercises

1. Why must the MLP project back to 64 channels before the residual addition?
2. Does a zero attention branch make the whole transformer an identity function?
3. Why is the output vocabulary dimension unrelated to the prompt length?

<details>
<summary>Worked answers</summary>

1. Its output must align with the existing `[B, T, 64]` residual stream. The 256-channel intermediate representation is internal to the branch.
2. No. The MLP branch, the other transformer block, final normalization, and output projection can still change the result. Zeroing both branches makes that individual block an identity.
3. Each position must score every possible next character. A longer prompt adds positions along `T`; it does not add new entries to the fixed vocabulary `V`.

</details>

**Checkpoint:** you can trace `[B, T, C]` through both branches and explain the distinct roles of attention, an MLP, residual additions, normalization, and the output head.
