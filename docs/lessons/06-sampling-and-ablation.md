# 6 · Sampling and intervention

[Previous: training](05-training.md) · [Course index](../README.md)

**Goal:** separate learned predictions from sampling choices, then make a controlled intervention in the model.

## A distribution is not a sentence

The model returns logits `z`. Temperature `τ > 0` changes the sampling distribution:

$$p_i = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}.$$

Lower temperature concentrates probability around high-scoring tokens; higher temperature spreads it more broadly. It changes how we sample from the existing model. It does not train better weights or add knowledge. The implementations subtract the largest logit before scaling and exponentiating to avoid overflow; this leaves probabilities unchanged. The Python sampler and inference engine also accept temperature zero as a special case: choose the highest-scoring token directly, rather than divide by zero.

**Top-k** keeps only the `k` highest logits before normalizing. With `k = 1`, every draw chooses the highest-scoring token (with a defined tie order). In the lab, `k = 0` means no top-k cutoff. The adjusted probability display follows these settings; the model's base distribution is the unmodified softmax.

Run:

```sh
python docs/experiments.py sampling
```

The experiment prints probability distributions and entropy for fixed logits at three temperatures. Entropy `−Σp log p` measures how spread out a distribution is. It also samples twice with the same seed and verifies that the sequences match. Edit the logits to create a tie; inspect how top-k and deterministic tie handling interact.

## Autoregressive generation

Generation repeatedly performs four steps: retain the last 64 supported input characters, compute logits for the final position, sample a character, and append it. Newly sampled characters become context for subsequent predictions. The current engine recomputes that window for clarity; it does not use a key/value cache. Longer output therefore takes more forward passes.

A fixed seed makes a comparison repeatable within an implementation. Python and JavaScript can use different random generators, so equal seeds alone do not guarantee identical sampled text across runtimes. This project tests agreement of logits and attention directly instead of using sample coincidence as its numerical correctness test.

## Disable one head

Select a layer and head in the lab, inspect a prompt, and record its next-token probabilities. Disable that head and inspect the **same prompt** again. The intervention zeros the head's value-weighted output before the heads are concatenated and projected. Its attention matrix may still be displayed because computing attention and using its output are separate steps.

Record the biggest probability changes. Restore the head and verify that the original distribution returns. Then repeat on several prompts. Generation is a useful follow-up, but once one character differs, later predictions see different contexts; that makes it harder to isolate the direct effect of the head.

An ablation can establish that a component affects a calculation under specified conditions. It does not prove a complete linguistic role for that component. Other heads and residual paths may compensate, and a tiny model may have learned fragile patterns.

## Exercises

1. Does changing temperature alter the causal attention heatmap for a fixed prompt and checkpoint?
2. Why should you compare probabilities before comparing full generated passages in a head-ablation experiment?
3. What does it mean if disabling a head barely changes one prompt's prediction?

<details>
<summary>Worked answers</summary>

1. No. Temperature is applied after the forward pass. During generation it can change the chosen next character, which changes the context and therefore future attention matrices.
2. The fixed-prompt comparison changes one internal operation while preserving the input. A different sampled character causes the rest of the two generation paths to diverge for additional reasons.
3. Only that its effect was small for that measurement. Test other prompts, layers, and quantitative differences before proposing that the head is redundant. A component can matter only in particular contexts.

</details>

**Checkpoint:** you can reproduce a controlled comparison and distinguish a model change, a sampling change, and a change in input context.
