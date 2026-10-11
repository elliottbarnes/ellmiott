# Architecture and numerical agreement

## The model in one path

```text
characters → IDs → token embeddings + position embeddings
  → block 0 → block 1 → final LayerNorm → vocabulary logits
  → temperature / top-k → sample one character → append and repeat

block:
  x → LayerNorm → causal multi-head attention → add x
  u → LayerNorm → linear → GELU → linear → add u
```

The shipped configuration has two decoder blocks, four heads per block, 64 embedding channels, and a context window of 64 characters. Each head has 16 channels. All linear layers include bias except the final vocabulary projection. Token embeddings and output weights are separate parameters. LayerNorm uses `ε = 1e-5`. Both runtimes use the tanh approximation of GELU. There is no encoder, retrieval system, instruction fine-tuning, or external language-model API.

## Tensor conventions

`B` is batch size, `T` is context length, `C` is channel width, `H` is head count, `D = C/H`, and `V` is vocabulary size. A linear layer stores weights as `[output, input]`, following PyTorch. JSON arrays are flattened in row-major order: the last dimension changes fastest.

| Operation | Shape |
| --- | --- |
| Input IDs | `[B, T]` |
| Embedding sum | `[B, T, C]` |
| Combined query/key/value projection | `[B, T, 3C]` |
| One projection after head split | `[B, H, T, D]` |
| Attention scores and weights | `[B, H, T, T]` |
| Concatenated head outputs | `[B, T, C]` |
| MLP expansion | `[B, T, 4C]` |
| Output logits | `[B, T, V]` |

For a layer index `i`, tensor names follow:

```text
token_embedding.weight
position_embedding.weight
blocks.i.ln1.weight / bias
blocks.i.attn.qkv.weight / bias
blocks.i.attn.proj.weight / bias
blocks.i.ln2.weight / bias
blocks.i.mlp.fc.weight / bias
blocks.i.mlp.proj.weight / bias
ln_f.weight / bias
lm_head.weight
```

The checkpoint includes `format`, `config`, `vocab`, `step`, and a `tensors` object. Every tensor contains its shape and finite numeric data. Checkpoints contain learned parameters, not prewritten answers. The [manifest](../site/models/manifest.json) records file size and SHA-256 digest for each checkpoint.

## Two implementations of the same forward pass

[The Python model](../ellmiott_gpt/model.py) trains with PyTorch. [The browser engine](../site/engine.js) implements inference over the exported arrays. [The worker](../site/worker.js) runs that computation off the main page thread so the interface can remain responsive. The browser fetches static weights and computes locally; inference does not call a hosted model.

[Parity fixtures](../tests/fixtures/parity.json) contain outputs produced by Python for specified prompts, checkpoints, layers, heads, and ablations. JavaScript tests compare full final-position logits within an absolute tolerance of `2e-4` and selected attention matrices within `2e-5`. The browser uses Float32 weight storage and JavaScript number accumulation. This is a stronger implementation check than comparing sampled passages, because sampling can conceal a small numerical mismatch or diverge due to different random generators. The browser uses Mulberry32; Python uses a seeded PyTorch generator. Equal seeds do not promise equal passages across the two implementations.

Agreement on fixture cases does not prove all possible inputs correct. Causal-mask, shape, tokenizer, sampling, and invalid-input tests provide additional coverage. The code favors inspectability over speed: it recomputes the retained context on each generated token and has no GPU kernels or key/value cache.

## Head ablation

The disabled head's weighted value output is set to zero before head concatenation and the output projection. Its attention weights are still valid calculated quantities and may still be inspected. Other heads, the projection bias, residual stream, and MLP remain active. In a later layer, ablation of an earlier layer can also alter the displayed attention matrix because its input representations change.

## Extension ideas

Good next experiments include a larger context window, tied embeddings, a byte tokenizer, a key/value cache, or a stronger validation protocol. Each changes a different part of the system. Preserve the current baseline, explain the change, and compare measured behavior before making a performance claim.
