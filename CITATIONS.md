# Sources and attribution

Ellmiott GPT is an educational implementation of established transformer ideas. These sources credit the architecture, training objective, and reference projects; they are not claims of affiliation or endorsement.

## Research

- Ashish Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762). The source of the transformer architecture and scaled dot-product, multi-head attention. This project uses causal decoder attention rather than reproducing the paper's full translation system.
- Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever (2018), [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). A primary reference for GPT's decoder-based language-model pretraining. Ellmiott GPT trains a much smaller character-level model and does not reproduce that paper's downstream fine-tuning or benchmark results.

## Educational implementations

- Andrej Karpathy, [minGPT](https://github.com/karpathy/minGPT): an educational PyTorch implementation of GPT, credited as a reference for keeping the model readable.
- Andrej Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT): a reference for compact language-model training workflows and small Shakespeare experiments.

The original source in this repository implements the explicit forward pass, exports portable weights, and provides a separate browser inference engine and teaching interface. PyTorch provides tensor operations, automatic differentiation, neural-network primitives, and the optimizer. The project does not implement its own automatic-differentiation system.

## Training corpus

The bundled `data/tinyshakespeare.txt` preserves the bytes of [Tiny Shakespeare in Karpathy's char-rnn repository](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt). The underlying text is by William Shakespeare; the compilation is attributed to Andrej Karpathy. The original plays are public-domain works. No separate MIT license is asserted for this compilation, and the repository's own MIT license does not relicense the corpus.

See [data/README.md](data/README.md) for the source URL and digest, and [MODEL_CARD.md](MODEL_CARD.md) for how the text was split and evaluated. Generated text may reproduce fragments of the training material.
