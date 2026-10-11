# Training corpus

`tinyshakespeare.txt` is the unmodified Tiny Shakespeare text distributed in
Andrej Karpathy's `char-rnn` repository:

<https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt>

The underlying works are by William Shakespeare and are public domain. The
selection/formatting is attributed to Andrej Karpathy. No separate license for
that compilation is asserted here. The project source-code MIT license does not
change the provenance or rights of this corpus.

Training uses the first 90% of characters, and validation uses the final 10%.
Windows never cross the boundary. The character vocabulary is derived from the
complete file; this exposes which characters exist, but validation token order
never contributes to optimization. Thus the validation set is held out from
training, but is not a new-author or new-domain evaluation.

The source URL, byte SHA-256, character counts, exact split, and model vocabulary
are recorded in `site/models/manifest.json`. Validation losses in the training
chart use fixed seeded batches from the validation split, rather than every
possible validation window. Overlapping windows are possible. The bigram CLI
reports every adjacent validation pair and therefore uses a different evaluation
sample; compare the methods with that qualification.

The preserved UTF-8 corpus is 1,115,394 bytes and characters. Its SHA-256 is
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.

The corpus is included so ordinary training does not make any network requests.
No personal messages, private documents, or user prompts were used for training.
