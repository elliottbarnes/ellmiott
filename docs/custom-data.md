# Train your own Ellmiott

## In the browser

Open [Train your own](https://elliottbarnes.github.io/ellmiott/#train-your-own), choose a UTF-8 `.txt` file or paste text, and inspect the vocabulary. Start with **Use practice text** to learn the controls. Select 100, 500, or 2,000 updates and press **Train my model**. Training is genuine backpropagation through a causal transformer; the displayed losses are measured, not simulated.

The browser creates new random weights for each run. It uses one layer, four heads, 32 embedding channels, a 32-character context, batch size 4, and Adam at 0.003. TensorFlow.js 4.22.0 supplies CPU automatic differentiation in a dedicated worker. It is loaded from this site, without a third-party CDN. Inference uses Ellmiott's separate plain-JavaScript engine.

Use 400–200,000 characters with 2–256 distinct Unicode characters. File inputs are limited to 1 MB. The first 90% of characters train the model; the final 10% are held out. The loss readings use fixed batches from those separate splits and a separate sampling seed. Repetitions spanning both splits can make validation unrealistically easy; low loss on the practice text is not evidence of general language ability.

**Stop & keep weights** finishes the current update and exports that stage. **Use my model** loads it into the existing token, probability, attention, head-ablation, and generation tools. The main loss chart changes to the custom run. **Your model** returns to that checkpoint after comparing the published models.

**Download weights** saves a portable `ellmiott-custom.json` file with its vocabulary, weights, settings, and measured loss history. Reopen it using the JSON file picker. Python-exported `trained.json` files are also accepted, though their separate manifest/history is not imported. Imports are limited to 10 MB and validated before use.

No corpus, prompt, or weights are uploaded. Training consumes local CPU, may take several minutes, and stops when the page closes; background tabs may be slower. Nothing is saved automatically. Downloads contain learned weights rather than a copy of the corpus, but weights can memorize sensitive text. Keep private-data models private. Browser exports contain inference weights, not the optimizer state needed to resume the same training run.

## A longer local Python run

Install the environment using the [README](../README.md). Put a UTF-8 text file outside the public `site/` directory. The Python model uses a larger 64-character context, so it requires at least 650 characters and 2–256 distinct characters.

From the repository root:

```sh
python -m ellmiott_gpt.train \
  --data /path/to/your-text.txt \
  --steps 4000 --early-step 100 \
  --output runs/custom/models \
  --run-dir runs/custom \
  --fixtures runs/custom/parity.json
```

All generated files go under ignored `runs/`; the published demo weights stay intact. Custom datasets receive generic provenance instead of Shakespeare attribution. The manifest records the corpus SHA-256 and actual split sizes without retaining its filename or contents. Export checks use prompts supported by your vocabulary.

Open `runs/custom/models/trained.json` with the lab's model picker, or generate in Python:

```sh
python -m ellmiott_gpt.sample \
  --checkpoint runs/custom/models/trained.json \
  --prompt 'a prompt using characters from your text' \
  --count 120
```

Omit `--prompt` to start with a supported character. See the [training guide](training.md) for resuming Python optimizer checkpoints; keep the same data and settings and use your custom paths.

This trains a tiny character-level language model from scratch. It does not fine-tune ChatGPT, retrieve documents, or turn a text collection into a dependable question-answering assistant.
