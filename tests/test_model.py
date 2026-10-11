import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from ellmiott_gpt.bigram import fit_bigram
from ellmiott_gpt.corpus import CharacterTokenizer, load_corpus, make_batch
from ellmiott_gpt.export import export_checkpoint, load_export, write_json
from ellmiott_gpt.model import GPT, GPTConfig
from ellmiott_gpt.sample import sample


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.config = GPTConfig(vocab_size=5, block_size=8, n_embd=16, n_head=4, n_layer=2)
        self.model = GPT(self.config).eval()
        self.tokenizer = CharacterTokenizer("\n ab?")

    def test_future_tokens_cannot_change_previous_predictions(self):
        left = torch.tensor([[0, 1, 2, 3, 4, 0]])
        right = torch.tensor([[0, 1, 2, 4, 1, 2]])
        a, weights = self.model(left, return_attention=True)
        b, _ = self.model(right)
        torch.testing.assert_close(a[:, :3], b[:, :3], atol=1e-7, rtol=0)
        for layer in weights:
            self.assertEqual(torch.triu(layer, diagonal=1).count_nonzero().item(), 0)
            torch.testing.assert_close(layer.sum(-1), torch.ones_like(layer.sum(-1)))

    def test_head_ablation_changes_output_but_keeps_its_display_weights(self):
        ids = torch.tensor([[0, 1, 2, 3]])
        full, attention = self.model(ids, return_attention=True)
        ablated, disabled_attention = self.model(ids, disabled_head=(0, 1), return_attention=True)
        self.assertGreater((full - ablated).abs().max().item(), 1e-5)
        torch.testing.assert_close(attention[0], disabled_attention[0])
        with self.assertRaises(ValueError):
            self.model(ids, disabled_head=(2, 0))

    def test_gradient_step_learns_a_repeated_sequence(self):
        ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2]])
        targets = torch.tensor([[1, 2, 3, 0, 1, 2, 3]])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        losses = []
        for _ in range(20):
            logits, _ = self.model(ids)
            loss = F.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1))
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))
            optimizer.step()
        self.assertLess(losses[-1], losses[0] / 3)

    def test_export_roundtrip_preserves_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            metadata = export_checkpoint(self.model, self.tokenizer, path, step=12)
            restored, payload = load_export(path)
            ids = torch.tensor([[0, 1, 2, 3]])
            torch.testing.assert_close(self.model(ids)[0], restored(ids)[0], atol=1e-6, rtol=1e-5)
            self.assertEqual(payload["step"], 12)
            self.assertEqual(metadata["bytes"], path.stat().st_size)
            self.assertEqual(sum(len(spec["data"]) for spec in payload["tensors"].values()), self.model.parameter_count)
            self.assertFalse(any("mask" in name for name in payload["tensors"]))
            with self.assertRaises(ValueError):
                write_json(path, {"bad": float("nan")})

    def test_tokenizer_roundtrip_and_strict_unknown_character(self):
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode("a b?\n")), "a b?\n")
        with self.assertRaises(KeyError):
            self.tokenizer.encode("☃")

    def test_bigram_learns_local_transition_frequencies(self):
        probabilities = fit_bigram(torch.tensor([0, 1, 0, 1, 0, 1, 0, 2]), 3)
        self.assertGreater(probabilities[0, 1], probabilities[0, 2])
        self.assertGreater(probabilities[1, 0], probabilities[1, 1])
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(3))

    def test_batch_targets_are_shifted_without_crossing_split(self):
        data = torch.arange(100)
        x, y = make_batch(data, 16, 8, torch.Generator().manual_seed(3))
        torch.testing.assert_close(y, x + 1)
        self.assertLess(y.max(), len(data))
        with self.assertRaises(ValueError):
            make_batch(torch.arange(8), 1, 8, torch.Generator())

    def test_sampling_seed_and_context_window(self):
        prompt = "a b?\n" * 4
        a = sample(self.model, self.tokenizer, prompt, count=10, seed=42)
        b = sample(self.model, self.tokenizer, prompt, count=10, seed=42)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 10)
        self.assertEqual(sample(self.model, self.tokenizer, "", count=0), "")
        self.assertEqual(sample(self.model, self.tokenizer, "a", count=3, temperature=0),
                         sample(self.model, self.tokenizer, "a", count=3, temperature=5e-324))
        with self.assertRaises(ValueError):
            sample(self.model, self.tokenizer, "a", temperature=float("inf"))

    def test_invalid_dimensions_and_sequence_length(self):
        with self.assertRaises(ValueError):
            GPTConfig(n_embd=15, n_head=4)
        for ids in (torch.zeros((1, 0), dtype=torch.long), torch.zeros((1, 9), dtype=torch.long)):
            with self.assertRaises(ValueError):
                self.model(ids)


class PublishedArtifactTests(unittest.TestCase):
    def test_parity_fixtures_match_the_downloadable_weights(self):
        base = Path(__file__).resolve().parents[1]
        path = base / "tests/fixtures/parity.json"
        if not path.exists():
            self.skipTest("Run training to create parity fixtures")
        cases = json.loads(path.read_text())["cases"]
        self.assertEqual({case["checkpoint"] for case in cases}, {"initial", "trained"})
        self.assertTrue(any(case["disabledHead"] for case in cases))
        for case in cases:
            model, payload = load_export(base / "site/models" / f"{case['checkpoint']}.json")
            ids = CharacterTokenizer(payload["vocab"]).encode(case["prompt"])
            self.assertEqual(ids, case["ids"])
            disabled = case["disabledHead"]
            with torch.inference_mode():
                logits, attention = model(torch.tensor([ids]), return_attention=True,
                                          disabled_head=(disabled["layer"], disabled["head"]) if disabled else None)
            torch.testing.assert_close(logits[0, -1], torch.tensor(case["logits"]), atol=2e-5, rtol=1e-5)
            torch.testing.assert_close(attention[case["layer"]][0, case["head"]],
                                       torch.tensor(case["attention"]), atol=1e-6, rtol=1e-5)

    def test_corpus_and_checkpoint_manifest_match_actual_bytes(self):
        from hashlib import sha256
        base = Path(__file__).resolve().parents[1]
        manifest_path = base / "site/models/manifest.json"
        if not manifest_path.exists():
            self.skipTest("Run training to create publication artifacts")
        manifest = json.loads(manifest_path.read_text())
        train, validation, tokenizer, corpus = load_corpus(base / "data/tinyshakespeare.txt")
        self.assertEqual(corpus, manifest["corpus"])
        self.assertEqual(len(train) + len(validation), corpus["characters"])
        self.assertEqual(manifest["vocab"], tokenizer.vocab)
        self.assertEqual({c["id"] for c in manifest["checkpoints"]}, {"initial", "early", "trained"})
        for checkpoint in manifest["checkpoints"]:
            content = (manifest_path.parent / checkpoint["file"]).read_bytes()
            self.assertEqual(len(content), checkpoint["bytes"])
            self.assertEqual(sha256(content).hexdigest(), checkpoint["sha256"])
            payload = json.loads(content)
            self.assertEqual(payload["step"], checkpoint["step"])
        initial, trained = manifest["history"][0], manifest["history"][-1]
        self.assertGreater(initial["validation_loss"], trained["validation_loss"])


if __name__ == "__main__":
    unittest.main()
