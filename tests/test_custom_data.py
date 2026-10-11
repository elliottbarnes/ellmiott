"""Custom data must survive training, export, metadata, and inference."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from ellmiott_gpt.corpus import load_corpus, CharacterTokenizer
from ellmiott_gpt.export import load_export
from ellmiott_gpt.sample import sample

class CustomDataTests(unittest.TestCase):
    def test_non_shakespeare_training_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data=root/'private-name.txt'
            data.write_text('αβγδε ζηθ😀. '*100,encoding='utf-8')
            subprocess.run([sys.executable,'-m','ellmiott_gpt.train','--data',str(data),'--steps','2','--early-step','1','--eval-every','1','--eval-batches','1','--batch-size','1','--threads','1','--output',str(root/'models'),'--run-dir',str(root/'run'),'--fixtures',str(root/'parity.json')],check=True,capture_output=True,text=True)
            meta=json.loads((root/'models/manifest.json').read_text())
            self.assertEqual(meta['corpus']['title'],'Custom training text')
            self.assertEqual(meta['corpus']['url'],'')
            self.assertNotIn('private-name',json.dumps(meta))
            cases=json.loads((root/'parity.json').read_text())
            self.assertTrue(cases)
            model,payload=load_export(root/'models/trained.json')
            torch.set_num_threads(1)
            output=sample(model,CharacterTokenizer(payload['vocab']),'',count=8)
            self.assertEqual(len(output),8)
            self.assertTrue(set(output)<=set(payload['vocab']))

    def test_invalid_training_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.txt'
            for text in ['abc','a'*700,'ab\0'*300]:
                path.write_text(text)
                with self.assertRaises(ValueError):load_corpus(path)
