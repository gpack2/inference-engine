"""The staged verifier must retain its independent numerical oracle."""

from pathlib import Path
import sys
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from verify_pretrained import captured_forward, reference_fixture, verify_fixture
from inference_engine.device import available_devices
from inference_engine.pretrained import import_qwen_weights


class VerificationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(1)
        config = Qwen2Config(vocab_size=31, hidden_size=32, intermediate_size=48,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=24, tie_word_embeddings=True)
        config._attn_implementation = "eager"
        self.reference = Qwen2ForCausalLM(config).eval()
        self.model = import_qwen_weights(config.to_dict(), self.reference.state_dict(), max_seq_len=24)
        self.tokens = torch.tensor([[1, 7, 12, 4]])
        self.cpu_logits, self.cpu_layers = captured_forward(self.reference, self.tokens, reference=True)

    def test_fixture_stays_on_cpu_and_retains_all_decode_comparisons(self):
        for device in available_devices():
            with self.subTest(device=device):
                self.reference.to(device)
                fixture = reference_fixture(self.reference, self.tokens.to(device), eos=30, new_tokens=3)
                self.reference.to("cpu")
                for tensor in [fixture["logits"], fixture["chunked"], fixture["generated"],
                               *fixture["layers"].values(),
                               *(x for step in fixture["decode"] for x in step.values())]:
                    self.assertEqual(tensor.device.type, "cpu")
                self.model.to(device)
                result = verify_fixture(self.model, self.tokens.to(device), fixture, self.cpu_logits,
                                        self.cpu_layers, eos=30, new_tokens=3)
                self.assertTrue(result["greedy_token_match_cached_and_uncached"])
                self.assertEqual(len(result["decode_logits"]), len(result["generated_ids"]))
                self.assertEqual(len(result["layers"]), 4)
                self.model.to("cpu")

    def test_corrupted_reference_is_rejected(self):
        fixture = reference_fixture(self.reference, self.tokens, eos=30, new_tokens=3)
        fixture["logits"] = fixture["logits"] + 1
        with self.assertRaises(AssertionError):
            verify_fixture(self.model, self.tokens, fixture, self.cpu_logits, self.cpu_layers,
                           eos=30, new_tokens=3)
