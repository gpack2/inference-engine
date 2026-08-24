"""Offline tests: tiny Transformers reference models, no checkpoint downloads."""

import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from inference_engine import generate
from inference_engine.pretrained import config_from_qwen, import_qwen_weights


class PretrainedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(1)
        self.config = Qwen2Config(vocab_size=31, hidden_size=32, intermediate_size=48,
                                  num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, max_position_embeddings=24,
                                  tie_word_embeddings=True, rope_theta=10000.0)
        self.config._attn_implementation = "eager"
        self.reference = Qwen2ForCausalLM(self.config).eval()
        self.raw = self.config.to_dict()

    def test_mapped_logits_and_cached_chunks_match_transformers(self):
        tokens = torch.tensor([[1, 7, 12, 2, 9, 6, 22], [3, 8, 11, 5, 7, 1, 13]])
        with torch.inference_mode():
            expected = self.reference(tokens).logits
        devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                model = import_qwen_weights(self.raw, self.reference.state_dict(), max_seq_len=24).to(device)
                self.assertIs(model.lm_head.weight, model.embedding.weight)
                actual = model(tokens.to(device))
                torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-5)
                cache = model.new_cache(batch_size=2)
                chunks = [model(tokens[:, :3].to(device), cache), model(tokens[:, 3:4].to(device), cache),
                          model(tokens[:, 4:].to(device), cache)]
                torch.testing.assert_close(torch.cat(chunks, dim=1).cpu(), expected, rtol=2e-4, atol=2e-5)

    def test_greedy_tokens_match_transformers(self):
        model = import_qwen_weights(self.raw, self.reference.state_dict(), max_seq_len=24)
        prompt = torch.tensor([[1, 7, 12, 4]])
        with torch.inference_mode():
            expected = self.reference.generate(prompt, attention_mask=torch.ones_like(prompt),
                                               max_new_tokens=5, do_sample=False,
                                               eos_token_id=None, pad_token_id=0)
        torch.testing.assert_close(generate(model, prompt, 5), expected, rtol=0, atol=0)

    def test_untied_output_weights_are_loaded(self):
        self.config.tie_word_embeddings = False
        reference = Qwen2ForCausalLM(self.config).eval()
        model = import_qwen_weights(self.config.to_dict(), reference.state_dict(), max_seq_len=24)
        self.assertIsNot(model.lm_head.weight, model.embedding.weight)
        tokens = torch.tensor([[1, 7, 12]])
        with torch.inference_mode():
            torch.testing.assert_close(model(tokens), reference(tokens).logits, rtol=2e-4, atol=2e-5)

    def test_missing_extra_wrong_shape_and_inconsistent_ties_rejected(self):
        source = self.reference.state_dict()
        cases = []
        missing = dict(source)
        del missing["model.layers.0.self_attn.q_proj.weight"]
        cases.append(missing)
        cases.append(source | {"unexpected.weight": torch.ones(1)})
        cases.append(source | {"model.norm.weight": torch.ones(3)})
        cases.append(source | {"lm_head.weight": source["lm_head.weight"] + 1})
        for weights in cases:
            with self.assertRaises(ValueError):
                import_qwen_weights(self.raw, weights, max_seq_len=24)

    def test_unsupported_config_rejected(self):
        for change in ({"model_type": "other"}, {"hidden_act": "gelu"},
                       {"use_sliding_window": True}, {"rope_scaling": {"factor": 2}},
                       {"use_mrope": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                config_from_qwen(self.raw | change, max_seq_len=24)
        with self.assertRaises(ValueError):
            config_from_qwen(self.raw, max_seq_len=25)


if __name__ == "__main__":
    unittest.main()
