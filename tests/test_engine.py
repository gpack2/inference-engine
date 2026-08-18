"""Numerical and runtime invariants for the first engine slice."""

import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from inference_engine import ModelConfig, TinyDecoder, generate
from inference_engine.ops import RMSNorm, apply_rope, attention


def config(**overrides):
    values = dict(vocab_size=31, hidden_size=32, intermediate_size=48,
                  num_layers=2, num_heads=4, num_kv_heads=2, max_seq_len=24)
    return ModelConfig(**(values | overrides))


def available_devices():
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)
        self.model = TinyDecoder(config())
        self.tokens = torch.randint(0, 31, (2, 9))

    def assert_close(self, actual, expected):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-4, atol=2e-5)

    def test_cached_logits_match_full_prefix_for_chunks_and_single_tokens(self):
        expected = self.model(self.tokens)
        for device in available_devices():
            self.model.to(device)
            tokens = self.tokens.to(device)
            for chunks in ([9], [3, 1, 1, 4], [1] * 9):
                with self.subTest(device=device, chunks=chunks):
                    cache = self.model.new_cache(batch_size=2)
                    outputs, offset = [], 0
                    for length in chunks:
                        outputs.append(self.model(tokens[:, offset:offset + length], cache))
                        offset += length
                        self.assertEqual(cache.length, offset)
                    self.assert_close(torch.cat(outputs, dim=1), expected)

    def test_no_future_token_leakage(self):
        for device in available_devices():
            with self.subTest(device=device):
                self.model.to(device)
                tokens = self.tokens.to(device)
                changed = tokens.clone()
                changed[:, 4:] = (changed[:, 4:] + 7) % 31
                self.assert_close(self.model(tokens)[:, :4], self.model(changed)[:, :4])

    def test_batched_requests_match_individual_requests(self):
        for device in available_devices():
            with self.subTest(device=device):
                self.model.to(device)
                tokens = self.tokens.to(device)
                batched = self.model(tokens, self.model.new_cache(batch_size=2))
                separate = torch.cat([self.model(row[None], self.model.new_cache()) for row in tokens])
                self.assert_close(batched, separate)

    def test_cache_capacity_reset_and_storage_reuse(self):
        for device in available_devices():
            with self.subTest(device=device):
                self.model.to(device)
                tokens = self.tokens.to(device)
                cache = self.model.new_cache(batch_size=2, capacity=9)
                keys, values = cache.keys, cache.values
                cfg = self.model.config
                self.assertEqual(cache.nbytes, 2 * cfg.num_layers * 2 * 9 * cfg.num_kv_heads * cfg.head_dim * 4)
                self.model(tokens, cache)
                with self.assertRaisesRegex(ValueError, "capacity"):
                    self.model(tokens[:, :1], cache)
                self.assertEqual(cache.length, 9)
                cache.reset()
                self.assertEqual(cache.length, 0)
                self.assertIs(cache.keys, keys)
                self.assertIs(cache.values, values)
                replacement = (tokens[:, :4] + 1) % 31
                self.assert_close(self.model(replacement, cache), self.model(replacement))

    def test_failed_forward_does_not_commit_cache_position(self):
        cache = self.model.new_cache(batch_size=2)
        self.model(self.tokens[:, :3], cache)
        with patch.object(self.model.layers[1], "forward", side_effect=RuntimeError("injected failure")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.model(self.tokens[:, 3:5], cache)
        self.assertEqual(cache.length, 3)
        actual = self.model(self.tokens[:, 3:5], cache)
        self.assert_close(actual, self.model(self.tokens[:, :5])[:, 3:5])

    def test_generation_cache_parity_and_eos(self):
        for device in available_devices():
            with self.subTest(device=device):
                self.model.to(device)
                prompt = self.tokens[:1, :4].to(device)
                cached = generate(self.model, prompt, 6)
                uncached = generate(self.model, prompt, 6, use_cache=False)
                torch.testing.assert_close(cached, uncached, rtol=0, atol=0)
                torch.testing.assert_close(cached[:, :4], prompt, rtol=0, atol=0)
                eos = cached[0, 4].item()
                stopped = generate(self.model, prompt, 6, eos_token_id=eos)
                self.assertEqual(stopped.shape[1], 5)
                self.assertEqual(stopped[0, -1].item(), eos)
                self.assert_close(generate(self.model, prompt, 0), prompt)

    def test_context_boundary_and_invalid_generation(self):
        prompt = self.tokens[:1]
        output = generate(self.model, prompt, 15)
        self.assertEqual(output.shape[1], 24)
        for kwargs in ({"max_new_tokens": 16}, {"max_new_tokens": -1},
                       {"eos_token_id": 31}, {"max_new_tokens": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate(self.model, prompt, **kwargs)
        with self.assertRaisesRegex(ValueError, "one request"):
            generate(self.model, self.tokens, 1)

    def test_invalid_tokens_and_cache_ownership(self):
        bad_inputs = [torch.empty(1, 0, dtype=torch.long), torch.tensor([[31]]),
                      torch.tensor([[-1]]), self.tokens.float(), self.tokens[0],
                      torch.ones(1, 25, dtype=torch.long)]
        for tokens in bad_inputs:
            with self.subTest(shape=tokens.shape), self.assertRaises(ValueError):
                self.model(tokens)
        with self.assertRaisesRegex(ValueError, "different model"):
            self.model(self.tokens, TinyDecoder(config()).new_cache(batch_size=2))
        with self.assertRaisesRegex(ValueError, "batch size"):
            self.model(self.tokens, self.model.new_cache(batch_size=1))
        for capacity in (0, 25, 1.5):
            with self.assertRaises(ValueError):
                self.model.new_cache(capacity=capacity)
        if torch.backends.mps.is_available():
            cache = self.model.new_cache(batch_size=2)
            self.model.to("mps")
            with self.assertRaisesRegex(ValueError, "cache device"):
                self.model(self.tokens.to("mps"), cache)

    def test_invalid_config(self):
        for overrides in ({"num_heads": 3}, {"num_kv_heads": 3}, {"hidden_size": 12},
                          {"num_layers": 0}, {"norm_eps": 0}, {"rope_theta": float("nan")}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                config(**overrides)


class OperationTests(unittest.TestCase):
    def test_gqa_against_independent_per_head_sdpa(self):
        torch.manual_seed(7)
        for start, count in ((0, 7), (6, 1), (3, 4)):
            q = torch.randn(2, 4, count, 8)
            k, v = torch.randn(2, 2, 7, 8), torch.randn(2, 2, 7, 8)
            allowed = torch.arange(7)[None, :] <= (torch.arange(count) + start)[:, None]
            expected = torch.cat([
                F.scaled_dot_product_attention(q[:, h:h+1], k[:, h//2:h//2+1],
                                               v[:, h//2:h//2+1], attn_mask=allowed)
                for h in range(4)
            ], dim=1)
            for device in available_devices():
                with self.subTest(device=device, start=start):
                    actual = attention(q.to(device), k.to(device), v.to(device), start)
                    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)

    def test_rope_zero_position_norm_and_absolute_offset(self):
        x = torch.randn(1, 2, 5, 8)
        positions = torch.arange(5)
        for device in available_devices():
            with self.subTest(device=device):
                rotated = apply_rope(x.to(device), positions.to(device), 10000.0).cpu()
                torch.testing.assert_close(rotated[:, :, 0], x[:, :, 0])
                torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1))
                last = apply_rope(x[:, :, 4:].to(device), positions[4:].to(device), 10000.0).cpu()
                torch.testing.assert_close(last, rotated[:, :, 4:])
                # First pair has frequency 1: validate signs and split-half pairing.
                expected = x[:, :, 1, 0] * torch.cos(torch.tensor(1.0)) - x[:, :, 1, 4] * torch.sin(torch.tensor(1.0))
                torch.testing.assert_close(rotated[:, :, 1, 0], expected)

    def test_rmsnorm_against_pytorch(self):
        x = torch.randn(2, 5, 32)
        norm = RMSNorm(32)
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.5, 1.5, 32))
        expected = F.rms_norm(x, (32,), norm.weight, eps=norm.eps).detach()
        for device in available_devices():
            with self.subTest(device=device):
                actual = norm.to(device)(x.to(device)).detach().cpu()
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
