"""Check phase accounting and output parity without asserting timing speedups."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from projection import phase_trial, run_case
from decode import decode_trial
from decode import run_case as decode_case
from inference_engine import ModelConfig, TinyDecoder, generate
from inference_engine.device import available_devices


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)
        self.model = TinyDecoder(ModelConfig(vocab_size=31, hidden_size=16, intermediate_size=24,
                                            num_layers=1, num_heads=2, num_kv_heads=1, max_seq_len=16))
        self.prompt = torch.tensor([[1, 2, 3]])

    def test_diagnostic_outputs_phase_counts_and_tensor_bytes(self):
        for device in available_devices():
            self.model.to(device)
            prompt = self.prompt.to(device)
            for count in (2, 4):
                for last_only in (False, True):
                    with self.subTest(device=device, count=count, last_only=last_only):
                        actual, trace = phase_trial(self.model, prompt, count, last_only)
                        expected = generate(self.model, prompt, count, last_token_only=last_only)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertEqual(len(trace["decode_step_ms"]), count - 1)
                        self.assertEqual(trace["prefill_logits_tensor_bytes"], (1 if last_only else 3) * 31 * 4)
                        self.assertEqual(trace["cache_allocated_bytes"], self.model.new_cache(capacity=3+count).nbytes)

    def test_paired_trials_keep_raw_samples_and_alternate_order(self):
        result = run_case(self.model, self.prompt, 3, warmup=0, repeats=2,
                          diagnostic_repeats=1, order_index=0)
        self.assertTrue(result["parity"]["generated_token_ids_equal"])
        self.assertEqual(len(result["generated_ids"]), 3)
        self.assertEqual(result["normal_generation_execution_order"],
                         [["full_logits", "last_token_logits"], ["last_token_logits", "full_logits"]])
        for variant in result["variants"].values():
            self.assertEqual(len(variant["generation_ms"]["samples"]), 2)
            self.assertEqual(len(variant["diagnostic_runs"]), 1)
            self.assertEqual(len(variant["diagnostic_runs"][0]["decode_step_ms"]), 2)

    def test_decode_profile_preserves_tokens_and_accounts_for_cache(self):
        for device in available_devices():
            self.model.to(device)
            prompt = self.prompt.to(device)
            expected = generate(self.model, prompt, 4)
            original_forward = self.model.lm_head.forward
            for mode in ({}, {"diagnostic": True}, {"host_profile": True}):
                with self.subTest(device=device, mode=mode):
                    actual, trace = decode_trial(self.model, prompt, 3, **mode)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(trace["initial_cache_length"], 3)
                    self.assertEqual(trace["final_cache_length"], 6)
                    self.assertEqual(trace["cache_allocated_bytes"], self.model.new_cache(capacity=7).nbytes)
                    self.assertEqual(trace["cache_valid_bytes"], self.model.new_cache(capacity=6).nbytes)
                    self.assertEqual(self.model.lm_head.forward, original_forward)
                    if mode.get("diagnostic"):
                        counts = trace["component_calls_per_step"]
                        self.assertEqual(counts["rope"], 2 * self.model.config.num_layers)
                        self.assertEqual(counts["attention"], self.model.config.num_layers)
                        self.assertEqual(counts["validation"], 1)
                        self.assertEqual(counts["vocabulary_projection"], 1)
                        total = sum(trace["component_ms_per_step"].values()) + trace["unattributed_ms_per_step"]
                        self.assertAlmostEqual(total, trace["decode_mean_step_ms"])
                    if mode.get("host_profile"):
                        self.assertTrue(trace["host_operator_events"])

    def test_decode_ablation_preserves_parity_and_paired_sample_accounting(self):
        result = decode_case(self.model, self.prompt, 2, warmup=0, repeats=2,
                             diagnostic_repeats=1, order_index=1)
        self.assertTrue(result["all_trial_token_ids_equal"])
        self.assertTrue(result["valid_cache_entries_close"])
        self.assertEqual(len(result["generated_ids"]), 3)
        self.assertEqual(result["paired_execution_order"],
                         [["shared_rope", "recompute_rope"], ["recompute_rope", "shared_rope"]])
        for variant in result["variants"].values():
            self.assertEqual(len(variant["generation_ms"]["samples"]), 2)
            self.assertEqual(len(variant["normal_runs"]), 2)
        optimized = result["variants"]["shared_rope"]["diagnostic_runs"][0]
        self.assertEqual(optimized["component_calls_per_step"]["shared_rope_setup"], 1)


if __name__ == "__main__":
    unittest.main()
