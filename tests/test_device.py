"""Backend selection failures must not silently fall back during GPU checks."""

import unittest
from unittest.mock import patch

from inference_engine.device import available_devices, resolve_device


class DeviceTests(unittest.TestCase):
    def test_unavailable_cuda_distinguishes_build_from_driver_visibility(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.version.cuda", None):
                with self.assertRaisesRegex(ValueError, "no CUDA support"):
                    resolve_device("cuda")
            with patch("torch.version.cuda", "12.4"):
                with self.assertRaisesRegex(ValueError, "no usable NVIDIA GPU"):
                    resolve_device("cuda")

    def test_available_cuda_is_included_and_selected(self):
        with patch("torch.cuda.is_available", return_value=True), patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(available_devices(), ["cpu", "cuda"])
            self.assertEqual(str(resolve_device("auto")), "cuda")
            self.assertEqual(str(resolve_device("cuda")), "cuda")

    def test_cpu_fallback_and_unknown_device(self):
        with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(available_devices(), ["cpu"])
            self.assertEqual(str(resolve_device("auto")), "cpu")
        with self.assertRaisesRegex(ValueError, "Unknown device"):
            resolve_device("typo")


if __name__ == "__main__":
    unittest.main()
