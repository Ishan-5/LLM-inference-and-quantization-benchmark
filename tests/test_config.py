from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmbench.config import load_config  # noqa: E402
from llmbench.backends import build_backend  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_published_config_stays_reproducible(self) -> None:
        cfg = load_config(ROOT / "configs" / "benchmark.yaml")
        self.assertEqual(cfg.benchmark.repeats, 1)
        self.assertEqual(cfg.quality.mmlu_examples, 40)
        self.assertNotIn("vllm_awq", cfg.backends)

    def test_v2_config_closes_known_gaps(self) -> None:
        cfg = load_config(ROOT / "configs" / "benchmark_v2.yaml")
        self.assertEqual(cfg.benchmark.repeats, 3)
        self.assertEqual(cfg.quality.mmlu_examples, 200)
        self.assertEqual(cfg.results_dir, "results_v2")
        self.assertIn("vllm_awq", cfg.backends)
        self.assertEqual(cfg.backend("vllm_awq").extra.get("quantization"), "awq")

    def test_vllm_awq_builds_as_vllm_backend(self) -> None:
        cfg = load_config(ROOT / "configs" / "benchmark_v2.yaml")
        backend = build_backend("vllm_awq", cfg.backend("vllm_awq"), cfg)
        self.assertEqual(backend.name, "vllm_awq")
        self.assertEqual(type(backend).__name__, "VLLMBackend")


if __name__ == "__main__":
    unittest.main()
