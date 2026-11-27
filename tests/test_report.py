from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmbench.config import load_config  # noqa: E402
from llmbench.report import (  # noqa: E402
    _mean_ci95,
    _sample_sd,
    _summarize_series,
    _wilson_ci,
    build_report,
)
from llmbench.utils import write_json  # noqa: E402


class ReportMathTests(unittest.TestCase):
    def test_wilson_ci_for_published_mmlu(self) -> None:
        self.assertEqual(_wilson_ci(23, 40), [42.2, 71.5])
        self.assertEqual(_wilson_ci(46, 200), [17.7, 29.3])
        self.assertIsNone(_wilson_ci(None, 40))
        self.assertIsNone(_wilson_ci(23, 0))

    def test_sample_sd_needs_two_repeats(self) -> None:
        self.assertIsNone(_sample_sd([10]))
        self.assertEqual(_sample_sd([10, 12, 14]), 2.0)

    def test_mean_ci95_is_wide_for_small_n(self) -> None:
        ci = _mean_ci95([10, 12, 14])
        self.assertIsNotNone(ci)
        self.assertLess(ci[0], 10)
        self.assertGreater(ci[1], 14)
        self.assertIsNone(_mean_ci95([10]))

    def test_series_aggregates_repeats(self) -> None:
        rows = [
            {"backend": "hf", "concurrency": 16, "out_tok_s": 100.0, "e2e_p95_ms": 1000.0},
            {"backend": "hf", "concurrency": 16, "out_tok_s": 120.0, "e2e_p95_ms": 1100.0},
        ]
        series = _summarize_series(rows)
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0]["repeats"], 2)
        self.assertEqual(series[0]["out_tok_s"], 110.0)
        self.assertEqual(series[0]["out_tok_s_sd"], 14.14)


class ReportGenerationTests(unittest.TestCase):
    def test_report_exposes_series_and_spread(self) -> None:
        cfg = load_config(ROOT / "configs" / "benchmark_v2.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_json(
                out / "raw" / "benchmark_hf.json",
                [
                    {
                        "backend": "hf",
                        "concurrency": 16,
                        "repeat": 0,
                        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                        "dtype": "float16",
                        "out_tok_s": 100.0,
                        "total_tok_s": 200.0,
                        "e2e_p95_ms": 1000.0,
                        "ttft_p95_ms": 10.0,
                        "peak_gpu_mib": 4000.0,
                        "errors": 0,
                    },
                    {
                        "backend": "hf",
                        "concurrency": 16,
                        "repeat": 1,
                        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                        "dtype": "float16",
                        "out_tok_s": 120.0,
                        "total_tok_s": 240.0,
                        "e2e_p95_ms": 1100.0,
                        "ttft_p95_ms": 12.0,
                        "peak_gpu_mib": 4100.0,
                        "errors": 0,
                    },
                ],
            )
            write_json(
                out / "raw" / "quality_hf.json",
                {
                    "backend": "hf",
                    "mmlu_accuracy_pct": 50.0,
                    "mmlu_correct": 100,
                    "mmlu_total": 200,
                    "perplexity": 20.0,
                    "mmlu_note": "",
                    "perplexity_note": "",
                },
            )
            payload = build_report(out, cfg)
            text = (out / "tradeoff_table.md").read_text(encoding="utf-8")

        self.assertEqual(payload["series"][0]["out_tok_s_sd"], 14.14)
        self.assertEqual(payload["summary"][0]["out_tok_s_spread_text"], "110.0 ± 14.1")
        self.assertIn("Out tok/s ±SD", text)
        self.assertIn("E2E p95 ±SD", text)
        self.assertIn("MMLU 95% CI", text)
        self.assertIn("3 repeats were measured", text)


if __name__ == "__main__":
    unittest.main()
