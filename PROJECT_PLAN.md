# 3–4 Week Execution Plan

The goal is not "run four tools." The goal is a defensible systems result with a mechanism explanation.

## Definition of done

- [ ] Four setups produce valid numbers: `hf`, `vllm`, `llamacpp`, `awq` (GPTQ optional).
- [ ] Concurrency sweep includes at least `1, 4, 8, 16`.
- [ ] Every run records GPU name, package versions, seed, and config.
- [ ] Perplexity and MMLU subset are reported for each setup, with caveats where logprobs are unavailable.
- [ ] `results/tradeoff_table.md` includes a clear recommendation and *why*.
- [ ] README explains batching, KV cache, and memory bandwidth in relation to the observed numbers.

---

## Week 1 — Instrumentation and baseline

**Goal:** trust the measurement harness before adding more backends.

1. Create the Colab T4 runtime and run `scripts/bootstrap_colab.sh`.
2. Run `python -m llmbench.cli doctor`; save the output.
3. Smoke test only HF:

   ```bash
   python -m llmbench.cli benchmark --backends hf --limit-prompts 8
   ```

4. Verify in `results/raw/benchmark_hf.json` that:
   - token counts are non-zero,
   - TTFT < E2E latency,
   - memory is sampled (not `null`),
   - errors = 0.
5. Run the full HF concurrency sweep.
6. Write three sentences explaining the HF baseline: where does p95 start to bend, and why?

**Exit criterion:** you believe the baseline numbers.

---

## Week 2 — Serving runtimes

**Goal:** quantify the batching/KV-cache advantage.

1. Run vLLM:

   ```bash
   python -m llmbench.cli benchmark --backends vllm
   ```

2. Run llama.cpp GGUF:

   ```bash
   python -m llmbench.cli benchmark --backends llamacpp
   ```

3. Compare concurrency 1 vs 16:
   - Does vLLM's advantage grow with concurrency? That is continuous batching.
   - Does p95 for HF grow faster than throughput? That is static batching + KV fragmentation.
   - Does llama.cpp fall behind at high concurrency? That is scheduler/runtime overhead, not necessarily bad kernels.
4. If `llama_cpp` reports CPU-only, record it and do not mix it into the GPU ranking.

**Exit criterion:** you can explain the concurrency curves, not just the top-line tokens/sec.

---

## Week 3 — Quantization and quality

**Goal:** separate speed gains from quality loss.

1. Run AWQ:

   ```bash
   python -m llmbench.cli quality --backends hf,awq
   python -m llmbench.cli benchmark --backends awq
   ```

2. Run GPTQ if time permits:

   ```bash
   python -m llmbench.cli all --backends gptq
   ```

3. Inspect `results/raw/quality_*.json`:
   - `perplexity` should be close to baseline (small positive ΔPPL is expected).
   - `mmlu_accuracy_pct` deltas of 1–3 pp are normal on a 40-example subset.
   - If a backend has `null` quality, read its `*_note` field before claiming anything.
4. Increase `mmlu_examples` to 100 and `repeats` to 2 if results are noisy.

**Exit criterion:** you can say "setup X saves N MiB and loses M points," with evidence.

---

## Week 4 — Analysis and presentation

**Goal:** turn numbers into an interview story.

1. Build final report:

   ```bash
   python -m llmbench.cli report
   ```

2. Plot three charts from `results/summary.csv` (optional but strong):
   - output tok/s vs concurrency,
   - p95 E2E latency vs concurrency,
   - peak GPU MiB vs throughput.
3. Add a short "Mechanism" section to README using your actual result:
   - **Batching:** did throughput rise with concurrency? For whom?
   - **KV cache:** whose p94 stayed flat while others spiked?
   - **Memory bandwidth:** did 4-bit weights help single-stream latency or mainly memory?
4. Replace the placeholder resume line in README with your generated numbers.
5. Commit/push. Keep raw JSON out of git if large, but keep the final markdown and CSV.

**Exit criterion:** a reviewer can understand the winner in 60 seconds and challenge you on methodology without finding a hole.

---

## Experiment matrix cheat sheet

| Variable | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Input tokens target | 128 |
| Max new tokens | 128 |
| Decoding | greedy, temperature 0 |
| Concurrency | 1, 4, 8, 16 |
| Requests per level | 32 (increase to 64 for stability) |
| Warmup | 4 requests |
| Quality | WikiText-2 6000 chars + MMLU 40 questions |
| Hardware | T4 16 GB (record exact name) |
| Seed | 42 |

---

## Interview talking points

- "I separated **prefill** and **decode** effects by keeping input length fixed and varying concurrency."
- "I used p95, not average, because serving SLAs fail at the tail."
- "I treated quantization as a memory-bandwidth intervention and checked quality so the throughput claim was honest."
- "I ran each backend in a separate process because shared CUDA allocator state would invalidate the comparison."
- "I reported where the result does **not** generalize: different GPU, different kernel versions, different batch sizes."
