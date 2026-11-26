from __future__ import annotations

import random
from typing import Any

# Deterministic, dependency-free workload. The filler is intentionally boring:
# we want to measure serving overhead, not prompt creativity.
_TEMPLATES = [
    "Summarize the main tradeoffs in {topic} for a small engineering team.",
    "Explain {topic} to a new graduate in five short paragraphs.",
    "Write a concise checklist for reviewing a design document about {topic}.",
    "Compare two common approaches to {topic} and recommend one.",
    "Draft three interview questions that test practical knowledge of {topic}.",
    "What are three failure modes teams commonly hit with {topic}?",
    "Give a step-by-step plan to debug an incident involving {topic}.",
    "Describe how you would monitor {topic} in production.",
]

_TOPICS = [
    "KV-cache memory management",
    "continuous batching",
    "4-bit weight quantization",
    "GPU memory bandwidth",
    "p95 latency under concurrency",
    "tokenizer alignment after quantization",
    "LLM serving autoscaling",
    "prompt length skew",
    "speculative decoding",
    "GEMM kernel selection",
]

_FILLERS = [
    "Keep the answer operational and mention concrete metrics.",
    "Prefer short paragraphs with no markdown tables.",
    "Include one sentence about what could go wrong.",
    "Assume the reader has basic systems knowledge.",
    "Avoid marketing language and avoid repeating the question.",
    "Be specific enough that an on-call engineer could act on it.",
    "Mention memory, latency, and throughput where relevant.",
    "Use plain language and prefer examples over abstractions.",
]


def make_workload_prompts(num_requests: int, input_tokens: int, seed: int) -> list[str]:
    """Create a fixed synthetic workload with roughly `input_tokens` tokens each."""
    rng = random.Random(seed)
    prompts: list[str] = []
    target_chars = max(64, int(input_tokens * 3.6))
    for i in range(num_requests):
        topic = _TOPICS[i % len(_TOPICS)]
        text = _TEMPLATES[i % len(_TEMPLATES)].format(topic=topic)
        j = i
        while len(text) < target_chars:
            text += " " + _FILLERS[j % len(_FILLERS)]
            j += 1
        # A tiny bit of deterministic variation prevents identical cache hits.
        text += f" Case id: {i:04d}, seed: {rng.randint(0, 1_000_000)}."
        prompts.append(text)
    return prompts


_FALLBACK_PPL_TEXT = """
Quantization changes the numeric format used to store and multiply model weights.
For a small language model on a single GPU, the dominant cost during decoding is
often moving weights from device memory into the compute units rather than raw
arithmetic. Four-bit formats reduce that traffic and can therefore improve token
generation speed even when the arithmetic itself is not faster. The tradeoff is
that lower precision changes the logits slightly, which can show up as a small
increase in perplexity or a small drop in multiple-choice accuracy. Continuous
batching improves throughput for a different reason: instead of waiting for a
whole batch to finish before admitting new requests, the scheduler keeps the GPU
busy by filling freed slots with waiting work. Paged attention reduces fragmentation
of the key-value cache, allowing more concurrent sequences to remain resident.
""".strip()


def load_perplexity_texts(chars: int, seed: int) -> list[str]:
    """Load a fixed Wikitext-2 slice, falling back to bundled text if offline."""
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        lines = [row["text"].strip() for row in ds if isinstance(row.get("text"), str) and row["text"].strip()]
        rng = random.Random(seed)
        rng.shuffle(lines)
        text = "\n\n".join(lines[:200])[: max(512, chars)]
        if len(text) >= 512:
            return [text]
    except Exception:
        pass
    return [_FALLBACK_PPL_TEXT[: max(512, chars)]]


_MMLU_FALLBACK = [
    {
        "subject": "miscellaneous",
        "question": "Which unit is used to measure electrical resistance?",
        "choices": ["Ampere", "Volt", "Ohm", "Watt"],
        "answer": 2,
    },
    {
        "subject": "miscellaneous",
        "question": "What does HTTP status code 404 mean?",
        "choices": ["OK", "Not Found", "Redirect", "Server Error"],
        "answer": 1,
    },
    {
        "subject": "high_school_mathematics",
        "question": "What is the derivative of x^2 with respect to x?",
        "choices": ["x", "2x", "x^3", "1"],
        "answer": 1,
    },
    {
        "subject": "college_computer_science",
        "question": "What is the average-case time complexity of binary search?",
        "choices": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
        "answer": 1,
    },
]


def load_mmlu(subjects: list[str], n_examples: int, seed: int) -> list[dict[str, Any]]:
    """Load a small fixed MMLU subset. Falls back to bundled examples if offline."""
    try:
        from datasets import load_dataset

        rows: list[dict[str, Any]] = []
        rng = random.Random(seed)
        if not subjects or "all" in subjects:
            ds = load_dataset("cais/mmlu", "all", split="test")
            by_subject: dict[str, list[dict[str, Any]]] = {}
            for row in ds:
                by_subject.setdefault(str(row["subject"]), []).append(row)
            keys = sorted(by_subject)
            per_subject = max(1, n_examples // max(1, len(keys)))
            for key in keys:
                sample = rng.sample(by_subject[key], min(per_subject, len(by_subject[key])))
                rows.extend(sample)
        else:
            per_subject = max(1, n_examples // len(subjects))
            for subject in subjects:
                try:
                    ds = load_dataset("cais/mmlu", subject, split="test")
                except Exception:
                    continue
                sample = rng.sample(list(ds), min(per_subject, len(ds)))
                rows.extend(sample)
        if rows:
            rng.shuffle(rows)
            return rows[:n_examples]
    except Exception:
        pass
    return _MMLU_FALLBACK[: min(n_examples, len(_MMLU_FALLBACK))]


CHOICE_LABELS = [" A", " B", " C", " D"]


def format_mmlu_prompt(row: dict[str, Any]) -> str:
    subject = str(row.get("subject", "general knowledge")).replace("_", " ")
    choices = row["choices"]
    choices_block = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices[:4]))
    return (
        f"The following is a multiple-choice question about {subject}.\n"
        f"Question: {row['question']}\n"
        f"{choices_block}\n"
        "Answer with exactly one letter: A, B, C, or D.\n"
        "Answer:"
    )
