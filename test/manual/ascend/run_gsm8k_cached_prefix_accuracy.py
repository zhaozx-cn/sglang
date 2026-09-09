#!/usr/bin/env python3
"""Run a leakage-free GSM8K accuracy check through the cached-prefix MLA path.

This is a manual Ascend diagnostic. It assumes an already-running SGLang server,
warms the same prompt prefix on every DP rank, routes scored requests across the
warmed ranks, and rejects the result if any scored request reports zero cached
tokens.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

INVALID = -9999999
STOP_STRINGS = ["Question", "Assistant:", "<|separator|>"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GSM8K after explicitly warming every DP rank."
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument(
        "--few-shot-data-path",
        help=(
            "Optional separate demonstrations file. When omitted, demonstrations, "
            "warmup, and scored questions are selected from disjoint ranges of "
            "--data-path."
        ),
    )
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--parallel", type=int, default=32)
    parser.add_argument("--dp-size", type=int, default=4)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument(
        "--output-file",
        help="JSONL output path (default: timestamped file in the current directory).",
    )
    parser.add_argument(
        "--no-flush-cache",
        action="store_true",
        help="Do not flush the server cache before per-DP warmup.",
    )
    args = parser.parse_args()

    if args.num_questions <= 0:
        parser.error("--num-questions must be positive")
    if args.num_shots <= 0:
        parser.error("--num-shots must be positive")
    if args.parallel <= 0:
        parser.error("--parallel must be positive")
    if args.dp_size <= 0:
        parser.error("--dp-size must be positive")
    return args


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def format_example(example: dict, include_answer: bool) -> str:
    text = f"Question: {example['question']}\nAnswer:"
    if include_answer:
        text += f" {example['answer']}"
    return text


def get_answer_value(answer: str) -> int:
    numbers = re.findall(r"\d+", answer.replace(",", ""))
    if not numbers:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID


def post_json(url: str, payload: dict | None, timeout: float) -> dict:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise TypeError(f"Expected one response object from {url}, got {type(result)}")
    return result


def flush_cache(base_url: str, timeout: float) -> None:
    response = requests.post(f"{base_url}/flush_cache", timeout=timeout)
    response.raise_for_status()


def generate(
    base_url: str,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    dp_rank: int,
    timeout: float,
) -> dict:
    return post_json(
        f"{base_url}/generate",
        {
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop": STOP_STRINGS,
            },
            "routed_dp_rank": dp_rank,
        },
        timeout,
    )


def main() -> None:
    args = parse_args()
    host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    if host != args.host:
        print("Client host 0.0.0.0 normalized to 127.0.0.1")
    base_url = f"http://{host}:{args.port}"

    data_path = Path(args.data_path).resolve()
    shot_path = Path(args.few_shot_data_path or args.data_path).resolve()
    eval_examples = read_jsonl(str(data_path))
    shot_examples = read_jsonl(str(shot_path))
    if shot_path == data_path:
        shot_examples = eval_examples
        # Keep demonstrations, the unscored warmup question, and scored questions
        # disjoint even when only one JSONL file is available.
        eval_start = args.num_shots
    else:
        eval_start = 0

    if len(shot_examples) < args.num_shots:
        raise ValueError(
            f"Need {args.num_shots} demonstrations, found {len(shot_examples)}"
        )
    required_eval_examples = eval_start + 1 + args.num_questions
    if len(eval_examples) < required_eval_examples:
        raise ValueError(
            f"Need at least {required_eval_examples} rows in {args.data_path}: "
            "one unscored warmup question plus the scored questions are required"
        )

    demonstrations = shot_examples[: args.num_shots]
    warmup_example = eval_examples[eval_start]
    scored_examples = eval_examples[
        eval_start + 1 : eval_start + 1 + args.num_questions
    ]
    shared_prefix = "".join(
        format_example(example, include_answer=True) + "\n\n"
        for example in demonstrations
    )
    warmup_prompt = shared_prefix + format_example(warmup_example, include_answer=False)

    print(
        f"Prepared {args.num_shots} demonstrations, 1 unscored warmup question, "
        f"and {len(scored_examples)} disjoint scored questions"
    )
    if not args.no_flush_cache:
        flush_cache(base_url, args.timeout)
        print("Flushed radix cache")

    # A full prompt, rather than the bare prefix, avoids tokenizer boundary
    # differences at the end of the shared prefix. The warmup question is never
    # scored and its answer is not included.
    for dp_rank in range(args.dp_size):
        result = generate(
            base_url,
            warmup_prompt,
            max_new_tokens=0,
            temperature=0.0,
            dp_rank=dp_rank,
            timeout=args.timeout,
        )
        meta = result.get("meta_info") or {}
        print(
            f"Warmed DP{dp_rank}: prompt_tokens={meta.get('prompt_tokens')}, "
            f"cached_tokens={meta.get('cached_tokens')}"
        )

    def evaluate(index_and_example: tuple[int, dict]) -> dict:
        index, example = index_and_example
        dp_rank = index % args.dp_size
        prompt = shared_prefix + format_example(example, include_answer=False)
        response = generate(
            base_url,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            dp_rank=dp_rank,
            timeout=args.timeout,
        )
        text = response.get("text") or ""
        meta = response.get("meta_info") or {}
        label = get_answer_value(example["answer"])
        prediction = get_answer_value(text)
        return {
            "index": index,
            "source_index": eval_start + 1 + index,
            "dp_rank": dp_rank,
            "question": example["question"],
            "label": label,
            "prediction": prediction,
            "correct": prediction == label,
            "cached_tokens": int(meta.get("cached_tokens") or 0),
            "prompt_tokens": int(meta.get("prompt_tokens") or 0),
            "completion_tokens": int(meta.get("completion_tokens") or 0),
            "output": text,
        }

    started_at = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [
            executor.submit(evaluate, item) for item in enumerate(scored_examples)
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            print(f"Completed {completed}/{len(futures)}", end="\r", flush=True)
    print()
    latency = time.perf_counter() - started_at
    results.sort(key=lambda item: item["index"])

    output_path = Path(
        args.output_file
        or f"gsm8k_cached_prefix_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    missed = [result for result in results if result["cached_tokens"] <= 0]
    accuracy = sum(result["correct"] for result in results) / len(results)
    invalid_rate = sum(result["prediction"] == INVALID for result in results) / len(
        results
    )
    completion_tokens = sum(result["completion_tokens"] for result in results)
    cached_values = [result["cached_tokens"] for result in results]

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Invalid: {invalid_rate:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {completion_tokens / latency:.3f} token/s")
    print(
        f"Cached tokens: min={min(cached_values)}, max={max(cached_values)}, "
        f"missed_requests={len(missed)}"
    )
    print(f"Results: {output_path.resolve()}")

    if missed:
        missed_indices = [result["index"] for result in missed]
        raise RuntimeError(
            "Accuracy is not a valid cached-prefix-path result because these "
            f"requests had cached_tokens=0: {missed_indices}"
        )
    print(
        "All scored requests hit radix cache. Confirm the server log contains "
        "'Entered Ascend MLA cached-prefix extend branch' for every DP rank."
    )


if __name__ == "__main__":
    main()
