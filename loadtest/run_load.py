from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import httpx

QUESTION_POOL = [
    "What is the standard deduction for a single filer in 2024?",
    "What is the standard deduction for married filing jointly in 2024?",
    "Who qualifies to file as head of household?",
    "What is a qualifying child for tax purposes?",
    "When do I have to pay estimated tax?",
    "What are the 2024 federal income tax brackets for a single filer?",
    "Can I claim my parent as a dependent?",
    "How much federal tax do I owe on $85,000 filing single, 0 dependents, 2024?",
    "Estimate my 2024 federal tax: married filing jointly, $150,000, 2 dependents.",
    "My taxable income is $50,000, single. What is my 2024 federal tax?",
    "What is my effective federal tax rate on $200,000, single, 2024?",
    "Does the standard deduction increase if I am 65 or older?",
    "What filing status applies if my spouse died this year?",
    "When are estimated tax payments due?",
    "What is the additional standard deduction for the blind?",
    "How is taxable income different from gross income?",
    "What is the top marginal federal tax rate in 2024?",
    "Do I need to file if my income is below the standard deduction?",
    "How much tax on $120,000, head of household, 1 dependent, 2024?",
    "What is the capital of France?",
]


def _client_factory() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=180)


def percentiles(latencies_ms: list[float]) -> dict:
    s = sorted(latencies_ms)
    q = statistics.quantiles(s, n=100, method="inclusive") if len(s) > 1 else [s[0]] * 99
    return {"p50": round(q[49], 1), "p90": round(q[89], 1), "p95": round(q[94], 1),
            "p99": round(q[98], 1), "mean": round(statistics.fmean(s), 1), "max": round(s[-1], 1)}


async def run_load(api_url: str, n: int = 100, concurrency: int = 4) -> dict:
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    node_times: dict[str, list[float]] = {}
    errors = 0

    async with _client_factory() as client:
        async def one(i: int) -> None:
            nonlocal errors
            q = QUESTION_POOL[i % len(QUESTION_POOL)]
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post(f"{api_url}/chat", json={"question": q})
                    r.raise_for_status()
                    latencies.append((time.perf_counter() - t0) * 1000)
                    for step in r.json().get("steps", []):
                        node_times.setdefault(step["node"], []).append(step["duration_ms"])
                except Exception:  # noqa: BLE001
                    errors += 1

        wall0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(n)))
        wall = time.perf_counter() - wall0

    return {
        "n": n, "concurrency": concurrency, "errors": errors,
        "throughput_rps": round((n - errors) / wall, 2) if wall else 0.0,
        "latency_ms": percentiles(latencies) if latencies else {},
        "per_node_ms": {k: round(statistics.fmean(v), 1) for k, v in node_times.items()},
        "latencies": latencies,
    }


def write_report(result: dict, out_md: str = "docs/loadtest-results.md",
                 out_png: str = "docs/loadtest-latency.png") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_node = result.get("per_node_ms", {})
    bottleneck = max(per_node, key=per_node.get) if per_node else "unknown"
    lines = [
        "# Load Test Results", "",
        f"- Requests: {result['n']}  |  Concurrency: {result['concurrency']}  |  Errors: {result['errors']}",
        f"- Throughput: {result['throughput_rps']} req/s", "",
        "## Latency (ms)", "",
        "| p50 | p90 | p95 | p99 | mean | max |",
        "|-----|-----|-----|-----|------|-----|",
        "| {p50} | {p90} | {p95} | {p99} | {mean} | {max} |".format(**result["latency_ms"]),
        "", "## Per-node mean (ms)", "",
        *[f"- {k}: {v}" for k, v in sorted(per_node.items(), key=lambda x: -x[1])],
        "", "## Bottleneck", "",
        f"`{bottleneck}` is the dominant per-request cost (mean {per_node.get(bottleneck, 0)} ms). "
        "In `dummy` mode this is vector search over the embedded Chroma index; in `ollama` "
        "mode LLM generation in `synthesize` typically dominates instead.",
        "", "## Optimization recommendations", "",
        "1. Cut LLM calls on the retrieval path: replace the `grade_docs` LLM grader with the "
        "score threshold only, and merge `expand_query` into a single call — removes ~2 LLM "
        "round-trips per request.",
        "2. Add a semantic response cache keyed on normalized question + route; skip the "
        "`validate` retry when citations are present and the calc number is already in the draft.",
    ]
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines) + "\n")

    latencies = result.get("latencies") or []
    if latencies:
        plt.figure()
        plt.hist(latencies, bins=min(30, max(5, len(latencies) // 3)))
        plt.title("Latency distribution")
        plt.xlabel("ms")
        plt.ylabel("count")
        plt.savefig(out_png, dpi=100)
        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    result = asyncio.run(run_load(args.api_url, args.n, args.concurrency))
    write_report(result)
    print(result)


if __name__ == "__main__":
    main()
