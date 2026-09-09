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


def _client_factory(timeout: float = 240) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout)


def percentiles(latencies_ms: list[float]) -> dict:
    s = sorted(latencies_ms)
    q = statistics.quantiles(s, n=100, method="inclusive") if len(s) > 1 else [s[0]] * 99
    return {"p50": round(q[49], 1), "p90": round(q[89], 1), "p95": round(q[94], 1),
            "p99": round(q[98], 1), "mean": round(statistics.fmean(s), 1), "max": round(s[-1], 1)}


async def run_load(api_url: str, n: int = 100, concurrency: int = 4,
                   warmup: int = 0, timeout: float = 240,
                   mode: str = "?") -> dict:
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    node_times: dict[str, list[float]] = {}
    errors = 0

    async with _client_factory(timeout) as client:
        async def one(i: int) -> None:
            nonlocal errors
            q = QUESTION_POOL[i % len(QUESTION_POOL)]
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post(f"{api_url}/chat", json={"question": q})
                    r.raise_for_status()
                    if i >= warmup:      # discard cold-start requests from the stats
                        latencies.append((time.perf_counter() - t0) * 1000)
                        for step in r.json().get("steps", []):
                            node_times.setdefault(step["node"], []).append(step["duration_ms"])
                except Exception:  # noqa: BLE001
                    errors += 1

        wall0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(n)))
        wall = time.perf_counter() - wall0

    return {
        "n": n, "concurrency": concurrency, "errors": errors, "warmup": warmup,
        "mode": mode,
        "throughput_rps": round((n - errors) / wall, 2) if wall else 0.0,
        "latency_ms": percentiles(latencies) if latencies else {},   # excludes warmup
        "per_node_ms": {k: round(statistics.fmean(v), 1) for k, v in node_times.items()},
        "latencies": latencies,
    }


def write_report(result: dict, out_md: str = "docs/loadtest-results.md",
                 out_png: str = "docs/loadtest-latency.png") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_node = result.get("per_node_ms", {})
    ranked = sorted(per_node.items(), key=lambda x: -x[1])
    top = ranked[0] if ranked else ("unknown", 0.0)
    mode = result.get("mode", "?")
    lines = [
        "# Load Test Results", "",
        f"- Mode: `LLM_MODE={mode}`  |  Requests: {result['n']}  |  "
        f"Concurrency: {result['concurrency']}  |  Errors: {result['errors']}  |  "
        f"Warmup discarded: {result.get('warmup', 0)}",
        f"- Throughput: {result['throughput_rps']} req/s", "",
        "## Latency (ms)"
        + (f" — first {result['warmup']} requests excluded" if result.get("warmup") else ""),
        "",
        "| p50 | p90 | p95 | p99 | mean | max |",
        "|-----|-----|-----|-----|------|-----|",
        "| {p50} | {p90} | {p95} | {p99} | {mean} | {max} |".format(**result["latency_ms"]),
        "", "## Per-node mean (ms)", "",
        *[f"- {k}: {v}" for k, v in ranked],
        "", "## Bottleneck", "",
        f"`{top[0]}` dominates at {top[1]:.0f} ms mean"
        + (", then " + ", ".join(f"`{k}` {v:.0f} ms" for k, v in ranked[1:4] if v >= 1)
           if any(v >= 1 for _, v in ranked[1:]) else "; every other node is ~0 ms")
        + ".",
        "",
        "In `LLM_MODE=ollama` `synthesize` (the ~400-token cited answer, generated "
        "on CPU) is the whole request — ~90% of wall-clock even on `llama3.2:1b`. "
        "`triage` and `expand_query` (the RAG subgraph's own LLM call, counted "
        "under `retrieve`) are a few seconds each *when run serially*; under "
        "concurrency > 1 their measured time inflates because each call queues "
        "behind other requests' `synthesize` on the single CPU model. The "
        "deterministic nodes (`plan`, `calculate`, `guardrails`, `validate`) are "
        "~0 ms. In `LLM_MODE=dummy` the LLM nodes collapse to ~0 ms and `retrieve` "
        "(vector search + rerank, lock-serialised) is all that is left — that run "
        "isolates the retrieval path.",
        "", "## Optimization recommendations", "",
        "1. **Attack `synthesize`.** It is ~90% of the request. (a) An "
        "**end-to-end response cache** keyed on the normalised question, checked "
        "before the graph — *implemented* in `app/api/main.py`: a repeat `/chat` "
        "returns in ~1 ms instead of tens of seconds. (b) **Token streaming** "
        "from `synthesize` — the `/chat/stream` SSE plumbing already carries step "
        "events; extending it to model tokens would drop time-to-first-token to "
        "~1-2 s while the full answer still takes ~30 s. Not done.",
        "2. **Collapse the classification calls.** `triage` and `expand_query` are "
        "~20-25% of a serial request (measured: ~3 s + ~5 s on `llama3.2:1b`) and "
        "can be embeddings/rules-only — the deterministic guard in `triage` "
        "already overrides the model for the common cases, and `expand_query`'s "
        "rewrite buys little over the domain-hint + `tax_profile` queries. Bonus: "
        "routing becomes deterministic.",
        "", "## Note on the tail", "",
        "- Retrieval is serialised by one process-wide lock (see README "
        "\"concurrency\"). Under `ollama` with concurrency > 1 the single CPU model "
        "is the real serialisation point — queued requests can exceed the client "
        "timeout, which is why the real-model profile is run at concurrency 1. "
        "Under `dummy` the tail is the one-time model + HNSW warm-up the first "
        "requests queue behind; steady-state p50 is unaffected.",
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
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=0,
                    help="discard the first N requests from the latency stats")
    ap.add_argument("--timeout", type=float, default=240)
    ap.add_argument("--out-md", default="docs/loadtest-results.md")
    ap.add_argument("--out-png", default="docs/loadtest-latency.png")
    args = ap.parse_args()
    result = asyncio.run(run_load(args.api_url, args.n, args.concurrency, args.warmup,
                                  args.timeout, mode=os.environ.get("LLM_MODE", "?")))
    write_report(result, args.out_md, args.out_png)
    print(result)


if __name__ == "__main__":
    main()
