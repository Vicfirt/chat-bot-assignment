from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from app.config import get_settings
from app.graph.main_graph import run_agent
from eval.retrieval_metrics import evaluate_retrieval


def load_eval_set(path: str = "eval/questions.yaml") -> list[dict]:
    return yaml.safe_load(Path(path).read_text())


def _keywords(text: str) -> list[str]:
    return [t.strip(".,$%") for t in text.split()
            if len(t) > 4 and (t.isupper() or any(c.isdigit() for c in t))]


def score_item(item: dict, state: dict) -> dict:
    cites = state.get("citations", []) or []
    pubs = {c.get("pub") for c in cites}
    expected_pubs = {s["pub"] for s in item.get("expected_sources", [])}
    number_ok: bool | None = None
    if item.get("expected_number") is not None:
        tt = (state.get("calc_result") or {}).get("total_tax")
        number_ok = tt is not None and abs(tt - item["expected_number"]) <= (item.get("tolerance") or 0)
    answer = state.get("final_answer", "")
    kw = _keywords(item["reference_answer"])
    return {
        "id": item["id"],
        "route_ok": state.get("route") == item["route_expected"],
        "retrieval_hit": (not expected_pubs) or bool(expected_pubs & pubs),
        "has_citation": bool(cites) or item["route_expected"] in {"needs_calc", "out_of_scope"},
        "number_ok": number_ok,
        "keyword_hit": all(k in answer for k in kw) if kw else True,
    }


def _rate(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r[key] is not None]
    return round(100 * sum(bool(v) for v in vals) / len(vals), 1) if vals else 0.0


def run_eval(path: str = "eval/questions.yaml", out: str = "docs/eval-results.md",
             retrieval: bool = True) -> dict:
    from app.observability.logging import configure_logging, new_request_id, set_request_id

    configure_logging()
    items = load_eval_set(path)

    def _run(item: dict) -> dict:
        set_request_id(f"eval-{item['id']}-{new_request_id()}")
        return score_item(item, run_agent(item["question"], []))

    rows = [_run(item) for item in items]

    agg = {"n": len(rows), "route_accuracy": _rate(rows, "route_ok"),
           "retrieval_hit_rate": _rate(rows, "retrieval_hit"), "citation_rate": _rate(rows, "has_citation"),
           "numeric_accuracy": _rate(rows, "number_ok"), "keyword_hit_rate": _rate(rows, "keyword_hit")}

    retr = evaluate_retrieval(items, k=get_settings().search_k) if retrieval else {"rows": [], "aggregate": {}}

    lines = ["# Functional Evaluation Results", "", f"Items: {agg['n']}", "",
             "| id | route_ok | retrieval_hit | has_citation | number_ok | keyword_hit |",
             "|----|----------|---------------|--------------|-----------|-------------|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['route_ok']} | {r['retrieval_hit']} | "
                     f"{r['has_citation']} | {r['number_ok']} | {r['keyword_hit']} |")
    lines += ["", "## Aggregate", ""] + [f"- **{k}**: {v}" for k, v in agg.items()]

    if retr["aggregate"]:
        ra = retr["aggregate"]
        lines += ["", f"## Retrieval quality (RAG subgraph, k={ra['k']}, "
                  f"{ra['n_labeled']} page-labeled questions)", "",
                  "| id | P@k | R@k | MRR | context_precision | RR fused | RR reranked |",
                  "|----|-----|-----|-----|-------------------|----------|-------------|"]
        def _f(v: float | None) -> str:
            return "-" if v is None else f"{v:.3f}"

        for r in retr["rows"]:
            lines.append(
                f"| {r['id']} | {_f(r['precision_at_k'])} | {_f(r['recall_at_k'])} | {_f(r['mrr'])} | "
                f"{_f(r['context_precision'])} | {_f(r['rr_fused'])} | {_f(r['rr_reranked'])} |")
        lines += ["", "### Aggregate", ""] + [f"- **{k}**: {v}" for k, v in ra.items()]

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n")
    Path(out).with_suffix(".json").write_text(
        json.dumps({"functional": {"aggregate": agg, "rows": rows},
                    "retrieval": retr}, indent=2) + "\n")

    result = {**agg, "retrieval": retr["aggregate"]}
    _push_to_gateway(agg, retr["aggregate"])
    return result


def _push_to_gateway(functional: dict, retrieval: dict) -> None:
    """Publish the run's headline numbers as gauges so Grafana can show the
    last offline-eval result next to the live traffic metrics. No-op unless
    PROM_PUSHGATEWAY is set (e.g. "pushgateway:9091")."""
    target = os.getenv("PROM_PUSHGATEWAY")
    if not target:
        return
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    reg = CollectorRegistry()
    flat = {f"rag_eval_{k}": v for k, v in functional.items() if isinstance(v, (int, float))}
    flat.update({f"rag_eval_{k}": v for k, v in retrieval.items() if isinstance(v, (int, float))})
    for name, value in flat.items():
        Gauge(name, f"Offline eval: {name}", registry=reg).set(value)
    try:
        push_to_gateway(target, job="agentic_rag_eval", registry=reg)
    except Exception as e:  # noqa: BLE001 - eval must not fail because Grafana is down
        print(f"pushgateway push failed ({e})")


if __name__ == "__main__":
    print(run_eval())
