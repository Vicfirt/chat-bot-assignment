from __future__ import annotations

from pathlib import Path

import yaml

from app.graph.main_graph import run_agent


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


def run_eval(path: str = "eval/questions.yaml", out: str = "docs/eval-results.md") -> dict:
    items = load_eval_set(path)
    rows = [score_item(item, run_agent(item["question"], [])) for item in items]

    def rate(key: str) -> float:
        vals = [r[key] for r in rows if r[key] is not None]
        return round(100 * sum(bool(v) for v in vals) / len(vals), 1) if vals else 0.0

    agg = {"n": len(rows), "route_accuracy": rate("route_ok"),
           "retrieval_hit_rate": rate("retrieval_hit"), "citation_rate": rate("has_citation"),
           "numeric_accuracy": rate("number_ok"), "keyword_hit_rate": rate("keyword_hit")}

    lines = ["# Functional Evaluation Results", "", f"Items: {agg['n']}", "",
             "| id | route_ok | retrieval_hit | has_citation | number_ok | keyword_hit |",
             "|----|----------|---------------|--------------|-----------|-------------|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['route_ok']} | {r['retrieval_hit']} | "
                     f"{r['has_citation']} | {r['number_ok']} | {r['keyword_hit']} |")
    lines += ["", "## Aggregate", ""] + [f"- **{k}**: {v}" for k, v in agg.items()]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n")
    return agg


if __name__ == "__main__":
    print(run_eval())
