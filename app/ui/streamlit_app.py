from __future__ import annotations

import json

import httpx
import streamlit as st

from app.config import get_settings


def call_api(question: str, history: list[dict], api_url: str,
             client: httpx.Client | None = None) -> dict:
    owns = client is None
    client = client or httpx.Client(timeout=300)
    try:
        resp = client.post(f"{api_url}/chat",
                           json={"question": question, "chat_history": history})
        resp.raise_for_status()
        return resp.json()
    finally:
        if owns:
            client.close()


def stream_api(question: str, history: list[dict], api_url: str,
               client: httpx.Client | None = None):
    """Yield parsed SSE events from /chat/stream: {"type": "step"|"final"|"error", ...}."""
    owns = client is None
    client = client or httpx.Client(timeout=300)
    try:
        with client.stream("POST", f"{api_url}/chat/stream",
                           json={"question": question, "chat_history": history}) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line and line.startswith("data: "):
                    yield json.loads(line[6:])
    finally:
        if owns:
            client.close()


def render_trace(steps: list[dict], citations: list[dict], route: str) -> None:
    with st.expander("Agent trace", expanded=True):
        st.markdown(f"**Route:** `{route}`")
        for s in steps:
            st.markdown(f"- **{s['node']}** ({s['duration_ms']} ms) — {s['summary']}")
        if citations:
            st.markdown("**Citations**")
            for c in citations:
                st.markdown(
                    f"- {c.get('pub','?')} — {c.get('section','')} "
                    f"(p.{c.get('page','?')}) — _{c.get('quote','')}_"
                )
        if steps:
            st.bar_chart({s["node"]: s["duration_ms"] for s in steps})


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title="Agentic RAG Tax Chatbot", layout="wide")
    st.title("Agentic RAG Tax Chatbot")
    st.caption(f"LLM mode: `{settings.llm_mode}` · model: `{settings.llm_model}` · "
               f"tax year: {settings.tax_year}")
    st.info("General information, not tax advice.")

    if "history" not in st.session_state:
        st.session_state.history = []

    for m in st.session_state.history:
        st.chat_message(m["role"]).write(m["content"])

    prompt = st.chat_input("Ask about U.S. federal income tax...")
    if not prompt:
        return

    st.chat_message("user").write(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})

    steps: list[dict] = []
    final: dict | None = None
    live = st.status("Running agent...", expanded=True)
    try:
        for ev in stream_api(prompt, st.session_state.history[:-1], settings.api_url):
            if ev["type"] == "step":
                steps.append(ev)
                live.write(f"✓ **{ev['node']}** ({ev['duration_ms']} ms) — {ev['summary']}")
            elif ev["type"] == "final":
                final = ev
            elif ev["type"] == "error":
                live.update(label="Agent error", state="error")
                st.error(ev["message"])
                return
    except Exception as e:  # noqa: BLE001
        live.update(label="API error", state="error")
        st.error(f"API error: {e}")
        return
    live.update(label=f"Done in {(final or {}).get('total_ms', 0)} ms", state="complete")

    data = final or {}
    answer = data.get("answer", "")
    if data.get("low_confidence"):
        answer = "⚠️ Low confidence.\n\n" + answer
    st.chat_message("assistant").write(answer)
    st.session_state.history.append({"role": "assistant", "content": answer})
    render_trace(steps, data.get("citations", []), data.get("route", "?"))


try:
    main()
except st.errors.StreamlitAPIException:
    pass
