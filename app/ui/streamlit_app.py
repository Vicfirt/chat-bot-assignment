from __future__ import annotations

import httpx
import streamlit as st

from app.config import get_settings


def call_api(question: str, history: list[dict], api_url: str,
             client: httpx.Client | None = None) -> dict:
    owns = client is None
    client = client or httpx.Client(timeout=120)
    try:
        resp = client.post(f"{api_url}/chat",
                           json={"question": question, "chat_history": history})
        resp.raise_for_status()
        return resp.json()
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

    with st.spinner("Running agent..."):
        try:
            data = call_api(prompt, st.session_state.history[:-1], settings.api_url)
        except Exception as e:  # noqa: BLE001
            st.error(f"API error: {e}")
            return

    answer = data["answer"]
    if data.get("low_confidence"):
        answer = "⚠️ Low confidence.\n\n" + answer
    st.chat_message("assistant").write(answer)
    st.session_state.history.append({"role": "assistant", "content": answer})
    render_trace(data.get("steps", []), data.get("citations", []), data.get("route", "?"))


try:
    main()
except st.errors.StreamlitAPIException:
    pass
