"""
EdAdvisor GenAI — Streamlit Chat Interface

Run: streamlit run ui/app.py
"""
from __future__ import annotations

import uuid

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="EdAdvisor", page_icon="🎓", layout="centered")

# ── Session state setup ──────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🎓 EdAdvisor")
st.caption("Ask me about university admissions, visas, scholarships, and programme requirements.")

col1, col2 = st.columns([4, 1])
with col2:
    if st.button("Clear chat", use_container_width=True):
        try:
            requests.delete(f"{API_URL}/v1/session/{st.session_state.session_id}", timeout=3)
        except Exception:
            pass
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# ── Render history ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 Sources ({len(msg['sources'])})", expanded=False):
                for s in msg["sources"]:
                    label = f"[Source {s['source_n']}] **{s['source']}**"
                    if s.get("section"):
                        label += f" — {s['section']}"
                    if s.get("page"):
                        label += f" (p.{s['page']})"
                    st.markdown(label)
                    st.caption(s["excerpt"])
        if msg.get("confidence") is not None:
            conf = msg["confidence"]
            colour = "🟢" if conf > 0.75 else ("🟡" if conf > 0.55 else "🔴")
            st.caption(f"{colour} Confidence: {conf:.0%}")
        if msg.get("escalated"):
            st.info("⚠️ This query was escalated — please verify with official sources or a counsellor.")

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about studying abroad…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            try:
                resp = requests.post(
                    f"{API_URL}/v1/chat",
                    json={"session_id": st.session_state.session_id, "question": prompt},
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    confidence = data.get("confidence", 0)
                    escalated = data.get("escalated", False)
                elif resp.status_code == 503:
                    answer = "⚠️ The knowledge base is not loaded. Please run `make ingest` first."
                    sources, confidence, escalated = [], 0, False
                else:
                    answer = f"Something went wrong (HTTP {resp.status_code}). Please try again."
                    sources, confidence, escalated = [], 0, False
            except requests.exceptions.ConnectionError:
                answer = "Cannot connect to the API. Make sure `make serve` is running."
                sources, confidence, escalated = [], 0, False

        st.markdown(answer)

        if sources:
            with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
                for s in sources:
                    label = f"[Source {s['source_n']}] **{s['source']}**"
                    if s.get("section"):
                        label += f" — {s['section']}"
                    if s.get("page"):
                        label += f" (p.{s['page']})"
                    st.markdown(label)
                    st.caption(s["excerpt"])

        if confidence:
            colour = "🟢" if confidence > 0.75 else ("🟡" if confidence > 0.55 else "🔴")
            st.caption(f"{colour} Confidence: {confidence:.0%}")

        if escalated:
            st.info("⚠️ This query was escalated — please verify with official sources.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources if "sources" in dir() else [],
        "confidence": confidence if "confidence" in dir() else None,
        "escalated": escalated if "escalated" in dir() else False,
    })
