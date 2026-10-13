"""Streamlit web UI:  streamlit run app.py"""
from pathlib import Path

import streamlit as st

from agent import config
from agent.factory import create_agent

st.set_page_config(page_title="Scholar - Academic AI Agent", page_icon="🎓", layout="wide")
config.ensure_dirs()


@st.cache_resource(show_spinner="Loading agent (first run downloads the embedding model)...")
def get_agent():
    return create_agent()


def render_trace(resp):
    with st.expander("🧠 Agent reasoning: plan → tools → answer", expanded=False):
        st.caption(f"Intent: **{resp.intent or '-'}** · LLM calls: **{resp.llm_calls}** · Time: **{resp.seconds}s**")
        for ev in resp.trace:
            k = ev["kind"]
            if k == "input_guard":
                st.write(f"🛡️ Input guard: `{ev['category']}`" + (" (personal data masked)" if ev.get("pii_found") else ""))
            elif k == "plan":
                st.write("📋 **Plan** – " + (ev.get("reasoning") or ""))
                for s in ev["plan"]:
                    st.write(f"&nbsp;&nbsp;{s['step']}. {s['goal']} → `{s['tool'] or 'answer'}`")
            elif k == "thought":
                st.write(f"💭 {ev['text']}")
            elif k == "tool":
                st.write(f"🔧 `{ev['tool']}` {'✅' if ev['ok'] else '❌'}")
                st.code(f"args: {ev['args']}\n{ev['preview']}", language="text")
            elif k in ("blocked", "warning", "replan"):
                st.write(f"⚠️ {k}: {ev}")
            elif k == "reflection":
                st.write("🔍 Self-check: " + ("passed" if ev["ok"] else f"revised ({ev['issues']})"))


agent = get_agent()

# ----------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.title("🎓 Scholar")
    st.caption(f"LLM: {config.LLM_PROVIDER} · Retrieval: {'hybrid (BM25 + dense)' if agent.kb.dense_enabled else 'BM25'}")

    st.subheader("📚 Knowledge base")
    files = st.file_uploader("Upload notes / papers (PDF, DOCX, TXT, MD)", type=["pdf", "docx", "txt", "md"],
                             accept_multiple_files=True)
    c1, c2 = st.columns(2)
    if c1.button("Index files", disabled=not files):
        added = 0
        for f in files:
            dest = config.UPLOAD_DIR / Path(f.name).name
            dest.write_bytes(f.getbuffer())
            added += agent.kb.add_file(dest)
        st.success(f"Indexed {added} new chunks.")
    if c2.button("Load samples"):
        st.success(f"Indexed {agent.kb.add_folder(config.SAMPLE_DIR)} new chunks.")
    st.write("Indexed:", ", ".join(agent.kb.sources()) or "nothing yet")

    st.subheader("👤 Profile")
    p = agent.memory.data["profile"]
    name = st.text_input("Name", p.get("name", ""))
    course = st.text_input("Course / semester", p.get("course", ""))
    hours = st.number_input("Study hours per day", 0.5, 12.0, float(p.get("study_hours_per_day", 3)), 0.5)
    if st.button("Save profile"):
        agent.memory.set_profile(name=name, course=course, study_hours_per_day=hours)
        st.success("Saved")

    st.subheader("⏰ Deadlines")
    for d in agent.memory.list_deadlines(config.today(), 60):
        st.write(f"• **{d['title']}** – {d['due_date']} ({d['days_left']}d)")

    st.subheader("⚙️ Safety & performance")
    agent.allow_side_effects = st.toggle("Allow agent to create files / save data", value=True,
                                         help="Human-in-the-loop switch for .ics files, notes and deadlines.")
    agent.enable_reflection = st.toggle("Self-check answers (slower, safer)", value=config.ENABLE_REFLECTION)
    agent.max_steps = st.slider("Max tool steps per request", 1, 10, agent.max_steps)
    if st.button("Clear conversation"):
        agent.history.clear()
        st.session_state.pop("messages", None)
        st.rerun()

# -------------------------------------------------------------------- chat ----
st.header("Academic AI Assistant")
st.caption("Try: “Make a study plan for DBMS exam on 2026-11-20 (normalization, transactions, indexing) and add it to my calendar”")

if "messages" not in st.session_state:
    st.session_state.messages = []

for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("resp"):
            render_trace(m["resp"])
            for fpath in m["resp"].files:
                fp = Path(fpath)
                if fp.exists():
                    st.download_button(f"⬇️ {fp.name}", fp.read_bytes(), file_name=fp.name, key=f"dl{i}{fp.name}")

if prompt := st.chat_input("Ask Scholar anything about your studies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.status("Agent working...", expanded=True) as status:
            def live(ev):
                if ev["kind"] == "plan":
                    status.write("📋 Planned: " + " → ".join(s["tool"] or "answer" for s in ev["plan"]) if ev["plan"] else "📋 Answering directly")
                elif ev["kind"] == "tool":
                    status.write(f"🔧 {ev['tool']} {'✅' if ev['ok'] else '❌'}")
            agent.on_event = live
            resp = agent.run(prompt)
            status.update(label="Done", state="complete", expanded=False)
        st.markdown(resp.answer)
        if resp.sources:
            st.caption("Sources: " + " · ".join(resp.sources))
        render_trace(resp)
        for fpath in resp.files:
            fp = Path(fpath)
            if fp.exists():
                st.download_button(f"⬇️ {fp.name}", fp.read_bytes(), file_name=fp.name, key=f"new{fp.name}")
    st.session_state.messages.append({"role": "assistant", "content": resp.answer, "resp": resp})
