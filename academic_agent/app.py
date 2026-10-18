"""Streamlit web UI  –  run with:  streamlit run app.py"""
from __future__ import annotations

import queue
import threading
from pathlib import Path

import streamlit as st

from agent import config
from agent.factory import create_agent

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Scholar – Academic AI Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)
config.ensure_dirs()

# ── Premium White UI CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Variables ── */
:root {
    --white:       #ffffff;
    --bg:          #f5f7fa;
    --bg-card:     #ffffff;
    --bg-hover:    #f0f4ff;
    --border:      #e2e8f0;
    --border-soft: #edf2f7;
    --accent:      #4f46e5;
    --accent-light:#ede9fe;
    --accent-2:    #7c3aed;
    --text:        #1a202c;
    --text-muted:  #718096;
    --text-light:  #a0aec0;
    --success:     #38a169;
    --success-bg:  #f0fff4;
    --warning:     #d69e2e;
    --warning-bg:  #fffbeb;
    --danger:      #e53e3e;
    --danger-bg:   #fff5f5;
    --shadow-sm:   0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md:   0 4px 12px rgba(0,0,0,0.08), 0 2px 6px rgba(0,0,0,0.04);
    --shadow-lg:   0 10px 30px rgba(0,0,0,0.1), 0 4px 12px rgba(0,0,0,0.05);
    --radius:      12px;
    --radius-sm:   8px;
    --radius-lg:   16px;
}

/* ── Global ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }
.stDeployButton { display: none !important; }

/* ── Main content area ── */
.main .block-container {
    padding: 1rem 2.5rem 5rem 2.5rem !important;
    max-width: 920px !important;
}

/* ─────────────── SIDEBAR ─────────────── */
[data-testid="stSidebar"] {
    background: var(--white) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stSidebar"] > div {
    padding-top: 1rem !important;
}
[data-testid="stSidebar"] .stMarkdown h4 {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: var(--text-light) !important;
    margin: 1.2rem 0 0.5rem 0 !important;
}
[data-testid="stSidebar"] .stDivider {
    margin: 0.8rem 0 !important;
    border-color: var(--border-soft) !important;
}

/* ── File uploader in sidebar ── */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: var(--bg) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.6rem !important;
}

/* ─────────────── BUTTONS ─────────────── */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 0.45rem 1rem !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 2px 6px rgba(79,70,229,0.3) !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    background: #4338ca !important;
    box-shadow: 0 4px 12px rgba(79,70,229,0.4) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button:disabled {
    background: var(--border) !important;
    color: var(--text-light) !important;
    box-shadow: none !important;
}

/* Sidebar buttons — secondary style */
[data-testid="stSidebar"] .stButton > button {
    background: var(--bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--bg-hover) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.1) !important;
    transform: translateY(-1px) !important;
}

/* ─────────────── CHAT MESSAGES ─────────────── */
[data-testid="stChatMessage"] {
    background: var(--white) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.8rem !important;
    box-shadow: var(--shadow-sm) !important;
    transition: box-shadow 0.2s !important;
}
[data-testid="stChatMessage"]:hover {
    box-shadow: var(--shadow-md) !important;
}

/* User message bubble — subtle blue tint */
[data-testid="stChatMessage"][data-testid*="user"],
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, #f5f3ff, #ede9fe) !important;
    border-color: rgba(79,70,229,0.2) !important;
}

/* ─────────────── CHAT INPUT ─────────────── */
[data-testid="stChatInput"] {
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-md) !important;
    border: 1.5px solid var(--border) !important;
    background: var(--white) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important;
    color: var(--text) !important;
    font-size: 0.93rem !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.12), var(--shadow-md) !important;
}

/* ─────────────── STATUS BOX ─────────────── */
[data-testid="stStatus"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-sm) !important;
    font-size: 0.86rem !important;
}

/* ─────────────── EXPANDERS ─────────────── */
details {
    background: var(--white) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: var(--radius) !important;
    margin-top: 10px !important;
    box-shadow: var(--shadow-sm) !important;
}
summary {
    font-size: 0.86rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    padding: 0.7rem 1rem !important;
}

/* ─────────────── TOGGLE / SLIDER ─────────────── */
.stToggle label { font-size: 0.84rem !important; color: var(--text) !important; }
.stSlider label { font-size: 0.84rem !important; color: var(--text) !important; }

/* ─────────────── DOWNLOAD BUTTON ─────────────── */
[data-testid="stDownloadButton"] > button {
    background: var(--success-bg) !important;
    border: 1px solid var(--success) !important;
    color: var(--success) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #c6f6d5 !important;
    transform: translateY(-1px) !important;
}

/* ─────────────── CUSTOM COMPONENTS ─────────────── */

/* Page header */
.scholar-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 1.2rem 0 1rem 0;
    border-bottom: 2px solid var(--border-soft);
    margin-bottom: 1.6rem;
}
.scholar-logo {
    width: 50px; height: 50px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    box-shadow: 0 4px 14px rgba(79,70,229,0.35);
    flex-shrink: 0;
}
.scholar-title {
    font-size: 1.7rem; font-weight: 800;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.scholar-sub { font-size: 0.83rem; color: var(--text-muted); margin-top: 3px; }

/* Sidebar brand */
.sidebar-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 0.2rem 0 1rem 0;
    border-bottom: 1px solid var(--border-soft);
    margin-bottom: 0.4rem;
}
.sidebar-logo {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px;
    box-shadow: 0 3px 10px rgba(79,70,229,0.3);
    flex-shrink: 0;
}
.sidebar-name  { font-size: 1rem; font-weight: 700; color: var(--text); }
.sidebar-tagline { font-size: 0.7rem; color: var(--text-light); }

/* Welcome banner */
.welcome-card {
    background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 50%, #e0e7ff 100%);
    border: 1.5px solid rgba(79,70,229,0.2);
    border-radius: var(--radius-lg);
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.welcome-card::before {
    content: "🎓";
    position: absolute; right: 1.5rem; top: 50%;
    transform: translateY(-50%);
    font-size: 3.5rem; opacity: 0.15;
}
.welcome-card h3 {
    margin: 0 0 8px 0; font-size: 1.15rem;
    font-weight: 700; color: #3730a3;
}
.welcome-card p {
    margin: 0; font-size: 0.87rem;
    color: #5145cd; line-height: 1.65;
    max-width: 80%;
}

/* Suggestion chip buttons — override stButton for chips */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1.4rem; }

/* Stat grid inside trace expander */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px; margin: 0.6rem 0 1rem 0;
}
.stat-card {
    background: var(--bg);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
    text-align: center;
}
.stat-val { font-size: 1.3rem; font-weight: 800; color: var(--accent); }
.stat-lbl { font-size: 0.68rem; color: var(--text-light); margin-top: 2px;
             text-transform: uppercase; letter-spacing: 0.05em; }

/* Source badge */
.src-badge {
    display: inline-flex; align-items: center; gap: 4px;
    background: var(--accent-light);
    border: 1px solid rgba(79,70,229,0.25);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.71rem;
    color: var(--accent);
    margin: 3px 4px 3px 0;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}

/* Trace step rows */
.trace-step {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 9px 0;
    border-bottom: 1px solid var(--border-soft);
    font-size: 0.83rem;
    line-height: 1.55;
}
.trace-step:last-child { border-bottom: none; }
.trace-icon { font-size: 1rem; min-width: 24px; flex-shrink: 0; padding-top: 1px; }
.trace-text { color: var(--text-muted); flex: 1; }
.trace-text code {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    color: var(--accent) !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    font-size: 0.8rem !important;
}

/* Deadline row in sidebar */
.dl-row {
    display: flex; align-items: flex-start; gap: 8px;
    padding: 7px 10px; margin: 4px 0;
    background: var(--bg);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-sm);
    font-size: 0.82rem;
}
.dl-title { font-weight: 600; color: var(--text); }
.dl-meta  { font-size: 0.74rem; color: var(--text-light); margin-top: 1px; }

/* Info pill (provider / retrieval) */
.info-pill {
    display: inline-block;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-right: 4px;
}
.info-pill b { color: var(--accent); }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Alert overrides */
.stAlert { border-radius: var(--radius-sm) !important; font-size: 0.87rem !important; }

/* Success toast */
[data-testid="stToast"] { border-radius: var(--radius-sm) !important; }

/* Quick suggestions chip row (shown during active chat) */
/* Make chip buttons lighter / smaller than primary action buttons */
[id^="chip_top_"] > button,
[id^="chip_bot_"] > button,
[id^="sug_"] > button {
    background: var(--bg) !important;
    color: var(--accent) !important;
    border: 1.5px solid rgba(79,70,229,0.3) !important;
    box-shadow: none !important;
    font-size: 0.79rem !important;
    font-weight: 500 !important;
    padding: 0.35rem 0.7rem !important;
}
[id^="chip_top_"] > button:hover,
[id^="chip_bot_"] > button:hover,
[id^="sug_"] > button:hover {
    background: var(--accent-light) !important;
    border-color: var(--accent) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.15) !important;
}

/* "More suggestions" expander — keep it visually lightweight */
.more-sug-expander details {
    border: 1px dashed var(--border) !important;
    background: transparent !important;
    box-shadow: none !important;
    border-radius: var(--radius-sm) !important;
}
</style>
""", unsafe_allow_html=True)

# ── load agent ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Loading Scholar — just a moment…")
def get_agent():
    return create_agent()

agent = get_agent()


# ── helper functions ──────────────────────────────────────────────────────────
def render_trace(resp):
    with st.expander("🧠 Reasoning trace — click to expand", expanded=False):
        st.markdown(
            f"""<div class="stat-grid">
              <div class="stat-card">
                <div class="stat-val">{resp.llm_calls}</div>
                <div class="stat-lbl">LLM calls</div>
              </div>
              <div class="stat-card">
                <div class="stat-val">{resp.seconds}s</div>
                <div class="stat-lbl">Time</div>
              </div>
              <div class="stat-card">
                <div class="stat-val">{len(resp.tools_used)}</div>
                <div class="stat-lbl">Tools used</div>
              </div>
              <div class="stat-card">
                <div class="stat-val" style="font-size:0.95rem">{resp.intent or "—"}</div>
                <div class="stat-lbl">Intent</div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
        rows = []
        for ev in resp.trace:
            k = ev["kind"]
            if k == "input_guard":
                icon = "🛡️"
                txt = f"Input guard — category: <b>{ev['category']}</b>" + \
                      (" · PII detected &amp; masked" if ev.get("pii_found") else "")
            elif k == "plan":
                steps = " → ".join(s.get("tool") or "answer" for s in ev.get("plan", [])) or "direct answer"
                icon = "📋"
                txt  = f"Plan: <b>{steps}</b>"
            elif k == "thought":
                icon = "💭"
                txt  = ev.get("text", "")[:220]
            elif k == "tool":
                status = "✅" if ev["ok"] else "❌"
                icon = "🔧"
                txt  = f"<code>{ev['tool']}</code> {status} &nbsp;— {ev.get('preview','')[:200]}"
            elif k == "reflection":
                icon = "🔍"
                txt  = "Self-check: " + ("passed ✓" if ev["ok"] else f"revised — {ev.get('issues', [])}")
            elif k in ("blocked", "warning", "replan"):
                icon = "⚠️"
                txt  = f"<b>{k}</b>: {ev}"
            else:
                continue
            rows.append(
                f'<div class="trace-step">'
                f'<div class="trace-icon">{icon}</div>'
                f'<div class="trace-text">{txt}</div>'
                f'</div>'
            )
        if rows:
            st.markdown("".join(rows), unsafe_allow_html=True)


def render_sources(sources: list[str]):
    if not sources:
        return
    badges = "".join(f'<span class="src-badge">🔗 {s}</span>' for s in sources)
    st.markdown(f'<div style="margin-top:8px;">{badges}</div>', unsafe_allow_html=True)


def run_agent_threaded(prompt: str, q: queue.Queue):
    def on_event(ev):
        q.put(("event", ev))
    agent.on_event = on_event
    try:
        resp = agent.run(prompt)
        q.put(("done", resp))
    except Exception as exc:
        q.put(("error", str(exc)))


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sidebar-brand">
      <div class="sidebar-logo">🎓</div>
      <div>
        <div class="sidebar-name">Scholar</div>
        <div class="sidebar-tagline">Academic AI Agent</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Knowledge Base ──────────────────────────────────────────────────────
    st.markdown("#### 📚 Knowledge Base")
    uploaded = st.file_uploader(
        "Drop PDFs, DOCX, TXT or MD files here",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    col1, col2 = st.columns(2)
    if col1.button("⬆ Index files", disabled=not uploaded, use_container_width=True):
        new_total, skip_total, fail_list = 0, 0, []
        with st.spinner("Indexing…"):
            for f in uploaded:
                dest = config.UPLOAD_DIR / Path(f.name).name
                dest.write_bytes(f.getbuffer())
                # count chunks before and after to detect duplicates vs failures
                before = len(agent.kb.chunks)
                try:
                    added = agent.kb.add_file(dest)
                except Exception as exc:
                    fail_list.append(f"{f.name}: {exc}")
                    continue
                after = len(agent.kb.chunks)
                if added == 0 and after == before:
                    # Check whether file actually has extractable text
                    from agent.rag import load_file as _lf
                    try:
                        pages = _lf(dest)
                        has_text = any(t.strip() for t, _ in pages)
                    except Exception:
                        has_text = False
                    if not has_text:
                        fail_list.append(f"{f.name} — no extractable text (scanned/image PDF?)")
                    else:
                        skip_total += 1  # file already indexed
                else:
                    new_total += added
        # Show results
        if new_total > 0:
            st.success(f"✅ Added {new_total} new chunk{'s' if new_total != 1 else ''} to knowledge base.")
        if skip_total > 0:
            st.info(f"ℹ️ {skip_total} file{'s' if skip_total != 1 else ''} already indexed — no duplicates added.")
        if fail_list:
            for msg in fail_list:
                st.warning(f"⚠️ {msg}")
        if new_total == 0 and skip_total == 0 and not fail_list:
            st.warning("⚠️ No chunks were added. The file may be empty.")

    if col2.button("📖 Samples", use_container_width=True):
        n = agent.kb.add_folder(config.SAMPLE_DIR)
        if n > 0:
            st.success(f"✅ {n} sample chunk{'s' if n != 1 else ''} added.")
        else:
            st.info("ℹ️ Sample notes already indexed — nothing new to add.")

    indexed = agent.kb.sources()
    if indexed:
        docs_html = "".join(
            f'<div style="display:flex;align-items:center;gap:6px;padding:3px 0;">'
            f'<span style="color:#4f46e5;font-size:0.85rem;">📄</span>'
            f'<span style="font-size:0.79rem;color:#4a5568;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{s}</span>'
            f'</div>'
            for s in indexed
        )
        st.markdown(
            f'<div style="background:#f8faff;border:1px solid #e2e8f0;border-radius:8px;'
            f'padding:8px 10px;margin-top:6px;">{docs_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p style="font-size:0.8rem;color:#a0aec0;margin:6px 0 0 0;">No documents indexed yet.</p>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Deadlines ───────────────────────────────────────────────────────────
    st.markdown("#### ⏰ Upcoming Deadlines")
    deadlines = agent.memory.list_deadlines(config.today(), 60)
    if deadlines:
        for d in deadlines:
            if d["days_left"] <= 3:
                dot, color = "🔴", "#fff5f5"
            elif d["days_left"] <= 7:
                dot, color = "🟡", "#fffbeb"
            else:
                dot, color = "🟢", "#f0fff4"
            st.markdown(
                f'<div class="dl-row" style="background:{color};">'
                f'  <span style="font-size:0.9rem;flex-shrink:0">{dot}</span>'
                f'  <div>'
                f'    <div class="dl-title">{d["title"]}</div>'
                f'    <div class="dl-meta">{d["due_date"]} &nbsp;·&nbsp; {d["days_left"]} days left'
                f'{" · " + d["course"] if d.get("course") else ""}</div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<p style="font-size:0.8rem;color:#a0aec0;margin:4px 0;">No deadlines saved yet.</p>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Settings ────────────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Settings")

    agent.allow_side_effects = st.toggle(
        "Allow file & memory writes",
        value=True,
        help="Enables saving .ics files, notes and deadlines to disk.",
    )
    agent.enable_reflection = st.toggle(
        "Self-check answers  (+1 LLM call)",
        value=False,
        help="Agent re-reads its own answer before sending — slower but more accurate.",
    )
    agent.max_steps = st.slider(
        "Max tool steps per request", min_value=1, max_value=10, value=agent.max_steps
    )

    if st.button("🗑️  Clear conversation", use_container_width=True):
        agent.history.clear()
        st.session_state.pop("messages", None)
        st.rerun()

    st.markdown(
        f'<div style="margin-top:14px;">'
        f'<span class="info-pill">Model: <b>{agent.llm.current_model}</b></span>'
        f'<span class="info-pill">RAG: <b>{"hybrid" if agent.kb.dense_enabled else "BM25"}</b></span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── MAIN AREA ─────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="scholar-header">
  <div class="scholar-logo">🎓</div>
  <div>
    <div class="scholar-title">Scholar</div>
    <div class="scholar-sub">Your AI-powered academic assistant — ask anything about your studies</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

SUGGESTIONS = [
    ("📅", "Study plan for DBMS exam on 2026-11-20 (normalization, transactions, indexing)"),
    ("📝", "Generate a 5-question quiz on CPU scheduling (medium difficulty)"),
    ("📊", "Calculate my CGPA — I have 3 courses"),
    ("🔍", "Find recent arXiv papers on retrieval-augmented generation"),
    ("⏰", "List all my upcoming deadlines"),
    ("📚", "Summarise my uploaded notes as bullet points"),
]

# ── Welcome banner (only on first load) ──────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
      <h3>👋 Welcome to Scholar!</h3>
      <p>I can build personalised study plans, search research papers, generate quizzes,
         calculate your CGPA, track deadlines, and answer questions from your uploaded notes.
         Pick a suggestion below or type your own question.</p>
    </div>
    """, unsafe_allow_html=True)

# ── Suggestion chips — ALWAYS visible ────────────────────────────────────────
# Show as compact row when chat is active, full grid when empty
if st.session_state.messages:
    st.markdown(
        '<p style="font-size:0.75rem;font-weight:600;color:#a0aec0;'
        'text-transform:uppercase;letter-spacing:0.06em;margin:0.5rem 0 0.4rem 0;">'
        '⚡ Quick suggestions</p>',
        unsafe_allow_html=True,
    )
    # Compact single row of 3 chips when chat is ongoing
    chip_cols = st.columns(3)
    for i, (emoji, text) in enumerate(SUGGESTIONS[:3]):
        short = text[:38] + "…" if len(text) > 38 else text
        if chip_cols[i].button(f"{emoji} {short}", key=f"chip_top_{i}", use_container_width=True):
            st.session_state["prefill"] = f"{emoji} {text}"
            st.rerun()
    with st.expander("More suggestions…", expanded=False):
        ex_cols = st.columns(3)
        for i, (emoji, text) in enumerate(SUGGESTIONS[3:], start=3):
            short = text[:38] + "…" if len(text) > 38 else text
            if ex_cols[(i - 3) % 3].button(f"{emoji} {short}", key=f"chip_bot_{i}", use_container_width=True):
                st.session_state["prefill"] = f"{emoji} {text}"
                st.rerun()
else:
    # Full 3-column grid on welcome screen
    c1, c2, c3 = st.columns(3)
    grid_cols = [c1, c2, c3]
    for i, (emoji, text) in enumerate(SUGGESTIONS):
        label = f"{emoji} {text[:46]}{'…' if len(text) > 46 else ''}"
        if grid_cols[i % 3].button(label, key=f"sug_{i}", use_container_width=True):
            st.session_state["prefill"] = f"{emoji} {text}"
            st.rerun()


# ── Render chat history ───────────────────────────────────────────────────────
for idx, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("resp"):
            render_sources(m["resp"].sources)
            render_trace(m["resp"])
            for fpath in m["resp"].files:
                fp = Path(fpath)
                if fp.exists():
                    st.download_button(
                        f"⬇️  Download {fp.name}",
                        fp.read_bytes(),
                        file_name=fp.name,
                        key=f"dl_{idx}_{fp.name}",
                    )


# ── Chat input ────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", None)
prompt  = st.chat_input("Ask Scholar anything about your studies…") or prefill

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        q      = queue.Queue()
        thread = threading.Thread(target=run_agent_threaded, args=(prompt, q), daemon=True)
        thread.start()

        resp = None
        with st.status("🤔 Scholar is thinking…", expanded=True) as status_box:
            while True:
                try:
                    msg_type, payload = q.get(timeout=120)
                except queue.Empty:
                    status_box.update(label="⏱️ Timed out — please try again", state="error")
                    break

                if msg_type == "event":
                    ev = payload
                    k  = ev["kind"]
                    if k == "plan":
                        if ev.get("plan"):
                            steps = " → ".join(s.get("tool") or "answer" for s in ev["plan"])
                            status_box.write(f"📋 **Plan:** {steps}")
                        else:
                            status_box.write("💬 Answering directly from knowledge…")
                    elif k == "tool":
                        icon = "✅" if ev["ok"] else "❌"
                        status_box.write(f"🔧 Running **{ev['tool']}** {icon}")
                    elif k == "thought":
                        t = ev.get("text", "")[:120]
                        if t:
                            status_box.write(f"💭 {t}")
                    elif k == "reflection":
                        status_box.write("🔍 Self-checking the answer…")

                elif msg_type == "done":
                    resp = payload
                    status_box.update(label="✅ Answer ready!", state="complete", expanded=False)
                    break

                elif msg_type == "error":
                    st.error(f"❌ Something went wrong: {payload}")
                    status_box.update(label="❌ Error", state="error")
                    break

        if resp:
            st.markdown(resp.answer)
            render_sources(resp.sources)
            render_trace(resp)
            for fpath in resp.files:
                fp = Path(fpath)
                if fp.exists():
                    st.download_button(
                        f"⬇️  Download {fp.name}",
                        fp.read_bytes(),
                        file_name=fp.name,
                        key=f"new_{fp.name}",
                    )
            st.session_state.messages.append(
                {"role": "assistant", "content": resp.answer, "resp": resp}
            )
