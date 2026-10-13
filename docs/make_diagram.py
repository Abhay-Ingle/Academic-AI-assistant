"""Generates docs/architecture.png  (python docs/make_diagram.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(0, 15); ax.set_ylim(0, 9); ax.axis("off")

def box(x, y, w, h, text, fc, ec="#333", fs=8.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)

def arrow(x1, y1, x2, y2, label="", style="->", color="#444", rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.5, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, ha="center", fontsize=8, color=color)

ax.text(7.5, 8.65, "Scholar - Agentic Academic AI Assistant: Architecture", ha="center", fontsize=15, fontweight="bold")

# UI
box(0.3, 6.9, 2.3, 1.1, "Student\n(Streamlit UI / CLI)\nchat · upload · toggles", "#E8F0FE", bold=True)
# agent pipeline (left column)
box(3.0, 7.0, 3.0, 0.9, "1. INPUT GUARDRAILS\ninjection · cheating · self-harm\nPII masking · length", "#FDE7E9")
box(3.0, 5.6, 3.0, 1.0, "2. PLANNER (LLM)\nintent + needs_tools?\nJSON plan, few-shot", "#FFF4D6")
box(3.0, 3.9, 3.0, 1.3, "3. EXECUTOR (ReAct loop)\nthought → action → observation\n@obs_N references · loop guard\nmax-steps · re-plan on failure", "#FFF4D6")
box(3.0, 2.5, 3.0, 0.9, "4. SYNTHESIZER (LLM)\ngrounded answer + citations", "#FFF4D6")
box(3.0, 1.3, 3.0, 0.9, "5. REFLECTOR (LLM)\nself-critique → revise if needed", "#FFF4D6")
box(3.0, 0.2, 3.0, 0.8, "6. OUTPUT GUARD\nkey redaction · empty-answer fallback", "#FDE7E9")

arrow(2.6, 7.45, 3.0, 7.45)
arrow(4.5, 7.0, 4.5, 6.6)
arrow(4.5, 5.6, 4.5, 5.2, "needs tools")
arrow(4.5, 3.9, 4.5, 3.4)
arrow(4.5, 2.5, 4.5, 2.2)
arrow(4.5, 1.3, 4.5, 1.0)
arrow(3.0, 0.6, 1.4, 0.6, color="#444"); arrow(1.4, 0.6, 1.4, 6.9, "final answer + trace + files")
ax.text(1.4, 5.9, "no tools needed:\nplanner → direct answer\n(1 LLM call, skips 3-5)", fontsize=7.5, color="#666", ha="center")

# tools layer
ax.add_patch(FancyBboxPatch((6.8, 2.2), 4.4, 5.8, boxstyle="round,pad=0.03,rounding_size=0.15", fc="#F3F8F1", ec="#5A8F4E", lw=1.5))
ax.text(9.0, 7.7, "TOOL LAYER (13 tools, allow-listed)", ha="center", fontsize=10, fontweight="bold")
box(7.0, 6.5, 4.0, 0.9, "Knowledge (RAG)\nsearch_notes · summarize_text", "#DFF0D8")
box(7.0, 5.4, 4.0, 0.9, "Research & web\nsearch_arxiv · wikipedia_summary\nweb_search", "#DFF0D8")
box(7.0, 4.3, 4.0, 0.9, "Computation (deterministic)\nmake_study_plan · calculate_cgpa\nrequired_gpa", "#DFF0D8")
box(7.0, 3.2, 4.0, 0.9, "Learning\ngenerate_quiz (grounded in notes)", "#DFF0D8")
box(7.0, 2.35, 4.0, 0.7, "Actions (need permission)\ncreate_calendar_file · add_deadline\nsave_note", "#F8E3C5")
arrow(6.0, 4.65, 6.8, 4.65, "call")
arrow(6.8, 4.2, 6.0, 4.2, color="#5A8F4E")
ax.text(6.4, 3.95, "result", fontsize=8, color="#5A8F4E", ha="center")

# right column
box(11.9, 6.3, 2.8, 1.5, "RAG KNOWLEDGE BASE\nPDF/DOCX/TXT → chunks\nBM25 + MiniLM/Chroma\nRRF fusion", "#E3F2FD", bold=False)
box(11.9, 4.4, 2.8, 1.4, "MEMORY (JSON)\nstudent profile\ndeadlines · saved notes", "#E3F2FD")
box(11.9, 2.5, 2.8, 1.5, "EXTERNAL APIs\narXiv · Wikipedia\nDuckDuckGo", "#E3F2FD")
box(11.9, 0.5, 2.8, 1.5, "LLM PROVIDER (LangChain)\nGroq · Gemini · OpenAI\nOllama (local)", "#EDE7F6", bold=True)
arrow(11.0, 6.95, 11.9, 6.95)
arrow(11.0, 4.9, 11.9, 4.9)
arrow(11.0, 5.85, 11.9, 3.4, rad=-0.15)
ax.annotate("", xy=(11.9, 1.2), xytext=(5.9, 1.75), arrowprops=dict(arrowstyle="<->", color="#7E57C2", lw=1.3, ls="--", connectionstyle="arc3,rad=0.12"))
ax.text(8.4, 0.75, "Planner · Executor · Synthesizer · Reflector · quiz/summary tools call the LLM", fontsize=8, color="#7E57C2", ha="center")

plt.savefig("docs/architecture.png", dpi=170, bbox_inches="tight", facecolor="white")
print("saved")
