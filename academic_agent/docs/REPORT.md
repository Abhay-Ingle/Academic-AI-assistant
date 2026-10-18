# Scholar: An Agentic AI Academic Assistant

**Course activity:** FA – Agentic AI Application Implementation  **Domain:** AI Academic Assistant
**Student(s):** ______________  **Roll No.:** ______  **Date:** October 2026
**Repository:** ______________ (GitHub / ZIP)

---

## 1. Introduction
College students juggle lectures, notes, assignments, exams and research, and lose time on routine academic
administration: building study timetables, hunting through PDFs, finding papers, calculating grades and tracking
deadlines. *Scholar* is an **agentic** AI assistant that automates these tasks. Rather than answering a single
prompt, it *plans* a solution, *selects and calls tools* (retrieval, research APIs, calculators, calendar export),
*feeds results from one step into the next*, *checks its own answer* and *remembers* the student's deadlines and profile.

## 2. Problem Statement
Students need a trusted assistant that can (a) answer from **their own course material** with sources, (b) turn a vague
goal such as "prepare me for my DBMS exam" into concrete, scheduled actions, and (c) do so safely, without
hallucinating, leaking personal data or helping with academic dishonesty. A plain chatbot cannot read local
notes, cannot create a calendar file, has no memory of deadlines and often invents citations; therefore an agent
with planning, tools and retrieval is required.

## 3. Objectives
1. Build a multi-step agent that plans and executes tasks autonomously with tool use.
2. Ground answers in the student's uploaded notes/PDFs using RAG with visible citations.
3. Automate study planning, deadline tracking, calendar export, quiz generation, paper search and GPA calculation.
4. Reduce time and effort: one request → complete result (e.g. plan + .ics file).
5. Enforce safety: prompt-injection defence, academic integrity, privacy, bounded autonomy, human control.
6. Keep it efficient: minimal LLM calls, deterministic tools where possible, caching, free/local model support.

## 4. Proposed Agent Architecture
*(see `architecture.png`)* The agent is a **Plan-and-Execute architecture with a ReAct executor and a reflection loop**:

| Stage | Role |
|---|---|
| 1. Input guardrails | blocks injection / cheating / self-harm, masks PII, length limit |
| 2. Planner (LLM) | classifies intent, decides if tools are needed, produces a JSON plan (≤ 6 steps) |
| 3. Executor (LLM, ReAct) | each step: thought → action(tool, args) → observation; can reference earlier results with `@obs_N`; loop guard, max-steps, one automatic re-plan after repeated failures |
| 4. Synthesizer (LLM) | writes the final answer using only observations, with citations |
| 5. Reflector (LLM) | critiques the draft (unsupported claims, missed parts, dishonesty) and revises if needed |
| 6. Output guard | redacts API keys, never returns an empty answer |

Supporting components: **Tool layer** (13 tools), **RAG knowledge base**, **persistent memory** (profile, deadlines, notes), and a
**provider-agnostic LLM wrapper** built on LangChain (Groq, Gemini, OpenAI or local Ollama). Simple requests skip stages 3–5
(planner decides `needs_tools=false`), saving cost and latency.

## 5. Prompt Design
Prompts live in `agent/prompts.py`. Techniques applied:
- **Role + principles prompting** – "Scholar" persona with explicit rules (teach, don't write submissions; cite; never invent).
- **Dynamic context injection** – today's date, student profile and upcoming deadlines are inserted into every system prompt, so "next Friday" and "what's due soon" resolve correctly.
- **Few-shot examples** – the planner sees four worked examples (chat, multi-step, research, notes Q&A).
- **Structured JSON output** – planner, executor and reflector must return one JSON object; a tolerant parser plus one automatic "fix your JSON" retry handles malformed output.
- **ReAct** – the executor alternates *thought → action → observation*.
- **Grounding and citation rules** – factual claims must come from observations and carry `[notes: file p.N]`, `[arXiv]`, `[web: domain]`.
- **Delimiter defence against prompt injection** – retrieved/web text is wrapped in `<untrusted_data>` and the model is told never to follow instructions inside it.
- **Self-critique** – a reviewer prompt checks completeness, support and integrity, and supplies a revised answer.
- **Task-specific prompts** – quiz generator (exactly 4 options, one correct, explanation) and faithful summariser (map-reduce for long text).

## 6. RAG Implementation
- **Ingestion:** PDF (pypdf, page numbers kept), DOCX, TXT, MD → paragraph-aware chunks (900 chars, 150 overlap, word-boundary overlap) → deduplicated by content hash.
- **Hybrid retrieval:** (i) a pure-Python **BM25** keyword index and (ii) **dense embeddings** (`all-MiniLM-L6-v2`) in **Chroma** with cosine distance; the two ranked lists are merged with **Reciprocal Rank Fusion**. Dense search is optional – the system degrades gracefully to BM25 if the model isn't installed.
- **Relevance cut-off:** off-topic queries return *nothing*, so the agent says "not found in your notes" instead of guessing.
- **Use:** `search_notes` returns passages with file/page for citations; `generate_quiz` and `summarize_text(document=…)` also draw on the knowledge base.
- **Security:** retrieved text is sanitised for injected instructions and wrapped as untrusted data.

## 7. Tool / API Integration
| Tool | Type | Purpose |
|---|---|---|
| `search_notes` | RAG | answer from uploaded material |
| `search_arxiv` | REST API (arXiv) | recent research papers with abstracts |
| `wikipedia_summary` | REST API (Wikipedia) | concept overview |
| `web_search` | DuckDuckGo | current information |
| `summarize_text` | LLM | summaries; map-reduce for long documents |
| `generate_quiz` | LLM + RAG | MCQs with answer key |
| `make_study_plan` | deterministic algorithm | difficulty-weighted day-by-day plan + revision phase |
| `calculate_cgpa`, `required_gpa` | deterministic | GPA (10-point/4-point) and target-CGPA feasibility |
| `add_deadline`, `list_deadlines` | memory | persistent deadline tracker |
| `create_calendar_file` | action | exports sessions to `.ics` (Google/Outlook/Apple Calendar) |
| `save_note` | action | saves summaries as Markdown |

Design choices: exact work (dates, GPA, schedules) is done by **code, not the LLM**, for correctness, speed and zero cost;
network results are cached (TTL); every tool returns `ok/error` so failures are handled gracefully; structured outputs can be
passed between tools using `@obs_N` references (e.g. study plan → calendar file) without the LLM retyping data.

## 8. Agent Workflow
Example: *"Exam on 20 Nov: normalization, transactions, indexing, 3 h/day. Make a plan and add it to my calendar."*
1. Guard passes; PII masked.
2. Planner → `intent=multi_step`; plan: (1) `make_study_plan`, (2) `create_calendar_file`.
3. Executor step 1 → `make_study_plan(...)` → observation `@obs_1` (table + structured sessions).
4. Executor step 2 → `create_calendar_file(events="@obs_1")` → `study_plan.ics` created.
5. Executor decides `final`; synthesizer writes the answer with the schedule table and the file name.
6. Reflector verifies; output guard; UI shows answer, download button and the full reasoning trace.

If a tool fails twice the agent **re-plans** once; duplicate calls are skipped; the loop stops at `MAX_STEPS`; an LLM outage returns a friendly error.

## 9. Safety Mechanisms
- **Prompt-injection defence:** direct (regex input filter) and indirect (sanitised, delimited tool/document content).
- **Academic integrity:** refuses cheating/proctoring bypass/"submit as my own" requests and redirects to legitimate help; the system prompt forbids ghost-writing submissions.
- **Wellbeing:** self-harm language triggers a supportive message with Tele-MANAS (India, 14416) and an encouragement to contact a counsellor, with no tool use.
- **Privacy:** emails, phone numbers and 12-digit ID numbers are masked before reaching the LLM or history; API keys are redacted from output; keys stay in `.env`.
- **Bounded autonomy:** allow-listed tools, argument validation, path-traversal protection, sandboxed output folder, step cap, loop guard.
- **Human-in-the-loop:** a UI switch disables all file/memory-writing tools.
- **Honesty:** citations required; "not found" preferred to guessing; reflector flags unsupported claims.

## 10. Results
*(Fill in after running the end-to-end tests in `TEST_REPORT.md` Part B.)*
- Automated tests: **43/43 passed** (tools, RAG, guardrails, full agent loop with a scripted LLM).
- End-to-end success rate: ___ / 10 · average LLM calls per request: ___ · average latency: ___ s.
- Time saved example: building a 3-topic exam plan and calendar file manually ≈ 20–30 min vs. ≈ ___ s with Scholar.
- Retrieval check: correct source ranked first for sample queries; off-topic queries return no passages.

## 11. Limitations
- Quality depends on the chosen LLM; small free models may produce malformed JSON (mitigated by retry) or weaker reasoning.
- Retrieval over scanned (image-only) PDFs needs OCR, which is not included.
- Regex-based guardrails can miss cleverly obfuscated attacks or flag borderline phrasing; they are a first layer, not a guarantee.
- Web search is rate-limited/unreliable; arXiv and Wikipedia only cover public sources.
- Study plans are heuristic (difficulty-weighted); they do not know the student's lecture timetable.
- Grading scale defaults to the Indian 10-point scale; institutions differ.
- Reflection and planning add LLM calls (latency / quota); they can be switched off.

## 12. Future Scope
- Multi-agent specialisation (researcher, planner, tutor) with a supervisor, e.g. LangGraph or CrewAI.
- Direct Google Calendar / Gmail / LMS (Moodle) integration with OAuth.
- OCR and table/figure extraction; citation-style export (APA/IEEE) for research notes.
- Adaptive learning: track quiz scores and re-plan weak topics with spaced repetition.
- Evaluation harness with an LLM judge for groundedness; voice and mobile interface; Hindi/Marathi support.

## References
LangChain documentation · Yao et al., *ReAct* (2022) · Lewis et al., *Retrieval-Augmented Generation* (2020) ·
Cormack et al., *Reciprocal Rank Fusion* (2009) · Robertson & Zaragoza, *BM25* (2009) · arXiv API · Wikipedia API.
