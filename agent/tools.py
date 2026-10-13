"""Tool library for the academic agent.

Each tool returns a ToolResult(text for the LLM, structured `data` other tools can reuse via "@obs_N").
Deterministic tools (CGPA, study planner, calendar, deadlines) never call the LLM - fast, free and exact.
"""
import functools
import json
import math
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from . import config, guardrails, prompts
from .config import today


# ------------------------------------------------------------------ plumbing ----
@dataclass
class ToolResult:
    text: str
    data: Any = None
    sources: list = field(default_factory=list)
    files: list = field(default_factory=list)
    ok: bool = True


@dataclass
class Tool:
    name: str
    description: str
    params: dict          # arg name -> short description
    required: list
    func: Callable[..., ToolResult]
    side_effect: bool = False   # writes files / memory (needs user permission in the UI)
    untrusted: bool = False     # output comes from the web / documents (sanitised + wrapped)

    def spec(self) -> str:
        args = ", ".join(f"{k}{'*' if k in self.required else ''}: {v}" for k, v in self.params.items())
        return f"- {self.name}({args}) -> {self.description}"


def ttl_cache(seconds: int = 900):
    """Cache successful results of slow network tools (saves time and API quota)."""
    def deco(fn):
        store = {}

        @functools.wraps(fn)
        def wrapper(*a, **kw):
            key = json.dumps([a, kw], sort_keys=True, default=str)
            hit = store.get(key)
            if hit and time.time() - hit[0] < seconds:
                return hit[1]
            val = fn(*a, **kw)
            if getattr(val, "ok", False):
                store[key] = (time.time(), val)
            return val
        return wrapper
    return deco


def err(msg: str) -> ToolResult:
    return ToolResult(f"ERROR: {msg}", ok=False)


def as_list(x):
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except json.JSONDecodeError:
            x = [s.strip() for s in re.split(r"[,\n;]", x) if s.strip()]
    if isinstance(x, dict):
        x = [x]
    return x if isinstance(x, list) else []


_DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d %B %Y", "%d %b %Y",
                 "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y"]


def parse_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip().lower()
    if s == "today":
        return today()
    if s == "tomorrow":
        return today() + timedelta(days=1)
    m = re.fullmatch(r"in (\d+) days?", s)
    if m:
        return today() + timedelta(days=int(m.group(1)))
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(str(value).strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date '{value}'. Use YYYY-MM-DD.")


# ------------------------------------------------------------ grades / CGPA ----
SCALE_10 = {"O": 10, "A+": 9, "A": 8, "B+": 7, "B": 6, "C": 5, "P": 4, "F": 0, "AB": 0}
SCALE_4 = {"A+": 4.0, "A": 4.0, "A-": 3.7, "B+": 3.3, "B": 3.0, "B-": 2.7, "C+": 2.3, "C": 2.0,
           "C-": 1.7, "D+": 1.3, "D": 1.0, "F": 0.0}


def calculate_cgpa(courses, scale="10") -> ToolResult:
    """courses: [{"name","credits","grade"}]; grade = letter (O, A+, ... / A, A-, ...) or numeric grade points."""
    table = SCALE_4 if str(scale).strip() == "4" else SCALE_10
    top = 4.0 if table is SCALE_4 else 10.0
    rows, total_pts, total_cr = [], 0.0, 0.0
    for c in as_list(courses):
        try:
            credits = float(c["credits"])
            g = str(c["grade"]).strip().upper()
            pts = table[g] if g in table else float(g)
        except (KeyError, ValueError, TypeError):
            return err(f"Could not read course {c!r}. Each course needs name, credits and a valid grade.")
        if credits <= 0 or not (0 <= pts <= top):
            return err(f"Invalid credits/grade in {c!r} (grade points must be 0-{top:g}).")
        rows.append(f"| {c.get('name', '-')} | {credits:g} | {g} | {pts:g} |")
        total_pts += pts * credits
        total_cr += credits
    if not rows:
        return err("No courses supplied.")
    gpa = total_pts / total_cr
    text = ("| Course | Credits | Grade | Points |\n|---|---|---|---|\n" + "\n".join(rows) +
            f"\nTotal credits: {total_cr:g}\nGPA (scale {top:g}): {gpa:.2f}")
    return ToolResult(text, data={"gpa": round(gpa, 2), "credits": total_cr})


def required_gpa(current_cgpa, completed_credits, upcoming_credits, target_cgpa, max_gpa=10) -> ToolResult:
    try:
        cur, done, up, tgt, mx = (float(x) for x in (current_cgpa, completed_credits, upcoming_credits,
                                                    target_cgpa, max_gpa))
    except (TypeError, ValueError):
        return err("All arguments must be numbers.")
    if up <= 0 or done < 0:
        return err("upcoming_credits must be > 0.")
    needed = (tgt * (done + up) - cur * done) / up
    if needed > mx:
        msg = (f"Target {tgt:g} is NOT reachable: you would need {needed:.2f} (> max {mx:g}) in the next "
               f"{up:g} credits. Best possible CGPA = {(cur * done + mx * up) / (done + up):.2f}.")
    elif needed <= 0:
        msg = f"You've already secured the target {tgt:g}; any passing result keeps you above it."
    else:
        msg = f"To reach CGPA {tgt:g} you need an average of {needed:.2f} over the next {up:g} credits."
    return ToolResult(msg, data={"needed": round(needed, 2)})


# ------------------------------------------------------------ study planner ----
def _norm_topics(topics) -> list[tuple[str, float]]:
    out = []
    for t in as_list(topics):
        if isinstance(t, dict):
            name = str(t.get("name") or t.get("topic") or "").strip()
            diff = t.get("difficulty", 3)
        else:
            name, diff = str(t).strip(), 3
        try:
            diff = min(5.0, max(1.0, float(diff)))
        except (TypeError, ValueError):
            diff = 3.0
        if name:
            out.append((name, diff))
    return out


def make_study_plan(topics, exam_date, hours_per_day=3, start_date=None) -> ToolResult:
    """Difficulty-weighted day-by-day schedule + a final revision phase (deterministic, no LLM)."""
    tp = _norm_topics(topics)
    if not tp:
        return err("Provide at least one topic.")
    try:
        exam = parse_date(exam_date)
        start = parse_date(start_date) if start_date else today()
        hpd = min(12.0, max(0.5, float(hours_per_day)))
    except (ValueError, TypeError) as e:
        return err(str(e))
    total_days = (exam - start).days
    if total_days < 1:
        return err(f"Exam date {exam} must be after the start date {start}.")

    rev_days = max(1, round(total_days * 0.2)) if total_days >= 7 else (1 if total_days >= 3 else total_days)
    study_days = total_days - rev_days
    sessions = []

    if study_days > 0:
        budget = study_days * hpd
        wsum = sum(d for _, d in tp)
        day, cap = 0, hpd
        for name, diff in tp:
            need = budget * diff / wsum
            while need > 0.01 and day < study_days:
                chunk = min(need, cap)
                sessions.append({"date": (start + timedelta(days=day)).isoformat(), "topic": name,
                                 "hours": round(chunk, 2), "type": "study", "title": f"Study: {name}"})
                need -= chunk
                cap -= chunk
                if cap <= 0.01:
                    day, cap = day + 1, hpd

    hardest_first = [n for n, _ in sorted(tp, key=lambda x: -x[1])]
    per_day = math.ceil(len(hardest_first) / rev_days)
    for j in range(rev_days):
        day_topics = [hardest_first[(j * per_day + t) % len(hardest_first)] for t in range(per_day)]
        label = "Revise" if j < rev_days - 1 else "Final revision + mock test"
        for name in day_topics:
            sessions.append({"date": (start + timedelta(days=study_days + j)).isoformat(), "topic": name,
                             "hours": round(hpd / len(day_topics), 2), "type": "revision",
                             "title": f"{label}: {name}"})

    lines = ["| Date | Day | Topic | Hours | Type |", "|---|---|---|---|---|"]
    for s in sessions:
        d = date.fromisoformat(s["date"])
        lines.append(f"| {s['date']} | {d.strftime('%a')} | {s['topic']} | {s['hours']:g} | {s['type']} |")
    total_hours = sum(s["hours"] for s in sessions)
    warn = ""
    if study_days > 0 and study_days * hpd / len(tp) < 1:
        warn = "\nWARNING: less than 1 hour per topic - consider more hours/day or fewer topics."
    head = (f"Plan {start} -> exam {exam}: {total_days} days ({study_days} study + {rev_days} revision), "
            f"{hpd:g} h/day, {total_hours:g} h total. Harder topics get more time.{warn}\n")
    return ToolResult(head + "\n".join(lines), data=sessions)


# ---------------------------------------------------------------- calendar ----
def _ics_escape(s: str) -> str:
    return (str(s).replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")
            .replace("\r\n", "\\n").replace("\n", "\\n"))


def _fold(line: str) -> list[str]:
    out = []
    while len(line) > 73:
        out.append(line[:73])
        line = " " + line[73:]
    out.append(line)
    return out


def create_calendar_file(events, filename="study_plan.ics", outdir=None) -> ToolResult:
    """events: [{"title","date", optional "start_time" HH:MM, "hours"/"duration_hours", "description"}] -> .ics file."""
    evs = as_list(events)
    if not evs:
        return err("No events supplied.")
    outdir = Path(outdir or config.OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Academic AI Agent//EN", "CALSCALE:GREGORIAN"]
    n = 0
    for i, ev in enumerate(evs):
        try:
            title = str(ev.get("title") or ev.get("topic") or "Study session")
            d = parse_date(ev["date"])
        except (KeyError, ValueError, AttributeError):
            return err(f"Event {ev!r} needs a title and a valid date (YYYY-MM-DD).")
        lines += ["BEGIN:VEVENT", f"UID:{stamp}-{i}@academic-agent", f"DTSTAMP:{stamp}",
                  f"SUMMARY:{_ics_escape(title)}"]
        st = ev.get("start_time")
        if st:
            try:
                h, m = (int(x) for x in str(st).split(":")[:2])
                begin = datetime(d.year, d.month, d.day, h, m)
                hrs = float(ev.get("duration_hours") or ev.get("hours") or 1)
            except (ValueError, TypeError):
                return err(f"Bad start_time/duration in {ev!r} (use HH:MM).")
            lines += [f"DTSTART:{begin:%Y%m%dT%H%M%S}", f"DTEND:{begin + timedelta(hours=hrs):%Y%m%dT%H%M%S}"]
        else:
            lines += [f"DTSTART;VALUE=DATE:{d:%Y%m%d}", f"DTEND;VALUE=DATE:{d + timedelta(days=1):%Y%m%d}"]
        desc = ev.get("description") or (f"{ev['hours']} h" if ev.get("hours") else "")
        if desc:
            lines.append(f"DESCRIPTION:{_ics_escape(desc)}")
        lines.append("END:VEVENT")
        n += 1
    lines.append("END:VCALENDAR")
    path = outdir / guardrails.safe_filename(filename, ".ics")
    folded = [seg for ln in lines for seg in _fold(ln)]
    path.write_bytes(("\r\n".join(folded) + "\r\n").encode("utf-8"))
    return ToolResult(f"Created calendar file '{path.name}' with {n} events (import it into Google/Outlook/Apple Calendar).",
                      files=[str(path)])


# ------------------------------------------------------------ web / papers ----
HEADERS = {"User-Agent": "AcademicAgent/1.0 (student project)"}


@ttl_cache(900)
def web_search(query: str, max_results: int = 4) -> ToolResult:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        hits = list(DDGS().text(query, max_results=int(max_results)))
    except Exception as exc:
        return err(f"web search unavailable ({type(exc).__name__}). Try again or use another tool.")
    if not hits:
        return ToolResult("No web results found.", ok=True)
    out, srcs = [], []
    for h in hits:
        url = h.get("href") or h.get("url") or ""
        dom = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
        out.append(f"- {h.get('title', '')} ({url})\n  {h.get('body', '')}")
        srcs.append(f"web: {dom}")
    return ToolResult("\n".join(out), sources=srcs)


@ttl_cache(900)
def search_arxiv(query: str, max_results: int = 3) -> ToolResult:
    import requests
    try:
        r = requests.get("https://export.arxiv.org/api/query", headers=HEADERS, timeout=20,
                         params={"search_query": f"all:{query}", "start": 0,
                                 "max_results": min(int(max_results), 6), "sortBy": "relevance"})
        r.raise_for_status()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entries = ET.fromstring(r.text).findall("a:entry", ns)
    except Exception as exc:
        return err(f"arXiv unavailable ({type(exc).__name__}).")
    if not entries:
        return ToolResult("No arXiv papers found for that query.")
    out = []
    for e in entries:
        title = " ".join(e.findtext("a:title", "", ns).split())
        summ = " ".join(e.findtext("a:summary", "", ns).split())[:500]
        authors = ", ".join(a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)[:3])
        out.append(f"- {title} ({e.findtext('a:published', '', ns)[:10]}) by {authors}\n"
                   f"  {e.findtext('a:id', '', ns)}\n  Abstract: {summ}")
    return ToolResult("\n".join(out), sources=["arXiv"])


@ttl_cache(3600)
def wikipedia_summary(topic: str) -> ToolResult:
    import requests
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php", headers=HEADERS, timeout=15, params={
            "action": "query", "format": "json", "prop": "extracts", "exintro": 1, "explaintext": 1,
            "redirects": 1, "generator": "search", "gsrsearch": topic, "gsrlimit": 1})
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
    except Exception as exc:
        return err(f"Wikipedia unavailable ({type(exc).__name__}).")
    if not pages:
        return ToolResult("No Wikipedia article found.")
    page = next(iter(pages.values()))
    return ToolResult(f"{page.get('title')}: {page.get('extract', '')[:1500]}", sources=["wikipedia"])


# -------------------------------------------------------------------- build ----
def build_tools(llm, kb, memory, outdir=None) -> dict[str, Tool]:
    outdir = Path(outdir or config.OUTPUT_DIR)

    def search_notes(query: str, k: int = 4) -> ToolResult:
        hits = kb.search(query, k=int(k))
        if not hits:
            return ToolResult("No relevant passages found in the uploaded notes.")
        parts, srcs = [], []
        for h in hits:
            pg = f" p.{h['page']}" if h.get("page") else ""
            parts.append(f"[notes: {h['source']}{pg}]\n{h['text']}")
            srcs.append(f"notes: {h['source']}{pg}")
        return ToolResult("\n\n".join(parts), sources=list(dict.fromkeys(srcs)))

    def summarize_text(text=None, document=None, style="bullets", max_words=150) -> ToolResult:
        if document and not text:
            chunks = kb.chunks_of(str(document))
            if not chunks:
                return err(f"Document '{document}' not found. Available: {', '.join(kb.sources()) or 'none'}.")
            text = "\n\n".join(c.text for c in chunks)
        if isinstance(text, (list, dict)):
            text = json.dumps(text, ensure_ascii=False)
        text = str(text or "").strip()
        if not text:
            return err("Nothing to summarise.")
        mw = min(400, max(30, int(max_words)))
        if len(text) <= 6000:                                     # single pass
            return ToolResult(llm.complete(prompts.SUMMARY_SYSTEM, prompts.summary_user(text, style, mw)))
        parts = [text[i:i + 5000] for i in range(0, min(len(text), 40000), 5000)]   # map-reduce for long text
        partial = [llm.complete(prompts.SUMMARY_SYSTEM, prompts.summary_user(p, "key points", 80)) for p in parts]
        return ToolResult(llm.complete(prompts.SUMMARY_SYSTEM,
                                       prompts.summary_user("\n".join(partial), style, mw)))

    def generate_quiz(topic: str, num_questions: int = 5, difficulty: str = "medium") -> ToolResult:
        n = min(15, max(1, int(num_questions)))
        hits = kb.search(topic, k=4)
        ctx = "\n\n".join(h["text"] for h in hits)
        data = llm.complete_json(prompts.QUIZ_SYSTEM, prompts.quiz_user(topic, n, difficulty, ctx))
        qs = [q for q in data.get("questions", [])
              if isinstance(q, dict) and len(q.get("options", [])) == 4 and str(q.get("answer", "")).upper()[:1] in "ABCD"]
        if not qs:
            return err("The quiz could not be generated. Try again with a more specific topic.")
        out, key = [], []
        for i, q in enumerate(qs[:n], 1):
            opts = "\n".join(f"   {l}) {o}" for l, o in zip("ABCD", q["options"]))
            out.append(f"Q{i}. {q['question']}\n{opts}")
            key.append(f"Q{i}: {q['answer'].upper()[:1]} - {q.get('explanation', '')}")
        srcs = list(dict.fromkeys(f"notes: {h['source']}" for h in hits))
        return ToolResult("\n\n".join(out) + "\n\nANSWER KEY\n" + "\n".join(key), data=qs, sources=srcs)

    def add_deadline(title: str, due_date: str, course: str = "") -> ToolResult:
        try:
            due = parse_date(due_date)
        except ValueError as e:
            return err(str(e))
        item = memory.add_deadline(str(title)[:120], due, str(course)[:60])
        return ToolResult(f"Saved deadline '{item['title']}' due {item['due_date']} ({(due - today()).days} days left).",
                          data=item)

    def list_deadlines(days_ahead: int = 30) -> ToolResult:
        items = memory.list_deadlines(today(), int(days_ahead))
        if not items:
            return ToolResult(f"No deadlines in the next {int(days_ahead)} days.", data=[])
        lines = [f"- {d['due_date']} ({d['days_left']} days): {d['title']}" + (f" [{d['course']}]" if d["course"] else "")
                 for d in items]
        return ToolResult("\n".join(lines), data=items)

    def save_note(title: str, content: str) -> ToolResult:
        folder = outdir / "notes"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / guardrails.safe_filename(title, ".md")
        path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
        memory.add_note(str(title), str(path))
        return ToolResult(f"Saved note '{path.name}'.", files=[str(path)])

    tools = [
        Tool("search_notes", "search the student's uploaded notes/PDFs (RAG); returns passages with sources",
             {"query": "search text", "k": "number of passages (default 4)"}, ["query"], search_notes, untrusted=True),
        Tool("web_search", "search the web for current information",
             {"query": "search text", "max_results": "default 4"}, ["query"], web_search, untrusted=True),
        Tool("search_arxiv", "find research papers on arXiv (titles, authors, abstracts)",
             {"query": "topic keywords", "max_results": "default 3"}, ["query"], search_arxiv, untrusted=True),
        Tool("wikipedia_summary", "short encyclopedic overview of a concept",
             {"topic": "concept name"}, ["topic"], wikipedia_summary, untrusted=True),
        Tool("summarize_text", "summarise text, an uploaded document, or a previous result (@obs_N)",
             {"text": "text or @obs_N", "document": "uploaded file name (alternative to text)",
              "style": "bullets|paragraph|exam-notes", "max_words": "default 150"}, [], summarize_text),
        Tool("generate_quiz", "make an MCQ quiz with answer key (uses the notes if relevant)",
             {"topic": "topic", "num_questions": "1-15 (default 5)", "difficulty": "easy|medium|hard"},
             ["topic"], generate_quiz),
        Tool("make_study_plan", "build a day-by-day study schedule before an exam; returns sessions usable as events",
             {"topics": "list of names or [{name, difficulty 1-5}]", "exam_date": "YYYY-MM-DD",
              "hours_per_day": "default 3", "start_date": "YYYY-MM-DD (default today)"},
             ["topics", "exam_date"], make_study_plan),
        Tool("calculate_cgpa", "compute GPA/CGPA from courses",
             {"courses": "[{name, credits, grade}] (letter or points)", "scale": "'10' (default) or '4'"},
             ["courses"], calculate_cgpa),
        Tool("required_gpa", "average grade needed in upcoming credits to reach a target CGPA",
             {"current_cgpa": "number", "completed_credits": "number", "upcoming_credits": "number",
              "target_cgpa": "number", "max_gpa": "default 10"},
             ["current_cgpa", "completed_credits", "upcoming_credits", "target_cgpa"], required_gpa),
        Tool("add_deadline", "save an assignment/exam deadline to the student's tracker",
             {"title": "what", "due_date": "YYYY-MM-DD", "course": "optional"}, ["title", "due_date"],
             add_deadline, side_effect=True),
        Tool("list_deadlines", "show upcoming deadlines",
             {"days_ahead": "default 30"}, [], list_deadlines),
        Tool("create_calendar_file", "export events to an .ics calendar file (events may be '@obs_N' of a study plan)",
             {"events": "[{title, date, start_time?, hours?}] or @obs_N", "filename": "e.g. study_plan.ics"},
             ["events"], lambda events, filename="study_plan.ics": create_calendar_file(events, filename, outdir),
             side_effect=True),
        Tool("save_note", "save study notes / a summary as a Markdown file",
             {"title": "note title", "content": "markdown text or @obs_N"}, ["title", "content"],
             save_note, side_effect=True),
    ]
    return {t.name: t for t in tools}
