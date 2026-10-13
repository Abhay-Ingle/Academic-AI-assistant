"""The agent: Guard -> Plan -> Execute (ReAct loop with tools) -> Synthesise -> Reflect -> Output guard."""
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import config, guardrails, prompts
from .config import today
from .tools import Tool, ToolResult


@dataclass
class Observation:
    n: int
    tool: str
    args: dict
    result: ToolResult


@dataclass
class AgentResponse:
    answer: str
    blocked: bool = False
    category: str = "ok"
    intent: str = ""
    plan: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    tools_used: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    files: list = field(default_factory=list)
    llm_calls: int = 0
    seconds: float = 0.0


class AcademicAgent:
    def __init__(self, llm, tools: dict[str, Tool], kb=None, memory=None, max_steps: int = config.MAX_STEPS,
                 enable_reflection: bool = config.ENABLE_REFLECTION, allow_side_effects: bool = True,
                 on_event: Callable[[dict], None] | None = None):
        self.llm, self.tools, self.kb, self.memory = llm, tools, kb, memory
        self.max_steps = max_steps
        self.enable_reflection = enable_reflection
        self.allow_side_effects = allow_side_effects   # human-in-the-loop switch for file/memory writes
        self.on_event = on_event
        self.history: list[dict] = []
        self._trace: list[dict] = []

    # ------------------------------------------------------------ helpers ----
    def _emit(self, kind: str, **payload):
        ev = {"kind": kind, **payload}
        self._trace.append(ev)
        if self.on_event:
            self.on_event(ev)

    def _tools_spec(self) -> str:
        return "\n".join(t.spec() for t in self.tools.values())

    def _system(self) -> str:
        deadlines = "none saved"
        if self.memory:
            items = self.memory.list_deadlines(today(), 14)
            if items:
                deadlines = "; ".join(f"{d['title']} on {d['due_date']}" for d in items[:5])
        profile = self.memory.profile_text() if self.memory else "not provided"
        return prompts.SYSTEM_PROMPT.format(today=today().isoformat(), profile=profile, deadlines=deadlines)

    def _history_text(self) -> str:
        turns = self.history[-config.HISTORY_TURNS:]
        return "\n".join(f"{t['role']}: {t['content'][:600]}" for t in turns) or "(no earlier messages)"

    def _obs_text(self, observations: list[Observation]) -> str:
        if not observations:
            return "(none yet)"
        blocks = []
        for o in observations:
            body = o.result.text
            if self.tools[o.tool].untrusted if o.tool in self.tools else False:
                body = f"<untrusted_data>\n{guardrails.sanitize_untrusted(body)}\n</untrusted_data>"
            body = body[:config.OBSERVATION_MAX_CHARS]
            ref = f" (structured data available as @obs_{o.n})" if o.result.data is not None else ""
            status = "ok" if o.result.ok else "FAILED"
            blocks.append(f"Observation @obs_{o.n} - {o.tool}({json.dumps(o.args, default=str)[:300]}) [{status}]{ref}:\n{body}")
        return "\n\n".join(blocks)

    # --------------------------------------------------------------- plan ----
    def _plan(self, message: str, failure_note: str = "") -> dict:
        system = prompts.PLANNER_PROMPT.format(tools=self._tools_spec())
        user = f"Conversation so far:\n{self._history_text()}\n\nToday: {today().isoformat()}\nRequest: {message}"
        if failure_note:
            user += f"\n\nThe previous plan hit problems: {failure_note}\nProduce an alternative plan."
        try:
            raw = self.llm.complete_json(system, user)
        except Exception as exc:
            self._emit("warning", text=f"Planner failed ({exc}); falling back to the executor.")
            raw = {"intent": "other", "needs_tools": True, "plan": []}
        steps = []
        for s in raw.get("plan") or []:
            if isinstance(s, dict):
                tool = s.get("tool") if s.get("tool") in self.tools else None
                steps.append({"step": len(steps) + 1, "goal": str(s.get("goal", ""))[:200], "tool": tool})
        return {"intent": str(raw.get("intent", "other")), "needs_tools": bool(raw.get("needs_tools", bool(steps))),
                "plan": steps[:6], "reasoning": str(raw.get("reasoning", ""))[:200]}

    # ------------------------------------------------------------ execute ----
    def _resolve(self, value: Any, observations: list[Observation]):
        if isinstance(value, str) and value.startswith("@obs_"):
            ref = value[5:]
            if ref.isdigit():
                for o in observations:
                    if o.n == int(ref):
                        return o.result.data if o.result.data is not None else o.result.text
        if isinstance(value, dict):
            return {k: self._resolve(v, observations) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve(v, observations) for v in value]
        return value

    def _run_tool(self, tool: Tool, args: dict) -> ToolResult:
        missing = [r for r in tool.required if r not in args]
        if missing:
            return ToolResult(f"ERROR: missing required argument(s) {missing}. Expected: {tool.spec()}", ok=False)
        try:
            return tool.func(**args)
        except TypeError as exc:
            return ToolResult(f"ERROR: bad arguments ({exc}). Expected: {tool.spec()}", ok=False)
        except Exception as exc:   # a failing tool must never crash the agent
            return ToolResult(f"ERROR: {type(exc).__name__}: {exc}", ok=False)

    def _execute(self, message: str, plan: dict) -> list[Observation]:
        observations: list[Observation] = []
        seen, consecutive_fail, replanned = set(), 0, False
        system = prompts.EXECUTOR_PROMPT.format(tools=self._tools_spec(), today=today().isoformat())
        for step_no in range(1, self.max_steps + 1):
            plan_txt = "\n".join(f"{s['step']}. {s['goal']} [{s['tool']}]" for s in plan["plan"]) or "(no fixed plan)"
            user = (f"Conversation so far:\n{self._history_text()}\n\nRequest: {message}\n\nPLAN:\n{plan_txt}\n\n"
                    f"OBSERVATIONS SO FAR:\n{self._obs_text(observations)}\n\nStep {step_no} of max {self.max_steps}. "
                    "Choose the next action or finish.")
            try:
                decision = self.llm.complete_json(system, user)
            except Exception as exc:
                self._emit("warning", text=f"Executor error: {exc}")
                break
            if decision.get("final") or not decision.get("action"):
                self._emit("thought", text=str(decision.get("thought", "Done."))[:300])
                break
            action = decision["action"] if isinstance(decision["action"], dict) else {}
            name, args = action.get("tool"), action.get("args") or {}
            self._emit("thought", text=str(decision.get("thought", ""))[:300])
            n = len(observations) + 1

            ok, reason = guardrails.check_tool_call(name, args, self.tools.keys())
            if not ok:
                observations.append(Observation(n, name if name in self.tools else "invalid", args,
                                                ToolResult(f"BLOCKED: {reason}", ok=False)))
                self._emit("blocked", tool=name, reason=reason)
                consecutive_fail += 1
            else:
                tool = self.tools[name]
                sig = name + json.dumps(args, sort_keys=True, default=str)
                if sig in seen:
                    res = ToolResult("SKIPPED: identical call already made - use the earlier observation.", ok=False)
                elif tool.side_effect and not self.allow_side_effects:
                    res = ToolResult("SKIPPED: the student has not allowed file/memory changes. "
                                     "Tell them to enable it, then ask again.", ok=False)
                else:
                    seen.add(sig)
                    res = self._run_tool(tool, self._resolve(args, observations))
                observations.append(Observation(n, name, args, res))
                self._emit("tool", n=n, tool=name, args=args, ok=res.ok, preview=res.text[:300])
                consecutive_fail = 0 if res.ok else consecutive_fail + 1

            if consecutive_fail >= 2 and not replanned:      # self-correction: re-plan once
                replanned = True
                note = observations[-1].result.text[:200]
                plan = self._plan(message, failure_note=note)
                self._emit("replan", plan=plan["plan"])
                consecutive_fail = 0
            elif consecutive_fail >= 3:
                break
        return observations

    # ------------------------------------------------- synthesise/reflect ----
    def _synthesize(self, message: str, observations: list[Observation]) -> str:
        user = (f"Conversation so far:\n{self._history_text()}\n\nStudent request: {message}\n\n"
                f"TOOL OBSERVATIONS:\n{self._obs_text(observations)}")
        files = [f for o in observations for f in o.result.files]
        if files:
            user += "\n\nFiles created: " + ", ".join(f.replace("\\", "/").split("/")[-1] for f in files)
        return self.llm.complete(self._system() + "\n" + prompts.SYNTH_PROMPT, user)

    def _reflect(self, message: str, draft: str, observations: list[Observation]) -> str:
        user = (f"Student request: {message}\n\nOBSERVATIONS:\n{self._obs_text(observations)}\n\nDRAFT ANSWER:\n{draft}")
        try:
            verdict = self.llm.complete_json(prompts.REFLECT_PROMPT, user)
        except Exception:
            return draft
        if verdict.get("ok", True) or not str(verdict.get("revised_answer", "")).strip():
            self._emit("reflection", ok=True, issues=[])
            return draft
        self._emit("reflection", ok=False, issues=verdict.get("issues", []))
        return str(verdict["revised_answer"])

    # ---------------------------------------------------------------- run ----
    def run(self, message: str) -> AgentResponse:
        t0 = time.time()
        calls0 = getattr(self.llm, "calls", 0)
        self._trace = []
        guard = guardrails.check_input(message)
        self._emit("input_guard", allowed=guard.allowed, category=guard.category, pii_found=guard.pii_found)
        if not guard.allowed:
            return AgentResponse(guard.message, blocked=True, category=guard.category, trace=list(self._trace),
                                 seconds=round(time.time() - t0, 2))
        msg = guard.text
        plan = self._plan(msg)
        self._emit("plan", intent=plan["intent"], needs_tools=plan["needs_tools"], plan=plan["plan"],
                   reasoning=plan["reasoning"])
        observations: list[Observation] = []
        try:
            if not plan["needs_tools"]:
                answer = self.llm.complete(self._system(), f"Conversation so far:\n{self._history_text()}\n\nStudent: {msg}")
            else:
                observations = self._execute(msg, plan)
                answer = self._synthesize(msg, observations)
                if self.enable_reflection and observations:
                    answer = self._reflect(msg, answer, observations)
        except Exception as exc:
            answer = f"Sorry - something went wrong while contacting the language model ({exc}). Please try again."
        answer = guardrails.check_output(answer)
        self.history += [{"role": "student", "content": msg}, {"role": "assistant", "content": answer}]
        self.history = self.history[-2 * config.HISTORY_TURNS:]
        return AgentResponse(
            answer=answer, intent=plan["intent"], plan=plan["plan"], trace=list(self._trace),
            tools_used=[o.tool for o in observations if o.result.ok],
            sources=list(dict.fromkeys(s for o in observations for s in o.result.sources)),
            files=list(dict.fromkeys(f for o in observations for f in o.result.files)),
            llm_calls=getattr(self.llm, "calls", 0) - calls0, seconds=round(time.time() - t0, 2))
