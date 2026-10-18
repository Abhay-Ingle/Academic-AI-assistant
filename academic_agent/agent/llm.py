"""Thin LLM wrapper (LangChain chat models underneath) with retries and robust JSON parsing.

The rest of the code only depends on `complete()` / `complete_json()`, so the provider can be swapped
(Groq, Gemini, OpenAI, local Ollama) via .env without touching the agent logic.
"""
import json
import re
import time

from . import config

# Primary model per provider
DEFAULT_MODELS = {
    "groq":   "qwen/qwen3.8-27b",       # no built-in tools, works cleanly with our agent
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1",
}

# Groq fallback chain — tried in order when a model hits its limit or errors
# Only models verified available on this account:
#   qwen/qwen3.8-27b, openai/gpt-oss-20b, openai/gpt-oss-120b
GROQ_FALLBACK_MODELS = [
    "qwen/qwen3.8-27b",         # primary  — no built-in tool conflict, stable
    "openai/gpt-oss-20b",       # fallback — may have tool_use_failed; caught by error handler
    "openai/gpt-oss-120b",      # last resort
]


class LLMError(RuntimeError):
    pass


def _is_rate_limit(exc: Exception) -> bool:
    """Return True for errors where retrying the same model won't help
    and we should immediately switch to the next fallback model:
      - 429 rate_limit_exceeded  : daily/minute token quota exhausted
      - 404 model_not_found      : model name wrong or not on this account
      - 400 tool_use_failed      : model has built-in tools that conflict with our agent
        (seen with openai/gpt-oss-* models calling browser.search)
    """
    msg = str(exc).lower()
    return (
        "429" in msg or "rate_limit_exceeded" in msg or "rate limit" in msg
        or "404" in msg or "model_not_found" in msg or "does not exist" in msg
        or "tool_use_failed" in msg or "tool choice is none" in msg
    )


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM reply (handles ```json fences and chatter)."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON")


def _build_chat_model(model: str | None = None):
    provider = config.LLM_PROVIDER
    m = model or config.LLM_MODEL or DEFAULT_MODELS.get(provider, "")
    temp = config.LLM_TEMPERATURE
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=m, temperature=temp)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=m, temperature=temp)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=m, temperature=temp)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=m, temperature=temp)
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'")


class LLMClient:
    def __init__(self):
        # Build the initial model — prefer the env override, then the default for the provider
        provider = config.LLM_PROVIDER
        initial = config.LLM_MODEL or DEFAULT_MODELS.get(provider, "")
        self._model_name = initial
        self._chat = _build_chat_model(initial)
        self.calls = 0
        # Fallback queue (only used for groq)
        self._fallback_queue = (
            [m for m in GROQ_FALLBACK_MODELS if m != initial]
            if provider == "groq" else []
        )

    # ── private ───────────────────────────────────────────────────────────────
    def _switch_model(self) -> bool:
        """Try the next fallback model. Returns True if switched, False if none left."""
        if not self._fallback_queue:
            return False
        next_model = self._fallback_queue.pop(0)
        print(f"[llm] switching to fallback model: {next_model}")
        self._model_name = next_model
        self._chat = _build_chat_model(next_model)
        return True

    def _invoke(self, messages) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: F401
        resp = self._chat.invoke(messages)
        content = resp.content
        if isinstance(content, list):
            content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        return str(content).strip()

    # ── public ────────────────────────────────────────────────────────────────
    def complete(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        last_exc = None

        # Try current model + each fallback
        attempts_remaining = config.LLM_MAX_RETRIES
        while True:
            for attempt in range(attempts_remaining):
                try:
                    self.calls += 1
                    return self._invoke(messages)
                except Exception as exc:
                    last_exc = exc
                    if _is_rate_limit(exc):
                        # Daily/token quota hit — no point sleeping, switch model immediately
                        break
                    # Transient error (network, 503, …) — short back-off then retry
                    time.sleep(min(2 ** attempt, 8))

            # Rate-limit path: try next model in the fallback chain
            if last_exc and _is_rate_limit(last_exc):
                if self._switch_model():
                    attempts_remaining = 2   # give the new model 2 quick tries
                    last_exc = None
                    continue
            # All models exhausted or non-recoverable error
            break

        raise LLMError(f"LLM call failed (model: {self._model_name}): {last_exc}")

    def complete_json(self, system: str, user: str) -> dict:
        raw = self.complete(system, user)
        try:
            return extract_json(raw)
        except (ValueError, json.JSONDecodeError):
            fixed = self.complete(
                system,
                user + "\n\nYour previous reply was not valid JSON. "
                       "Reply with ONE valid JSON object and nothing else.",
            )
            return extract_json(fixed)

    @property
    def current_model(self) -> str:
        return self._model_name
