"""Thin LLM wrapper (LangChain chat models underneath) with retries and robust JSON parsing.

The rest of the code only depends on `complete()` / `complete_json()`, so the provider can be swapped
(Groq, Gemini, OpenAI, local Ollama) via .env without touching the agent logic.
"""
import json
import re
import time

from . import config

DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-2.0-flash",   # check your provider's console for the current model names
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1",
}


class LLMError(RuntimeError):
    pass


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


def _build_chat_model():
    provider = config.LLM_PROVIDER
    model = config.LLM_MODEL or DEFAULT_MODELS.get(provider, "")
    temp = config.LLM_TEMPERATURE
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temp)          # needs GROQ_API_KEY
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temp)  # needs GOOGLE_API_KEY
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temp)        # needs OPENAI_API_KEY
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=temp)        # local, no key
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'")


class LLMClient:
    def __init__(self):
        self._chat = _build_chat_model()
        self.calls = 0  # number of LLM calls made (used for cost/latency reporting)

    def complete(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        last = None
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                self.calls += 1
                resp = self._chat.invoke([SystemMessage(content=system), HumanMessage(content=user)])
                content = resp.content
                if isinstance(content, list):  # some providers return content blocks
                    content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
                return str(content).strip()
            except Exception as exc:  # rate limit / network -> exponential back-off
                last = exc
                time.sleep(2 ** attempt)
        raise LLMError(f"LLM call failed after {config.LLM_MAX_RETRIES} attempts: {last}")

    def complete_json(self, system: str, user: str) -> dict:
        raw = self.complete(system, user)
        try:
            return extract_json(raw)
        except (ValueError, json.JSONDecodeError):
            fixed = self.complete(system, user + "\n\nYour previous reply was not valid JSON. "
                                                 "Reply with ONE valid JSON object and nothing else.")
            return extract_json(fixed)
