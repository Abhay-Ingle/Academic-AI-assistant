"""Safety layer: input checks, PII redaction, prompt-injection defence, tool-call validation, output filtering."""
import json
import re
from dataclasses import dataclass
from pathlib import PurePath

from . import config

INJECTION_PATTERNS = [
    r"ignore (all |any |the )*(previous|prior|above|earlier|system) (instructions|prompts?|rules|messages)",
    r"disregard (all |any |the )*(previous|prior|above|earlier|system) (instructions|prompts?|rules)",
    r"(reveal|show|print|repeat|leak|display) (me )?(your|the) (system|hidden|initial|secret) (prompt|instructions|message)",
    r"\bdeveloper mode\b",
    r"\bjailbreak\w*\b",
    r"\bDAN mode\b",
    r"act as (an? )?(unrestricted|unfiltered|uncensored)",
    r"pretend (that )?(you have|there are) no (rules|restrictions|guidelines)",
]

CHEATING_PATTERNS = [
    r"\b(bypass|evade|fool|trick|beat|get around)\w*\b.{0,40}\b(proctor\w*|plagiarism|turnitin|ai[- ]?detect\w*)",
    r"\b(undetect\w*|untraceable)\b.{0,40}\b(plagiaris\w*|ai[- ]?detect\w*|cheat\w*)",
    r"\bcheat\w*\b.{0,30}\b(exam|test|quiz|viva)",
    r"\b(take|attempt|write|give|appear for)\b.{0,15}\bmy (online )?(exam|test|quiz)\b.{0,15}\bfor me\b",
    r"(submit|turn (it )?in|pass (it )?off).{0,30}as my own",
]

SELF_HARM_PATTERNS = [
    r"\b(kill myself|end my life|want to die|suicid\w*|self[- ]?harm|hurt myself|don'?t want to live)\b",
]

SECRET_PATTERN = re.compile(r"(gsk_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_\-]{20,}|AIza[0-9A-Za-z\-_]{30,})")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
AADHAAR = re.compile(r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)")
PHONE = re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")

REFUSAL_INJECTION = ("I can't follow instructions that try to override my rules or reveal my internal prompt. "
                     "I'm happy to help with your studies instead - try asking about a concept, a study plan, "
                     "your notes, or a research topic.")
REFUSAL_CHEATING = ("I can't help with cheating, bypassing proctoring/plagiarism checks, or work meant to be "
                    "submitted as your own. I *can* explain the concepts, outline an answer, give practice "
                    "questions, or build a study plan so you can do it yourself - want me to do one of those?")
SUPPORT_MESSAGE = ("I'm really sorry you're feeling this way - you deserve support, and you don't have to handle "
                   "it alone. Please consider talking to someone you trust, your college counsellor, or a "
                   "helpline right now. In India you can call Tele-MANAS at 14416; if you're elsewhere, contact "
                   "your local emergency number or crisis line. If you're in immediate danger, call emergency "
                   "services. When you feel ready, I'm here to help with study stress too - we can make a lighter "
                   "plan together.")


@dataclass
class GuardResult:
    allowed: bool
    category: str = "ok"          # ok | empty | too_long | injection | cheating | self_harm
    message: str = ""             # what to show the user when blocked
    text: str = ""                # sanitised text to pass downstream
    pii_found: bool = False


def _any(patterns, text) -> bool:
    return any(re.search(p, text, re.I | re.S) for p in patterns)


def redact_pii(text: str) -> tuple[str, bool]:
    out = EMAIL.sub("[EMAIL]", text)
    out = AADHAAR.sub("[ID-NUMBER]", out)
    out = PHONE.sub("[PHONE]", out)
    return out, out != text


def check_input(text: str) -> GuardResult:
    text = (text or "").strip()
    if not text:
        return GuardResult(False, "empty", "Please type a question or task.")
    if len(text) > config.MAX_INPUT_CHARS:
        return GuardResult(False, "too_long",
                           f"That message is too long ({len(text)} characters). Please keep it under "
                           f"{config.MAX_INPUT_CHARS} or upload the document instead.")
    if _any(SELF_HARM_PATTERNS, text):
        return GuardResult(False, "self_harm", SUPPORT_MESSAGE)
    if _any(INJECTION_PATTERNS, text):
        return GuardResult(False, "injection", REFUSAL_INJECTION)
    if _any(CHEATING_PATTERNS, text):
        return GuardResult(False, "cheating", REFUSAL_CHEATING)
    clean, found = redact_pii(text)
    return GuardResult(True, "ok", "", clean, found)


def sanitize_untrusted(text: str, max_chars: int = 6000) -> str:
    """Neutralise indirect prompt injection hidden in web pages / uploaded documents."""
    text = text[:max_chars]
    for p in INJECTION_PATTERNS:
        text = re.sub(p, "[removed suspicious instruction]", text, flags=re.I | re.S)
    return text


_PATH_KEYS = {"filename", "file", "path", "document"}


def check_tool_call(name, args, allowed_names) -> tuple[bool, str]:
    if name not in set(allowed_names):
        return False, f"Unknown tool '{name}'. Allowed tools: {', '.join(sorted(allowed_names))}."
    if not isinstance(args, dict):
        return False, "Tool arguments must be a JSON object."
    if len(json.dumps(args, default=str)) > 20000:
        return False, "Tool arguments are too large."
    for key, val in args.items():
        if key in _PATH_KEYS and isinstance(val, str):
            if ".." in PurePath(val).parts or val.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", val):
                return False, f"Unsafe path in '{key}'. Only plain file names are allowed."
    return True, ""


def safe_filename(name: str, ext: str = "") -> str:
    base = PurePath(str(name)).name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "file"
    if ext and not base.lower().endswith(ext):
        base += ext
    return base[:80]


def check_output(text: str) -> str:
    """Last line of defence: never leak keys; never leave the user with an empty answer."""
    text = SECRET_PATTERN.sub("[REDACTED-KEY]", text or "")
    return text.strip() or "Sorry, I couldn't produce an answer. Please rephrase your request."
