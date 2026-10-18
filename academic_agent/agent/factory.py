"""Wires everything together (used by the CLI and the Streamlit app)."""
from . import config
from .core import AcademicAgent
from .llm import LLMClient
from .memory import Memory
from .rag import KnowledgeBase
from .tools import build_tools


def create_agent(on_event=None, llm=None) -> AcademicAgent:
    config.ensure_dirs()
    kb = KnowledgeBase()
    memory = Memory()
    llm = llm or LLMClient()
    return AcademicAgent(llm, build_tools(llm, kb, memory), kb=kb, memory=memory, on_event=on_event)
