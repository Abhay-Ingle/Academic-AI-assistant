"""Test doubles: a scripted LLM so the whole agent loop can be tested offline and deterministically."""
import json


class ScriptedLLM:
    """Returns pre-written replies in order; records every prompt it receives."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.prompts = []

    def complete(self, system, user):
        self.calls += 1
        self.prompts.append((system, user))
        if not self.replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        r = self.replies.pop(0)
        return r if isinstance(r, str) else json.dumps(r)

    def complete_json(self, system, user):
        from agent.llm import extract_json
        return extract_json(self.complete(system, user))
