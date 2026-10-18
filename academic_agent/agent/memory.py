"""Persistent memory: student profile, deadlines and saved notes (JSON file, atomic writes)."""
import json
import os
import tempfile
import uuid
from datetime import date, timedelta
from pathlib import Path

from . import config


class Memory:
    def __init__(self, path=None):
        self.path = Path(path or config.MEMORY_FILE)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {"profile": {}, "deadlines": [], "notes": []}
        if self.path.exists():
            try:
                self.data.update(json.loads(self.path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                pass  # corrupt file -> start fresh instead of crashing

    def _save(self):
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    # profile ---------------------------------------------------------------
    def set_profile(self, **kwargs):
        self.data["profile"].update({k: v for k, v in kwargs.items() if v not in (None, "")})
        self._save()

    def profile_text(self) -> str:
        p = self.data["profile"]
        return "; ".join(f"{k}: {v}" for k, v in p.items()) or "not provided"

    # deadlines -------------------------------------------------------------
    def add_deadline(self, title: str, due: date, course: str = "") -> dict:
        item = {"id": uuid.uuid4().hex[:8], "title": title, "due_date": due.isoformat(), "course": course}
        self.data["deadlines"].append(item)
        self._save()
        return item

    def list_deadlines(self, today: date, days_ahead: int | None = None) -> list[dict]:
        out = []
        for d in self.data["deadlines"]:
            due = date.fromisoformat(d["due_date"])
            left = (due - today).days
            if left < 0:
                continue
            if days_ahead is not None and due > today + timedelta(days=days_ahead):
                continue
            out.append({**d, "days_left": left})
        return sorted(out, key=lambda x: x["due_date"])

    def remove_deadline(self, deadline_id: str) -> bool:
        before = len(self.data["deadlines"])
        self.data["deadlines"] = [d for d in self.data["deadlines"] if d["id"] != deadline_id]
        self._save()
        return len(self.data["deadlines"]) < before

    # notes -----------------------------------------------------------------
    def add_note(self, title: str, path: str):
        self.data["notes"].append({"title": title, "path": path})
        self._save()
