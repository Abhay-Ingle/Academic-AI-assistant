"""Command-line interface:  python cli.py [--verbose] [--load-samples]"""
import argparse

from agent import config
from agent.factory import create_agent


def main():
    ap = argparse.ArgumentParser(description="Academic AI Assistant (agentic)")
    ap.add_argument("--verbose", action="store_true", help="show plan / tool calls")
    ap.add_argument("--load-samples", action="store_true", help="index data/sample_notes first")
    args = ap.parse_args()

    def show(ev):
        if args.verbose:
            kind = ev["kind"]
            if kind == "plan":
                print(f"  [plan] {ev['intent']}: " + " -> ".join(s["tool"] or "answer" for s in ev["plan"]))
            elif kind == "tool":
                print(f"  [tool] {ev['tool']} {'ok' if ev['ok'] else 'FAILED'}")
            elif kind in ("blocked", "warning", "replan", "reflection"):
                print(f"  [{kind}] {ev}")

    agent = create_agent(on_event=show)
    if args.load_samples:
        print(f"Indexed {agent.kb.add_folder(config.SAMPLE_DIR)} new chunks from sample notes.")
    print("Academic AI Assistant - type 'exit' to quit.")
    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"exit", "quit"}:
            break
        r = agent.run(text)
        print(f"\nScholar: {r.answer}")
        if r.files:
            print("Files:", ", ".join(r.files))
        print(f"({r.llm_calls} LLM calls, {r.seconds}s, tools: {', '.join(r.tools_used) or 'none'})")


if __name__ == "__main__":
    main()
