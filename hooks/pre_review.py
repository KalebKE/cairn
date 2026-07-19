#!/usr/bin/env python3.11
"""Pre-commit review hook — fast risk assessment on staged changes.

Runs Revue's deterministic summarizer (no LLM, <100ms) and prints
a risk assessment. Does not block commits.

Usage:
    python3.11 hooks/pre_review.py          # Quick summary
    python3.11 hooks/pre_review.py --full   # Full LLM review (slower)
"""

import subprocess
import sys
import os


def get_staged_diff() -> str:
    """Get the staged diff."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--unified=3"],
        capture_output=True, text=True,
    )
    return result.stdout


def main():
    diff = get_staged_diff()
    if not diff or diff.count("\n") < 5:
        return 0

    full_mode = "--full" in sys.argv

    try:
        # Add revue to path
        revue_path = os.path.expanduser("~/Projects/revue")
        sys.path.insert(0, os.path.join(revue_path, "src"))

        if full_mode:
            # Full LLM review
            import asyncio
            from revue.engine import RevueEngine

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                try:
                    import json
                    with open(os.path.expanduser("~/.cairn/secrets.json")) as f:
                        api_key = json.load(f).get("ANTHROPIC_API_KEY")
                except Exception:
                    pass

            if not api_key:
                print("[revue] No API key — skipping full review")
                return 0

            engine = RevueEngine(api_key=api_key)
            result = asyncio.run(engine.review_diff(diff, mode="strict"))

            if result.findings:
                print(f"\n[revue] {result.review_summary_text}")
                for f in result.findings:
                    print(f"  [{f['severity'].upper()}] {f['title']}")
                    if f.get("file_path"):
                        print(f"    {f['file_path']}:{f.get('line_start', '')}")
                print()
        else:
            # Fast summary only (no LLM, <100ms)
            from revue.engine import RevueEngine
            engine = RevueEngine()
            summary = engine.summarize(diff)

            risk = summary.get("risk", "low")
            if risk in ("high", "critical"):
                print(f"\n[revue] Risk: {risk.upper()} — {summary.get('risk_reason', '')}")
                for focus in summary.get("review_focus", []):
                    print(f"  → {focus}")
                print()
            # low/medium risk = silent
    except ImportError:
        pass  # Revue not installed — silent skip
    except Exception as e:
        # Never block commits due to review errors
        print(f"[revue] Warning: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
