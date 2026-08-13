from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "demo_mini_chat" / "static" / "app.js"
INDEX = ROOT / "src" / "demo_mini_chat" / "static" / "index.html"
WORKFLOW = ROOT / ".github" / "workflows" / "fix-removed-token-strip.yml"


def run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    app = APP.read_text(encoding="utf-8")
    stale_statement = "    $('token-strip').innerHTML = '';\n"
    if app.count(stale_statement) != 1:
        raise RuntimeError("Expected exactly one stale token-strip dereference in app.js")

    index = INDEX.read_text(encoding="utf-8")
    if 'id="token-strip"' in index:
        raise RuntimeError("The token-strip element still exists; refusing the targeted cleanup")

    APP.write_text(app.replace(stale_statement, ""), encoding="utf-8")

    updated = APP.read_text(encoding="utf-8")
    if "$('token-strip')" in updated:
        raise RuntimeError("A direct token-strip dereference remains in app.js")

    run("node", "--check", str(APP.relative_to(ROOT)))
    run("node", "--test", "tests/test_state.mjs")

    WORKFLOW.unlink(missing_ok=True)
    Path(__file__).unlink(missing_ok=True)
    try:
        (ROOT / "tools").rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
