from __future__ import annotations

import base64
import io
from pathlib import Path
import shutil
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "tools" / "demo_v4_chunks"
EXPECTED = {
    "src/demo_mini_chat/static/state.js",
    "src/demo_mini_chat/static/visuals.js",
    "src/demo_mini_chat/static/app.js",
    "src/demo_mini_chat/static/index.html",
    "src/demo_mini_chat/static/styles.css",
    "src/demo_mini_chat/static/modes.css",
    "tests/test_state.mjs",
}


def run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    encoded = "".join(path.read_text(encoding="ascii").strip() for path in sorted(CHUNKS.glob("chunk_*.txt")))
    payload = base64.b64decode(encoded, validate=True)

    extracted: set[str] = set()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:xz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            relative = Path(member.name)
            target = (ROOT / relative).resolve()
            if relative.is_absolute() or ROOT.resolve() not in target.parents:
                raise RuntimeError(f"Unsafe archive path: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Missing archive content: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())
            extracted.add(relative.as_posix())

    if extracted != EXPECTED:
        raise RuntimeError(f"Unexpected payload files: expected={sorted(EXPECTED)} extracted={sorted(extracted)}")

    run("node", "--check", "src/demo_mini_chat/static/state.js")
    run("node", "--check", "src/demo_mini_chat/static/visuals.js")
    run("node", "--check", "src/demo_mini_chat/static/app.js")
    run("node", "--test", "tests/test_state.mjs")

    shutil.rmtree(CHUNKS)
    (ROOT / ".github" / "workflows" / "apply-demo-v4.yml").unlink(missing_ok=True)
    Path(__file__).unlink(missing_ok=True)
    try:
        (ROOT / "tools").rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
