from __future__ import annotations

import base64
import lzma
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "tools" / "demo_v5_chunks"
SCRIPT = ROOT / "tools" / "apply_demo_v5.py"


def main() -> None:
    encoded = "".join(path.read_text(encoding="ascii").strip() for path in sorted(CHUNKS.glob("chunk_*.txt")))
    SCRIPT.write_bytes(lzma.decompress(base64.b64decode(encoded, validate=True)))
    shutil.rmtree(CHUNKS)
    Path(__file__).unlink(missing_ok=True)
    subprocess.run(["python", str(SCRIPT)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
