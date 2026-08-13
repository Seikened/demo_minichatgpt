from __future__ import annotations

from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "src/demo_mini_chat/static/index.html"
APP_JS = ROOT / "src/demo_mini_chat/static/app.js"
STYLES = ROOT / "src/demo_mini_chat/static/styles.css"
MODES = ROOT / "src/demo_mini_chat/static/modes.css"
WEB = ROOT / "src/demo_mini_chat/web.py"
CLI = ROOT / "src/demo_mini_chat/cli.py"
README = ROOT / "README.md"
TEST = ROOT / "tests/test_presentation_contract.mjs"
WORKFLOW = ROOT / ".github/workflows/apply-demo-v6.yml"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label}; found {count}")
    return text.replace(old, new, 1)


def write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def patch_index() -> None:
    text = INDEX.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "<title>Más allá del prompt · Mini ChatGPT</title>",
        "<title>Más allá del prompt · Qué ve realmente un modelo</title>",
        "browser title",
    )
    text = replace_once(
        text,
        """        <h1>¿Qué ve realmente un modelo?</h1>\n        <p>Texto → tokens → probabilidades → siguiente token.</p>""",
        """        <h1>¿Qué ve realmente un modelo?</h1>\n        <p class=\"hero-subtitle\">Inteligencia artificial generativa en procesamiento de lenguaje natural.</p>\n        <p class=\"hero-flow\">Texto → tokens → probabilidades → siguiente token.</p>""",
        "hero copy",
    )
    text = replace_once(
        text,
        "        <div id=\"token-strip\" class=\"token-strip\"></div>\n",
        "",
        "duplicate input-token strip",
    )

    lower_grid = re.compile(
        r'    <section class="lower-grid">\s*'
        r'<article class="card concept-card">.*?</article>\s*'
        r'(<article class="card model-card">.*?</article>)\s*'
        r'</section>',
        re.DOTALL,
    )
    match = lower_grid.search(text)
    if match is None:
        raise RuntimeError("Could not find the lower concept/model section")
    model_card = match.group(1)
    replacement = f'''    <section class="model-section">\n      {model_card}\n    </section>\n\n    <footer class="credits card">\n      <div class="credits-primary">\n        <span class="kicker">CRÉDITOS</span>\n        <p><strong>Concepto, dirección y desarrollo del código:</strong> Fernando Leon Franco.</p>\n        <p><strong>Diseño e implementación de la interfaz:</strong> Fernando Leon Franco, con asistencia de ChatGPT (OpenAI).</p>\n      </div>\n      <div class="credits-model">\n        <p><strong>Familia y arquitectura GPT-2:</strong> OpenAI.</p>\n        <p><strong>Pesos en español utilizados:</strong> <code>mrm8488/spanish-gpt2</code>.</p>\n        <p class="credits-note">Los componentes de terceros conservan sus licencias y atribuciones.</p>\n      </div>\n    </footer>'''
    text = text[: match.start()] + replacement + text[match.end() :]
    write(INDEX, text)


def patch_app() -> None:
    text = APP_JS.read_text(encoding="utf-8")
    text = replace_once(
        text,
        """    sentence.append(user, model);\n  }""",
        """    sentence.append(user, model);\n    requestAnimationFrame(() => {\n      sentence.scrollTop = sentence.scrollHeight;\n    });\n  }""",
        "sentence autoscroll hook",
    )
    write(APP_JS, text)


def patch_branding() -> None:
    web = WEB.read_text(encoding="utf-8")
    web = replace_once(
        web,
        'app = FastAPI(title="Más allá del prompt · Mini ChatGPT", docs_url=None, redoc_url=None)',
        'app = FastAPI(title="Más allá del prompt · Qué ve realmente un modelo", docs_url=None, redoc_url=None)',
        "FastAPI title",
    )
    web = replace_once(
        web,
        'print("Mini ChatGPT listo en http://127.0.0.1:8080")',
        'print("Demo de lenguaje lista en http://127.0.0.1:8080")',
        "server startup message",
    )
    write(WEB, web)

    cli = CLI.read_text(encoding="utf-8")
    cli = replace_once(
        cli,
        'console.rule("[bold]Mini ChatGPT · siguiente token")',
        'console.rule("[bold]Modelo de lenguaje · siguiente token")',
        "CLI title",
    )
    write(CLI, cli)

    readme = README.read_text(encoding="utf-8")
    readme = replace_once(
        readme,
        "# Demo Mini ChatGPT",
        "# Más allá del prompt · Modelo de lenguaje visual",
        "README title",
    )
    write(README, readme)


def append_styles() -> None:
    styles = STYLES.read_text(encoding="utf-8")
    marker = "/* demo-v6-presentation-credits */"
    if marker in styles:
        raise RuntimeError("V6 style marker already exists")
    styles += '''\n\n/* demo-v6-presentation-credits */\n.hero-subtitle {\n  margin: 10px 0 0;\n  color: #263b55;\n  font-size: clamp(1rem, 1.45vw, 1.28rem);\n  font-weight: 650;\n  letter-spacing: -0.015em;\n}\n\n.hero-flow {\n  margin: 7px 0 0;\n  color: #526176;\n  font-size: 0.92rem;\n  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;\n}\n\n.model-section { margin-bottom: 16px; }\n.model-section .model-card { padding: 22px; }\n\n.credits {\n  display: grid;\n  grid-template-columns: minmax(0, 1.25fr) minmax(300px, 0.75fr);\n  gap: 26px;\n  padding: 20px 22px;\n  align-items: start;\n}\n\n.credits p {\n  margin: 7px 0 0;\n  color: #43546a;\n  font-size: 0.82rem;\n  line-height: 1.5;\n}\n\n.credits strong { color: #172b45; }\n.credits-model { padding-left: 24px; border-left: 1px solid var(--line-soft); }\n.credits code {\n  padding: 2px 6px;\n  border: 1px solid #c6d3e1;\n  border-radius: 7px;\n  background: #f2f6fb;\n  color: #154f9c;\n}\n.credits .credits-note { color: #66768a; font-size: 0.72rem; }\n\n@media (max-width: 800px) {\n  .credits { grid-template-columns: 1fr; }\n  .credits-model { padding-left: 0; padding-top: 16px; border-left: 0; border-top: 1px solid var(--line-soft); }\n}\n'''
    write(STYLES, styles)

    modes = MODES.read_text(encoding="utf-8")
    mode_marker = "/* demo-v6-sentence-autoscroll */"
    if mode_marker in modes:
        raise RuntimeError("V6 mode marker already exists")
    modes += '''\n\n/* demo-v6-sentence-autoscroll */\n#autocomplete-view .sentence-card.is-stuck .sentence {\n  scroll-behavior: smooth;\n  overscroll-behavior: contain;\n  scrollbar-gutter: stable;\n}\n'''
    write(MODES, modes)


def write_contract_test() -> None:
    TEST.write_text(
        '''import assert from 'node:assert/strict';\nimport test from 'node:test';\nimport fs from 'node:fs';\n\nconst index = fs.readFileSync('src/demo_mini_chat/static/index.html', 'utf8');\nconst app = fs.readFileSync('src/demo_mini_chat/static/app.js', 'utf8');\nconst web = fs.readFileSync('src/demo_mini_chat/web.py', 'utf8');\n\ntest('presentation branding is precise and removes Mini ChatGPT', () => {\n  assert.ok(index.includes('Inteligencia artificial generativa en procesamiento de lenguaje natural.'));\n  assert.ok(!index.includes('Mini ChatGPT'));\n  assert.ok(!web.includes('Mini ChatGPT'));\n});\n\ntest('presentation removes duplicated token strip and concept card', () => {\n  assert.ok(!index.includes('id="token-strip"'));\n  assert.ok(!index.includes('IDEA CLAVE'));\n  assert.ok(!index.includes('class="card concept-card"'));\n});\n\ntest('credits identify project author and model provenance', () => {\n  assert.ok(index.includes('Fernando Leon Franco'));\n  assert.ok(index.includes('ChatGPT (OpenAI)'));\n  assert.ok(index.includes('Familia y arquitectura GPT-2'));\n  assert.ok(index.includes('mrm8488/spanish-gpt2'));\n});\n\ntest('current sentence follows its latest generated content', () => {\n  assert.ok(app.includes('sentence.scrollTop = sentence.scrollHeight'));\n});\n''',
        encoding="utf-8",
    )


def validate() -> None:
    run("node", "--check", "src/demo_mini_chat/static/state.js")
    run("node", "--check", "src/demo_mini_chat/static/visuals.js")
    run("node", "--check", "src/demo_mini_chat/static/app.js")
    run("node", "--test", "tests/test_state.mjs", "tests/test_presentation_contract.mjs")
    run("python", "-m", "compileall", "-q", "src/demo_mini_chat")


def cleanup() -> None:
    WORKFLOW.unlink(missing_ok=True)
    Path(__file__).unlink(missing_ok=True)
    try:
        (ROOT / "tools").rmdir()
    except OSError:
        pass


def main() -> None:
    patch_index()
    patch_app()
    patch_branding()
    append_styles()
    write_contract_test()
    validate()
    cleanup()


if __name__ == "__main__":
    main()
