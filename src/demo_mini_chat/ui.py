from time import sleep

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .engine import GenerationStep, MiniLanguageEngine


console = Console()


def _probability_bar(probability: float, width: int = 24) -> str:
    filled = min(width, max(0, round(probability * width)))
    return "█" * filled + "·" * (width - filled)


def render_state(
    text: str,
    step: GenerationStep,
    temperature: float,
    deterministic: bool,
) -> Group:
    mode = "argmax / determinista" if deterministic else "muestreo / probabilístico"
    context = "  ·  ".join(step.context)

    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("token")
    table.add_column("probabilidad", justify="right")
    table.add_column("distribución")
    for candidate in step.candidates:
        marker = "→ " if candidate.token == step.chosen.token else "  "
        table.add_row(
            f"{marker}{candidate.token}",
            f"{candidate.probability:6.2%}",
            _probability_bar(candidate.probability),
        )

    neighbor_text = " · ".join(f"{word} ({distance:.2f})" for word, distance in step.neighbors)
    if not neighbor_text:
        neighbor_text = "—"

    meta = Text()
    meta.append(f"modo: {mode}   ")
    meta.append(f"temperatura: {temperature:.2f}   ")
    meta.append(f"elegida: {step.chosen.token} ({step.chosen.probability:.2%})")

    return Group(
        Panel(text or " ", title="texto generado", border_style="bright_blue"),
        Panel(context, title="ventana de contexto (3 tokens)", border_style="cyan"),
        Panel(table, title="softmax · siguientes tokens", border_style="magenta"),
        Panel(neighbor_text, title="vecinos en el embedding", border_style="green"),
        Panel(meta, title="decodificación", border_style="yellow"),
    )


def animate_generation(
    engine: MiniLanguageEngine,
    seed: str,
    *,
    max_tokens: int,
    temperature: float,
    deterministic: bool,
    delay: float,
) -> str:
    text = engine.normalized_seed(seed)

    with Live(console=console, refresh_per_second=12, transient=False) as live:
        for _ in range(max_tokens):
            step = engine.next_step(
                text,
                temperature=temperature,
                deterministic=deterministic,
            )
            live.update(render_state(text, step, temperature, deterministic), refresh=True)
            sleep(delay)

            if step.chosen.token == engine.data.EOS:
                break
            text = engine.append_token(text, step.chosen.token)
            live.update(render_state(text, step, temperature, deterministic), refresh=True)
            sleep(delay)

    return text
