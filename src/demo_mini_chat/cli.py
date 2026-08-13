from argparse import ArgumentParser
from pathlib import Path

from rich.console import Console

from .config import DemoConfig
from .engine import MiniLanguageEngine
from .ui import animate_generation


console = Console()
CHECKPOINT = Path(".artifacts/mini_language_model.pt")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Mini language-model demo")
    parser.add_argument("--seed", help="Text to continue. If omitted, starts interactive mode.")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--deterministic", action="store_true", help="Use argmax instead of sampling.")
    parser.add_argument("--retrain", action="store_true", help="Ignore the cached checkpoint and train again.")
    return parser


def run_once(
    engine: MiniLanguageEngine,
    seed: str,
    temperature: float,
    max_tokens: int,
    delay: float,
    deterministic: bool,
) -> None:
    console.rule("[bold]Modelo de lenguaje · siguiente token")
    result = animate_generation(
        engine,
        seed,
        max_tokens=max_tokens,
        temperature=temperature,
        deterministic=deterministic,
        delay=delay,
    )
    console.print(f"\n[bold]Resultado:[/] {result}\n")


def interactive_loop(
    engine: MiniLanguageEngine,
    temperature: float,
    max_tokens: int,
    delay: float,
    deterministic: bool,
) -> None:
    console.print("[bold]Escribe una frase y el modelo la continuará.[/]")
    console.print("Comandos: [cyan]:q[/], [cyan]:temp 0.7[/], [cyan]:mode sample[/], [cyan]:mode greedy[/]\n")

    while True:
        seed = console.input("[bold blue]tú > [/]").strip()
        if seed == ":q":
            return
        if seed.startswith(":temp "):
            try:
                temperature = float(seed.split(maxsplit=1)[1])
                console.print(f"temperatura = {temperature:.2f}\n")
            except ValueError:
                console.print("[red]Temperatura inválida.[/]\n")
            continue
        if seed == ":mode sample":
            deterministic = False
            console.print("modo = muestreo probabilístico\n")
            continue
        if seed == ":mode greedy":
            deterministic = True
            console.print("modo = argmax determinista\n")
            continue
        if not seed:
            continue

        run_once(engine, seed, temperature, max_tokens, delay, deterministic)


def main() -> None:
    args = build_parser().parse_args()
    config = DemoConfig()

    if CHECKPOINT.exists() and not args.retrain:
        console.print("[dim]Cargando mini modelo entrenado...[/]")
    else:
        console.print("[dim]Entrenando mini modelo didáctico...[/]")

    engine = MiniLanguageEngine.load_or_train(CHECKPOINT, config=config, retrain=args.retrain)
    if engine.training is not None:
        console.print(
            f"[dim]Entrenamiento listo: {engine.training.epochs} épocas · "
            f"loss {engine.training.best_loss:.4f} · vocabulario {engine.data.size}[/]\n"
        )

    if args.seed:
        run_once(
            engine,
            args.seed,
            args.temperature,
            args.max_tokens,
            args.delay,
            args.deterministic,
        )
    else:
        interactive_loop(
            engine,
            args.temperature,
            args.max_tokens,
            args.delay,
            args.deterministic,
        )
