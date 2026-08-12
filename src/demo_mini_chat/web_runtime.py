from __future__ import annotations

import asyncio

import numpy as np
from nicegui import ui

from .config import DemoConfig
from .engine import MiniLanguageEngine
from .web import CHECKPOINT, DemoPage, _display_token


def _update_chart(chart, options: dict[str, object]) -> None:
    """Update NiceGUI EChart options through the supported mutable mapping API."""
    chart.options.update(options)
    chart.update()


class RuntimeDemoPage(DemoPage):
    """NiceGUI 3.x runtime fixes for the visual demo."""

    def refresh_distribution(self, chosen: str | None) -> np.ndarray:
        probs, context = self.engine.distribution(self.text, temperature=self.temperature)
        self.cloud.set_content(self._cloud_html(probs, chosen))
        self.cloud_meta.set_text(f"{len(self._ranked_ids(probs))} palabras · tamaño = probabilidad relativa")
        _update_chart(self.bar_chart, self._bar_options(probs, chosen))
        _update_chart(self.embedding_chart, self._embedding_options(chosen))
        context_text = "  →  ".join(_display_token(token) for token in context)
        self.phase_label.set_text(f"contexto actual: {context_text}")
        return probs

    async def advance(self) -> None:
        if self.busy:
            return
        self.busy = True
        try:
            probs = self.refresh_distribution(chosen=None)
            self.chosen_label.set_text("elegida: todavía ninguna")
            self.phase_label.set_text("softmax → ya existe una distribución sobre el vocabulario")
            await asyncio.sleep(max(0.15, self.delay * 0.34))

            step = self.engine.next_step(
                self.text,
                temperature=self.temperature,
                deterministic=self.deterministic,
                top_k=10,
            )
            self.cloud.set_content(self._cloud_html(probs, step.chosen.token))
            _update_chart(self.bar_chart, self._bar_options(probs, step.chosen.token))
            _update_chart(self.embedding_chart, self._embedding_options(step.chosen.token))

            mode = "argmax" if self.deterministic else "muestreo"
            self.chosen_label.set_text(f"elegida: {_display_token(step.chosen.token)} · {step.chosen.probability:.2%}")
            self.phase_label.set_text(f"selección ({mode}) → {_display_token(step.chosen.token)}")
            await asyncio.sleep(max(0.15, self.delay * 0.33))

            if step.chosen.token == self.engine.data.EOS:
                self.playing = False
                self.play_button.set_text("Reproducir")
                self.phase_label.set_text("el modelo eligió ⟨FIN⟩ · generación terminada")
                return

            self.text = self.engine.append_token(self.text, step.chosen.token)
            self.step_count += 1
            self.sentence_label.set_text(self.text)
            self.step_label.set_text(f"paso {self.step_count}")
            self.phase_label.set_text("la palabra elegida entra ahora al contexto del siguiente paso")
            await asyncio.sleep(max(0.15, self.delay * 0.33))
        finally:
            self.busy = False


def main() -> None:
    engine = MiniLanguageEngine.load_or_train(CHECKPOINT, config=DemoConfig())

    @ui.page("/")
    def index() -> None:
        RuntimeDemoPage(engine).build()

    ui.run(
        title="Más allá del prompt · Mini ChatGPT",
        favicon="🧠",
        port=8080,
        reload=False,
        show=True,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
