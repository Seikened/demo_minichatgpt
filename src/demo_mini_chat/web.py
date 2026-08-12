from __future__ import annotations

import asyncio
import html
from pathlib import Path

import numpy as np
from nicegui import ui

from .config import DemoConfig
from .engine import MiniLanguageEngine


CHECKPOINT = Path(".artifacts/mini_language_model.pt")
DEFAULT_SEED = "la inteligencia artificial"


def _display_token(token: str) -> str:
    return {"</s>": "⟨FIN⟩", "<unk>": "⟨UNK⟩", "<s>": "⟨INICIO⟩"}.get(token, token)


def _embedding_projection(engine: MiniLanguageEngine) -> np.ndarray:
    weights = engine.model.emb.weight.detach().cpu().numpy().astype(np.float64)
    centered = weights - weights.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh[:2].T
    scale = float(np.abs(coords).max()) or 1.0
    return coords / scale


class DemoPage:
    def __init__(self, engine: MiniLanguageEngine) -> None:
        self.engine = engine
        self.coords = _embedding_projection(engine)
        self.text = DEFAULT_SEED
        self.temperature = 0.85
        self.delay = 3.0
        self.deterministic = False
        self.playing = False
        self.busy = False
        self.step_count = 0

    def build(self) -> None:
        ui.dark_mode().enable()
        ui.page_title("Más allá del prompt · Mini ChatGPT")
        ui.add_head_html(
            """
            <style>
              body { background: #080b12; }
              .demo-shell { max-width: 1500px; margin: 0 auto; }
              .glass { background: rgba(17,24,39,.72); border: 1px solid rgba(148,163,184,.16);
                       border-radius: 20px; box-shadow: 0 18px 50px rgba(0,0,0,.22); backdrop-filter: blur(12px); }
              .word-cloud { min-height: 250px; display:flex; align-content:center; align-items:center;
                            justify-content:center; flex-wrap:wrap; gap:12px 16px; padding:26px; line-height:1; }
              .word-cloud span { display:inline-block; transition:transform .35s ease, opacity .35s ease; }
              .word-cloud .chosen { text-shadow:0 0 22px rgba(96,165,250,.9); transform:scale(1.16); font-weight:800; }
              .pipeline { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:8px; }
              .pipeline-step { border:1px solid rgba(148,163,184,.18); border-radius:14px; padding:10px 8px;
                               text-align:center; font-size:.78rem; color:#cbd5e1; background:rgba(15,23,42,.72); }
              .metric { font-variant-numeric:tabular-nums; color:#93c5fd; }
              @media (max-width:900px) { .pipeline { grid-template-columns:repeat(3,minmax(0,1fr)); } }
            </style>
            """
        )

        with ui.column().classes("demo-shell w-full px-6 py-6 gap-5"):
            with ui.row().classes("w-full items-end justify-between gap-3"):
                with ui.column().classes("gap-0"):
                    ui.label("MÁS ALLÁ DEL PROMPT").classes("text-xs tracking-[0.32em] text-slate-400")
                    ui.label("Mini modelo de lenguaje").classes("text-3xl md:text-5xl font-bold")
                    ui.label("Observa cómo una distribución de probabilidad se convierte en una oración.").classes(
                        "text-slate-400 text-base"
                    )
                ui.label(
                    f"N-grama {self.engine.config.order} · ventana {self.engine.config.window_size} · "
                    f"vocabulario {self.engine.data.size}"
                ).classes("text-sm text-slate-400")

            with ui.card().classes("glass w-full p-5"):
                with ui.row().classes("w-full items-end gap-4"):
                    self.seed_input = ui.input(
                        "Texto inicial", value=self.text, placeholder="Escribe el inicio de una frase"
                    ).classes("grow min-w-[320px]")
                    ui.button("Reiniciar", icon="restart_alt", on_click=self.reset).props("outline")
                    ui.button("Siguiente token", icon="skip_next", on_click=self.advance)
                    self.play_button = ui.button("Reproducir", icon="play_arrow", on_click=self.toggle_play).props(
                        "outline"
                    )

                with ui.row().classes("w-full items-center gap-6 mt-3"):
                    with ui.column().classes("grow min-w-[240px] gap-1"):
                        ui.label("Temperatura").classes("text-xs text-slate-400")
                        slider = ui.slider(min=0.2, max=2.0, step=0.05, value=self.temperature).props("label-always")
                        slider.on_value_change(self.on_temperature)
                    with ui.column().classes("grow min-w-[240px] gap-1"):
                        ui.label("Pausa didáctica por token (segundos)").classes("text-xs text-slate-400")
                        speed = ui.slider(min=0.5, max=6.0, step=0.5, value=self.delay).props("label-always")
                        speed.on_value_change(self.on_delay)
                    with ui.column().classes("min-w-[250px] gap-1"):
                        ui.label("Decodificación").classes("text-xs text-slate-400")
                        mode = ui.toggle({"sample": "Muestreo", "greedy": "Argmax"}, value="sample")
                        mode.on_value_change(self.on_mode)

            with ui.card().classes("glass w-full p-6"):
                ui.label("ORACIÓN").classes("text-xs tracking-[0.2em] text-slate-500")
                self.sentence_label = ui.label(self.text).classes("text-3xl md:text-5xl font-semibold leading-tight py-3")
                with ui.row().classes("items-center gap-5"):
                    self.phase_label = ui.label("Listo para calcular la siguiente palabra").classes("text-slate-400")
                    self.chosen_label = ui.label("elegida: —").classes("metric font-semibold")
                    self.step_label = ui.label("paso 0").classes("text-slate-500")

            ui.html(self._pipeline_html()).classes("w-full")

            with ui.row().classes("w-full gap-5 items-stretch"):
                with ui.card().classes("glass grow basis-[62%] min-w-[600px] p-5"):
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label("Universo de palabras").classes("text-xl font-semibold")
                        self.cloud_meta = ui.label("probabilidad del siguiente token").classes("text-xs text-slate-500")
                    self.cloud = ui.html("", sanitize=False).classes("word-cloud w-full")
                with ui.card().classes("glass grow basis-[34%] min-w-[390px] p-5"):
                    ui.label("Top de candidatos").classes("text-xl font-semibold")
                    self.bar_chart = ui.echart(self._empty_bar_options()).classes("w-full h-[340px]")

            with ui.row().classes("w-full gap-5 items-stretch"):
                with ui.card().classes("glass grow basis-[58%] min-w-[560px] p-5"):
                    with ui.row().classes("w-full justify-between items-center"):
                        ui.label("Espacio de embeddings").classes("text-xl font-semibold")
                        ui.label("PCA 2D solo para visualizar").classes("text-xs text-slate-500")
                    self.embedding_chart = ui.echart(self._embedding_options(None)).classes("w-full h-[370px]")
                with ui.card().classes("glass grow basis-[38%] min-w-[420px] p-5"):
                    ui.label("Qué ocurre detrás").classes("text-xl font-semibold")
                    cfg = self.engine.config
                    ui.html(
                        f"""
                        <div class="text-slate-300 leading-7 mt-3">
                          <div><b>1.</b> Contexto: <span class="metric">{cfg.window_size} tokens</span></div>
                          <div><b>2.</b> Embedding: token → <span class="metric">{cfg.embedding_dim} dimensiones</span></div>
                          <div><b>3.</b> Concatenación: <span class="metric">{cfg.window_size * cfg.embedding_dim}</span> valores</div>
                          <div><b>4.</b> Capa oculta: <span class="metric">{cfg.hidden_dim}</span> neuronas + ReLU</div>
                          <div><b>5.</b> Salida: un logit por palabra del vocabulario</div>
                          <div><b>6.</b> Softmax: logits → distribución de probabilidad</div>
                        </div>
                        """,
                        sanitize=False,
                    )
                    with ui.expansion("Ver el núcleo del paso", icon="code").classes("w-full mt-4"):
                        ui.code(
                            """x = embedding(context_ids)
x = flatten(x)
h = relu(fc1(x))
logits = fc2(h)
probs = softmax(logits / temperature)
next_token = sample(probs)  # o argmax"""
                        ).classes("w-full")

            self.refresh_distribution(chosen=None)

    def _pipeline_html(self) -> str:
        labels = ["1 · contexto", "2 · embeddings", "3 · red neuronal", "4 · logits", "5 · softmax", "6 · selección"]
        return '<div class="pipeline">' + "".join(f'<div class="pipeline-step">{label}</div>' for label in labels) + "</div>"

    def _ranked_ids(self, probs: np.ndarray) -> list[int]:
        return [
            int(idx)
            for idx in np.argsort(probs)[::-1]
            if self.engine.data.id_to_word[int(idx)] != self.engine.data.SOS
        ]

    def _cloud_html(self, probs: np.ndarray, chosen: str | None) -> str:
        ranked = self._ranked_ids(probs)
        max_probability = max((float(probs[idx]) for idx in ranked), default=1.0) or 1.0
        spans: list[str] = []
        for idx in ranked:
            token = self.engine.data.id_to_word[idx]
            probability = float(probs[idx])
            ratio = probability / max_probability
            size = 12 + 34 * (ratio ** 0.42)
            opacity = 0.22 + 0.78 * (ratio ** 0.35)
            weight = 350 + int(350 * ratio)
            cls = "chosen" if token == chosen else ""
            label = html.escape(_display_token(token))
            title = html.escape(f"{_display_token(token)} · {probability:.4%}")
            spans.append(
                f'<span class="{cls}" title="{title}" style="font-size:{size:.1f}px;opacity:{opacity:.3f};font-weight:{weight};">{label}</span>'
            )
        return "".join(spans)

    def _empty_bar_options(self) -> dict[str, object]:
        return {
            "animationDurationUpdate": 500,
            "grid": {"left": 90, "right": 42, "top": 15, "bottom": 28},
            "xAxis": {
                "type": "value",
                "axisLabel": {"formatter": "{value}%"},
                "splitLine": {"lineStyle": {"opacity": 0.12}},
            },
            "yAxis": {"type": "category", "inverse": True, "data": []},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "series": [
                {
                    "type": "bar",
                    "data": [],
                    "barMaxWidth": 24,
                    "label": {"show": True, "position": "right", "formatter": "{c}%"},
                }
            ],
        }

    def _bar_options(self, probs: np.ndarray, chosen: str | None) -> dict[str, object]:
        ranked = self._ranked_ids(probs)[:10]
        tokens = [_display_token(self.engine.data.id_to_word[idx]) for idx in ranked]
        values = [round(float(probs[idx]) * 100, 2) for idx in ranked]
        data: list[object] = []
        for idx, value in zip(ranked, values, strict=True):
            token = self.engine.data.id_to_word[idx]
            item: dict[str, object] = {"value": value}
            if token == chosen:
                item["itemStyle"] = {"borderWidth": 2, "borderColor": "#ffffff"}
            data.append(item)
        options = self._empty_bar_options()
        options["yAxis"]["data"] = tokens
        options["series"][0]["data"] = data
        return options

    def _embedding_options(self, chosen: str | None) -> dict[str, object]:
        special = {self.engine.data.SOS, self.engine.data.EOS, self.engine.data.UNK}
        neighbor_tokens = {word for word, _ in self.engine.closest_words(chosen, k=7)} if chosen else set()
        background: list[dict[str, object]] = []
        neighbors: list[dict[str, object]] = []
        selected: list[dict[str, object]] = []

        for idx, token in self.engine.data.id_to_word.items():
            if token in special:
                continue
            point = {"name": token, "value": [round(float(self.coords[idx, 0]), 4), round(float(self.coords[idx, 1]), 4)]}
            if token == chosen:
                selected.append(point)
            elif token in neighbor_tokens:
                neighbors.append(point)
            else:
                background.append(point)

        return {
            "animationDurationUpdate": 650,
            "grid": {"left": 18, "right": 18, "top": 18, "bottom": 18},
            "xAxis": {"type": "value", "show": False, "min": -1.05, "max": 1.05},
            "yAxis": {"type": "value", "show": False, "min": -1.05, "max": 1.05},
            "tooltip": {"formatter": "{b}"},
            "series": [
                {"type": "scatter", "data": background, "symbolSize": 7, "itemStyle": {"opacity": 0.22}},
                {
                    "type": "scatter",
                    "data": neighbors,
                    "symbolSize": 15,
                    "label": {"show": True, "formatter": "{b}", "position": "top"},
                    "itemStyle": {"opacity": 0.82},
                },
                {
                    "type": "scatter",
                    "data": selected,
                    "symbolSize": 26,
                    "label": {"show": True, "formatter": "{b}", "position": "top", "fontWeight": "bold"},
                },
            ],
        }

    def refresh_distribution(self, chosen: str | None) -> np.ndarray:
        probs, context = self.engine.distribution(self.text, temperature=self.temperature)
        self.cloud.set_content(self._cloud_html(probs, chosen))
        self.cloud_meta.set_text(f"{len(self._ranked_ids(probs))} palabras · tamaño = probabilidad relativa")
        self.bar_chart.options = self._bar_options(probs, chosen)
        self.bar_chart.update()
        self.embedding_chart.options = self._embedding_options(chosen)
        self.embedding_chart.update()
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
            self.bar_chart.options = self._bar_options(probs, step.chosen.token)
            self.bar_chart.update()
            self.embedding_chart.options = self._embedding_options(step.chosen.token)
            self.embedding_chart.update()

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

    async def toggle_play(self) -> None:
        if self.playing:
            self.playing = False
            self.play_button.set_text("Reproducir")
            return

        self.playing = True
        self.play_button.set_text("Pausar")
        while self.playing and self.step_count < 60:
            await self.advance()
            await asyncio.sleep(0)
        self.playing = False
        self.play_button.set_text("Reproducir")

    def reset(self) -> None:
        self.playing = False
        self.busy = False
        self.step_count = 0
        self.text = self.engine.normalized_seed(str(self.seed_input.value or DEFAULT_SEED))
        self.sentence_label.set_text(self.text)
        self.chosen_label.set_text("elegida: —")
        self.step_label.set_text("paso 0")
        self.play_button.set_text("Reproducir")
        self.refresh_distribution(chosen=None)
        self.phase_label.set_text("Listo para calcular la siguiente palabra")

    def on_temperature(self, event) -> None:
        self.temperature = float(event.value)
        if not self.busy:
            self.refresh_distribution(chosen=None)
            self.phase_label.set_text("temperatura actualizada · observa cómo cambia la distribución")

    def on_delay(self, event) -> None:
        self.delay = float(event.value)

    def on_mode(self, event) -> None:
        self.deterministic = event.value == "greedy"
        mode = "argmax determinista" if self.deterministic else "muestreo probabilístico"
        self.phase_label.set_text(f"modo: {mode}")


def main() -> None:
    engine = MiniLanguageEngine.load_or_train(CHECKPOINT, config=DemoConfig())

    @ui.page("/")
    def index() -> None:
        DemoPage(engine).build()

    ui.run(
        title="Más allá del prompt · Mini ChatGPT",
        favicon="🧠",
        port=8080,
        reload=False,
        show=True,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
