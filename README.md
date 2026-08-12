# Demo Mini ChatGPT

Demo didáctica e interactiva de un **modelo pequeño de lenguaje** para la charla _Más allá del prompt_. El objetivo no es recrear ChatGPT: es hacer visible, en una red diminuta, la misma idea fundamental de **usar contexto para producir una distribución sobre el siguiente token**.

La base viene del notebook `procesamiento_lenguaje/modelo_red_lenguaje.ipynb` de `Seikened/semestre_v`: N-gramas de orden 4, embeddings, una red feed-forward, logits, softmax y generación token por token. Esta versión separa esas piezas en módulos y agrega una visualización en vivo.

## Ejecutar

```bash
uv sync
uv run python main.py
```

También puede correrse directamente:

```bash
uv run demo-mini-chat --seed "la inteligencia artificial" --max-tokens 20
```

La primera ejecución entrena un modelo pequeño y guarda el checkpoint en `.artifacts/mini_language_model.pt`. Las siguientes ejecuciones lo cargan directamente.

Para volver a entrenar:

```bash
uv run demo-mini-chat --retrain
```

## Qué se ve en pantalla

Mientras el texto se genera aparecen simultáneamente:

- la frase actual;
- la ventana de contexto de tres tokens;
- los siguientes tokens con mayor probabilidad;
- la distribución producida por softmax;
- el token seleccionado;
- palabras cercanas en el espacio de embeddings;
- el modo de decodificación y la temperatura.

El modo por defecto usa **muestreo probabilístico**. Para comparar con una salida determinista:

```bash
uv run demo-mini-chat --deterministic --seed "el modelo"
```

En el modo interactivo existen los comandos `:temp 0.7`, `:mode sample`, `:mode greedy` y `:q`.

## Corpus

El notebook original utilizaba tweets de MEX-A3T. Esta demo usa un corpus sintético, pequeño y limpio para que el entrenamiento sea rápido y el contenido pueda proyectarse sin sorpresas. La arquitectura didáctica se mantiene; el dataset se cambió deliberadamente para la presentación.

## Pruebas

```bash
uv run python -m unittest discover -s tests -v
```
