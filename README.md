# Demo Mini ChatGPT

Demo didáctica para la charla **Más allá del prompt**. Parte del pequeño modelo de lenguaje construido en `Seikened/semestre_v/procesamiento_lenguaje/modelo_red_lenguaje.ipynb` y hace visible el ciclo de predicción del siguiente token.

La arquitectura conserva la idea del ejercicio de clase: N-grama de orden 4, ventana de tres tokens, embeddings, red feed-forward, logits, softmax y generación token por token. Para la presentación se usa un corpus sintético limpio y pequeño, de modo que el modelo pueda entrenarse rápido y proyectarse sin sorpresas.

## Demo visual

```bash
uv sync
uv run python main.py
```

Se abrirá `http://localhost:8080` en el navegador.

La interfaz permite detener y observar cada token. Primero muestra la distribución completa sobre el vocabulario; después resalta la palabra elegida; finalmente incorpora esa palabra a la oración y vuelve a calcular el siguiente paso.

En pantalla se muestran simultáneamente:

- la oración que se está formando;
- la ventana actual de tres tokens;
- el universo completo de palabras, donde tamaño y opacidad representan probabilidad relativa;
- el Top-10 de candidatos con su probabilidad;
- la palabra finalmente elegida;
- un mapa PCA 2D de los embeddings y los vecinos de la palabra seleccionada;
- el flujo `contexto → embeddings → red → logits → softmax → selección`;
- temperatura y comparación entre muestreo probabilístico y `argmax` determinista.

La pausa didáctica puede ajustarse entre 0.5 y 6 segundos por token. También puede avanzarse manualmente con **Siguiente token**.

## Demo de terminal

La versión anterior con Rich sigue disponible como fallback:

```bash
uv run demo-mini-chat-cli
```

## Modelo

La primera ejecución entrena el modelo pequeño y guarda el checkpoint en `.artifacts/mini_language_model.pt`. Las siguientes ejecuciones cargan ese checkpoint.

Para forzar un nuevo entrenamiento desde la terminal:

```bash
uv run demo-mini-chat-cli --retrain
```

## Pruebas

```bash
uv run python -m unittest discover -s tests -v
```
