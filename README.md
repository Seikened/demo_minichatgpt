# Demo Mini ChatGPT

Demo didáctica para **Más allá del prompt**: observar, paso a paso, cómo un modelo de lenguaje convierte contexto en una distribución de probabilidad y selecciona el siguiente token.

## Ejecutar

```bash
uv sync
uv run python main.py
```

Abre `http://127.0.0.1:8080`.

La primera ejecución de la V2 descarga y carga `mrm8488/spanish-gpt2`. Hugging Face lo deja en caché local para ejecuciones posteriores. Si el modelo no puede cargarse, la pantalla permite usar el modelo pequeño de la clase.

## V2 visual

La interfaz está hecha con HTML/CSS/JavaScript nativo, D3 y un backend local FastAPI. La oración es editable en cualquier momento: el siguiente paso siempre se recalcula usando exactamente el texto que esté escrito.

Cada token generado guarda su `GenerationState`: texto anterior y posterior, tokenización, candidatos, probabilidades, token elegido, temperatura y estrategia de selección. En la interfaz puedes hacer clic en cualquier token del historial y volver a inspeccionar el estado en el que fue generado.

El flujo visual principal es:

```text
texto → tokens → modelo → logits → softmax → probabilidades → selección → nuevo token
```

El universo de tokens usa una escala visual suavizada para que no desaparezcan todas las alternativas cuando existe un candidato dominante. Se muestran los candidatos principales y, por separado, la masa de probabilidad del resto del vocabulario.

## Modelos

### Spanish GPT-2

Modelo principal de la V2: `mrm8488/spanish-gpt2`, un GPT-2 pequeño entrenado desde cero en español sobre `large_spanish_corpus` (~20 GB). Tiene un vocabulario BPE mucho mayor que la demo de clase y permite observar subpalabras/tokens reales.

### Modelo de clase

Se conserva la red del notebook `Seikened/semestre_v/procesamiento_lenguaje/modelo_red_lenguaje.ipynb` como comparación didáctica.

La versión compacta actual usa 792 frases sintéticas, 5,182 tokens observados, 141 tokens únicos + 3 especiales (144 de vocabulario) y 5,974 ejemplos `contexto → siguiente token`.

## Terminal

La demo anterior con Rich sigue disponible:

```bash
uv run demo-mini-chat-cli
```

## Pruebas

```bash
uv run python -m unittest discover -s tests -v
```
