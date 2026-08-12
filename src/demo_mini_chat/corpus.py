from itertools import product


def build_demo_corpus() -> list[str]:
    """Build a clean synthetic Spanish corpus with obvious learnable patterns.

    The original notebook trained on MEX-A3T tweets. For a public university demo
    we keep the same modelling idea while using a small, deterministic corpus that
    is safe to display and fast to train.
    """

    fixed = [
        "la inteligencia artificial aprende patrones del lenguaje",
        "un modelo de lenguaje predice la siguiente palabra",
        "el modelo observa contexto y calcula probabilidades",
        "una probabilidad alta no significa una verdad absoluta",
        "el contexto cambia la probabilidad de la siguiente palabra",
        "los modelos aprenden relaciones estadísticas entre palabras",
        "la ingeniería combina modelos con reglas y sistemas",
        "un sistema confiable necesita validación y evidencia",
        "la inteligencia artificial puede cometer errores",
        "una respuesta convincente también puede estar equivocada",
        "el modelo no consulta internet cada vez que genera texto",
        "el entrenamiento ajusta parámetros usando muchos ejemplos",
        "los embeddings representan palabras usando vectores",
        "palabras parecidas pueden terminar cerca en el espacio vectorial",
        "la temperatura cambia qué tan aventurada es una generación",
        "con temperatura baja el modelo suele elegir opciones probables",
        "con temperatura alta aparecen opciones menos probables",
        "el muestreo permite generar respuestas diferentes",
        "el modo determinista elige la palabra con mayor probabilidad",
        "evidencia mata todo cuando queremos verificar una afirmación",
        "un buen ingeniero mide antes de confiar",
        "usar inteligencia artificial no elimina la responsabilidad humana",
        "un prompt claro ayuda pero no reemplaza la ingeniería",
        "más texto en un prompt no siempre produce una mejor respuesta",
    ]

    subjects = [
        "el modelo",
        "la red neuronal",
        "el sistema",
        "la inteligencia artificial",
    ]
    verbs = [
        "aprende",
        "predice",
        "observa",
        "procesa",
        "combina",
    ]
    objects = [
        "patrones del lenguaje",
        "el contexto anterior",
        "relaciones entre palabras",
        "señales estadísticas",
        "ejemplos de entrenamiento",
    ]

    generated = [" ".join(parts) for parts in product(subjects, verbs, objects)]

    starts = ["yo quiero", "hoy quiero", "mañana quiero", "ahora quiero"]
    actions = [
        "aprender inteligencia artificial",
        "entender redes neuronales",
        "programar un modelo pequeño",
        "explicar probabilidades",
        "probar una idea nueva",
    ]
    generated.extend(f"{start} {action}" for start, action in product(starts, actions))

    # Repetition is intentional: it makes the tiny network learn visibly in seconds.
    return (fixed * 8) + (generated * 5)
