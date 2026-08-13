from __future__ import annotations

import asyncio
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .backends import ModelManager
from .schemas import GenerateRequest, GenerationState, ModelInfo, ModelLoadRequest, StepRequest


STATIC_DIR = Path(__file__).parent / "static"
manager = ModelManager()
app = FastAPI(title="Más allá del prompt · Mini ChatGPT", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status() -> dict[str, object]:
    return {
        "ready": manager.ready,
        "error": manager.error,
        "model": manager.backend.info.model_dump() if manager.backend else None,
    }


@app.post("/api/load", response_model=ModelInfo)
async def load_model(request: ModelLoadRequest) -> ModelInfo:
    try:
        return await asyncio.to_thread(manager.load, request.model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo cargar el modelo: {exc}") from exc


@app.post("/api/step", response_model=GenerationState)
async def next_token(request: StepRequest) -> GenerationState:
    if not manager.backend:
        raise HTTPException(status_code=409, detail="Primero carga un modelo.")
    try:
        return await asyncio.to_thread(
            manager.backend.step,
            request.text,
            request.temperature,
            request.mode,
            request.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo calcular el siguiente token: {exc}") from exc


@app.post("/api/generate", response_model=list[GenerationState])
async def generate_tokens(request: GenerateRequest) -> list[GenerationState]:
    if not manager.backend:
        raise HTTPException(status_code=409, detail="Primero carga un modelo.")
    try:
        return await asyncio.to_thread(
            manager.backend.generate,
            request.text,
            request.temperature,
            request.mode,
            request.top_k,
            request.max_tokens,
            request.stop_strings,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo generar la respuesta: {exc}") from exc



def _generation_events(request: GenerateRequest):
    backend = manager.backend
    if backend is None:
        return
    try:
        for state in backend.iter_generate(
            request.text,
            request.temperature,
            request.mode,
            request.top_k,
            request.max_tokens,
            request.stop_strings,
        ):
            yield f'{{"type":"state","state":{state.model_dump_json()}}}\n'
    except Exception as exc:
        detail = json.dumps(f"{type(exc).__name__}: {exc}", ensure_ascii=False)
        yield f'{{"type":"error","detail":{detail}}}\n'


@app.post("/api/generate-stream")
def generate_tokens_stream(request: GenerateRequest) -> StreamingResponse:
    if not manager.backend:
        raise HTTPException(status_code=409, detail="Primero carga un modelo.")
    return StreamingResponse(
        _generation_events(request),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def main() -> None:
    print("Mini ChatGPT listo en http://127.0.0.1:8080")
    uvicorn.run(
        "demo_mini_chat.web:app",
        host="127.0.0.1",
        port=8080,
        reload=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
