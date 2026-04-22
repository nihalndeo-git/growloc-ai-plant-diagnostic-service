from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference import load_models, run_inference


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        load_models()
    except FileNotFoundError as exc:
        # Allow startup without models so the API is reachable for debugging.
        print(f"[growloc-ai] model preload skipped: {exc}")
    yield


app = FastAPI(title="Growloc AI Service", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict[str, float]:
    data = await file.read()
    try:
        metrics = run_inference(data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "canopy_height": metrics["canopy_height"],
        "canopy_width": metrics["canopy_width"],
        "canopy_area": metrics["canopy_area"],
    }
