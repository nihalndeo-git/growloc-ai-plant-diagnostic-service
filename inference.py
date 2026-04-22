"""Inference pipeline using canopy, fruit, and leaf YOLO models."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO

_models_loaded = False
_canopy_model: YOLO | None = None
_fruit_model: YOLO | None = None
_leaf_model: YOLO | None = None


def _default_models_dir() -> Path:
    # /fullstack_app/growloc/ai-service/inference.py -> /fullstack_app
    return Path(__file__).resolve().parents[2] / "modelsToBeIntegrated"


def _candidate_model_dirs() -> list[Path]:
    here = Path(__file__).resolve()
    workspace_root = here.parents[2]
    app_root = here.parents[1]
    candidates = [
        Path(os.getenv("MODELS_DIR", "")) if os.getenv("MODELS_DIR") else None,
        workspace_root / "modelsToBeIntegrated",
        app_root / "modelsToBeIntegrated",
    ]
    return [c for c in candidates if c is not None]


def _resolve_model_path(model_name: str) -> Path:
    # Keep the previous default behavior for compatibility.
    models_dir = Path(os.getenv("MODELS_DIR", str(_default_models_dir())))
    return models_dir / model_name


def load_models() -> None:
    """Load all 3 YOLO checkpoints once at startup."""
    global _models_loaded, _canopy_model, _fruit_model, _leaf_model
    if _models_loaded:
        return

    model_names = ("canopy_model.pt", "fruit_model.pt", "leaf_model.pt")
    selected_paths: tuple[Path, Path, Path] | None = None
    attempted: list[str] = []
    for d in _candidate_model_dirs():
        attempted.append(str(d))
        paths = tuple(d / name for name in model_names)
        if all(p.exists() for p in paths):
            selected_paths = paths  # type: ignore[assignment]
            break

    if selected_paths is None:
        raise FileNotFoundError(
            "Could not find required model files "
            f"{list(model_names)} in any candidate directory: {attempted}. "
            "Set MODELS_DIR to the folder containing these files."
        )

    canopy_path, fruit_path, leaf_path = selected_paths

    _canopy_model = YOLO(str(canopy_path))
    _fruit_model = YOLO(str(fruit_path))
    _leaf_model = YOLO(str(leaf_path))
    _models_loaded = True


def _decode_image(image_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as img:
        return np.array(img.convert("RGB"))


def _run_model(model: YOLO | None, image: np.ndarray) -> Any:
    if model is None:
        return None
    results = model.predict(source=image, verbose=False)
    return results[0] if results else None


def _extract_canopy_hw_ratio(result: Any, image_shape: tuple[int, int, int]) -> tuple[float, float]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return 0.0, 0.0

    xyxy = boxes.xyxy.cpu().numpy()
    widths = xyxy[:, 2] - xyxy[:, 0]
    heights = xyxy[:, 3] - xyxy[:, 1]
    max_w = float(np.max(widths)) if len(widths) else 0.0
    max_h = float(np.max(heights)) if len(heights) else 0.0
    img_h, img_w = image_shape[0], image_shape[1]
    return (max_h / img_h if img_h else 0.0, max_w / img_w if img_w else 0.0)


def _extract_leaf_area_ratio(result: Any, image_shape: tuple[int, int, int]) -> float:
    masks = getattr(result, "masks", None)
    data = getattr(masks, "data", None) if masks is not None else None
    if data is None or len(data) == 0:
        return 0.0
    mask_stack = data.cpu().numpy()
    union_mask = np.any(mask_stack > 0.5, axis=0)
    area_pixels = float(np.count_nonzero(union_mask))
    image_area = float(image_shape[0] * image_shape[1]) if image_shape[0] and image_shape[1] else 0.0
    return area_pixels / image_area if image_area else 0.0


def run_inference(image_bytes: bytes) -> dict[str, float]:
    """Run all three models and return canopy-focused metrics."""
    if not _models_loaded:
        load_models()

    image = _decode_image(image_bytes)

    canopy_result = _run_model(_canopy_model, image)
    _ = _run_model(_fruit_model, image)
    leaf_result = _run_model(_leaf_model, image)

    canopy_height, canopy_width = _extract_canopy_hw_ratio(canopy_result, image.shape)
    leaf_area = _extract_leaf_area_ratio(leaf_result, image.shape)
    canopy_area = leaf_area if leaf_area > 0 else canopy_height * canopy_width

    return {
        "canopy_height": round(float(canopy_height), 4),
        "canopy_width": round(float(canopy_width), 4),
        "canopy_area": round(float(canopy_area), 4),
    }
