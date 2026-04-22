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


def _extract_canopy_hw_px(result: Any) -> tuple[float, float]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return 0.0, 0.0

    xyxy = boxes.xyxy.cpu().numpy()
    widths = xyxy[:, 2] - xyxy[:, 0]
    heights = xyxy[:, 3] - xyxy[:, 1]
    max_w = float(np.max(widths)) if len(widths) else 0.0
    max_h = float(np.max(heights)) if len(heights) else 0.0
    return max_h, max_w


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


def _classify_color_name(rgb_crop: np.ndarray) -> str:
    if rgb_crop.size == 0:
        return "unknown"
    mean_rgb = rgb_crop.reshape(-1, 3).mean(axis=0)
    r, g, b = float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])

    max_c = max(r, g, b)
    min_c = min(r, g, b)
    delta = max_c - min_c
    if max_c == 0:
        return "unknown"
    saturation = delta / max_c

    if saturation < 0.16:
        if max_c < 50:
            return "black"
        if max_c > 200:
            return "white"
        return "gray"

    if max_c == r:
        hue = (60 * ((g - b) / delta) + 360) % 360 if delta else 0
    elif max_c == g:
        hue = (60 * ((b - r) / delta) + 120) if delta else 0
    else:
        hue = (60 * ((r - g) / delta) + 240) if delta else 0

    if hue < 15 or hue >= 345:
        return "red"
    if hue < 45:
        return "orange"
    if hue < 70:
        return "yellow"
    if hue < 170:
        return "green"
    if hue < 255:
        return "blue"
    if hue < 320:
        return "purple"
    return "pink"


def _extract_detections(result: Any, image: np.ndarray) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    names = getattr(result, "names", {}) or {}
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else None
    cls = boxes.cls.cpu().numpy() if getattr(boxes, "cls", None) is not None else None

    detections: list[dict[str, Any]] = []
    for idx, box in enumerate(xyxy):
        class_id = int(cls[idx]) if cls is not None and idx < len(cls) else -1
        score = float(conf[idx]) if conf is not None and idx < len(conf) else 0.0
        label = str(names.get(class_id, class_id))
        x1 = max(0, int(round(float(box[0]))))
        y1 = max(0, int(round(float(box[1]))))
        x2 = min(image.shape[1], int(round(float(box[2]))))
        y2 = min(image.shape[0], int(round(float(box[3]))))
        color = _classify_color_name(image[y1:y2, x1:x2])
        detections.append(
            {
                "label": label,
                "color": color,
                "confidence": round(score, 4),
                "bbox": {
                    "x1": round(float(box[0]), 2),
                    "y1": round(float(box[1]), 2),
                    "x2": round(float(box[2]), 2),
                    "y2": round(float(box[3]), 2),
                },
            }
        )
    return detections


def _count_by_key(detections: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in detections:
        value = str(item.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def run_inference(image_bytes: bytes) -> dict[str, Any]:
    """Run all three models and return combined multi-model outputs."""
    if not _models_loaded:
        load_models()

    image = _decode_image(image_bytes)

    canopy_result = _run_model(_canopy_model, image)
    fruit_result = _run_model(_fruit_model, image)
    leaf_result = _run_model(_leaf_model, image)

    canopy_height_px, canopy_width_px = _extract_canopy_hw_px(canopy_result)
    pixel_to_cm = float(os.getenv("CANOPY_PIXEL_TO_CM", "1.0"))
    canopy_height_cm = canopy_height_px * pixel_to_cm
    canopy_width_cm = canopy_width_px * pixel_to_cm
    canopy_area_cm2 = canopy_height_cm * canopy_width_cm
    leaf_area = _extract_leaf_area_ratio(leaf_result, image.shape)
    canopy_detections = _extract_detections(canopy_result, image)
    fruit_detections = _extract_detections(fruit_result, image)
    leaf_detections = _extract_detections(leaf_result, image)
    fruit_counts = _count_by_key(fruit_detections, "label")
    fruit_color_counts = _count_by_key(fruit_detections, "color")
    leaf_counts = _count_by_key(leaf_detections, "label")
    leaf_color_counts = _count_by_key(leaf_detections, "color")
    leaf_detection_count = len(leaf_detections)

    return {
        # Backward-compatible fields now represent centimeters.
        "canopy_height": round(float(canopy_height_cm), 2),
        "canopy_width": round(float(canopy_width_cm), 2),
        "canopy_area": round(float(canopy_area_cm2), 4),
        "canopy_height_px": round(float(canopy_height_px), 2),
        "canopy_width_px": round(float(canopy_width_px), 2),
        "canopy_height_cm": round(float(canopy_height_cm), 2),
        "canopy_width_cm": round(float(canopy_width_cm), 2),
        "canopy_area_cm2": round(float(canopy_area_cm2), 2),
        "canopy_pixel_to_cm": pixel_to_cm,
        "canopy_calibrated": pixel_to_cm != 1.0,
        "canopy_detections": canopy_detections,
        "fruit_detections": fruit_detections,
        "leaf_detections": leaf_detections,
        "fruit_counts": fruit_counts,
        "fruit_color_counts": fruit_color_counts,
        "leaf_counts": leaf_counts,
        "leaf_color_counts": leaf_color_counts,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "leaf": {
            "mask_area_ratio": round(float(leaf_area), 4),
            "detection_count": leaf_detection_count,
        },
    }
