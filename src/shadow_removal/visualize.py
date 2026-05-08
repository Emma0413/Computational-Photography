from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .io import ensure_dir, write_rgb


def label_image(rgb: np.ndarray, label: str) -> np.ndarray:
    canvas = rgb.copy()
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 260), 30), (0, 0, 0), -1)
    cv2.putText(canvas, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return np.repeat((mask.astype(np.uint8) * 255)[:, :, None], 3, axis=2)


def contact_sheet(path: str | Path, images: list[tuple[str, np.ndarray]], thumb_width: int = 280) -> None:
    thumbs = []
    for label, image in images:
        scale = thumb_width / image.shape[1]
        thumb = cv2.resize(image, (thumb_width, max(1, int(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        thumbs.append(label_image(thumb, label))
    max_h = max(t.shape[0] for t in thumbs)
    padded = []
    for thumb in thumbs:
        if thumb.shape[0] < max_h:
            pad = np.full((max_h - thumb.shape[0], thumb.shape[1], 3), 255, dtype=np.uint8)
            thumb = np.vstack([thumb, pad])
        padded.append(thumb)
    sheet = np.hstack(padded)
    ensure_dir(Path(path).parent)
    write_rgb(path, sheet)
