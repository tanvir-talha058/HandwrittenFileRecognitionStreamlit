from __future__ import annotations

import cv2
import numpy as np


def deskew_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))

    if len(coords) == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_image(image: np.ndarray, max_width: int = 1800) -> np.ndarray:
    processed = image.copy()

    height, width = processed.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        processed = cv2.resize(processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    processed = deskew_image(processed)

    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)
    normalized = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
