from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np

from .preprocessor import preprocess_image


class OCREngine:
    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = False,
        use_angle_cls: bool = True,
        det_db_thresh: float = 0.3,
    ) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls
        self.det_db_thresh = det_db_thresh
        self.last_error: str | None = None
        self._ocr = self._build_ocr()

    def _build_ocr(self) -> Any:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install dependencies with: pip install -r requirements.txt"
            ) from exc

        try:
            return PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                use_gpu=self.use_gpu,
                det_db_thresh=self.det_db_thresh,
            )
        except ValueError as exc:
            # PaddleOCR >= 3.x removed several legacy constructor args.
            if "Unknown argument" not in str(exc):
                raise

            device = "gpu" if self.use_gpu else "cpu"
            init_kwargs: dict[str, Any] = {
                "lang": self.lang,
                "device": device,
                "text_det_thresh": self.det_db_thresh,
            }
            if device == "cpu":
                # PaddleOCR 3.x enables MKL-DNN on CPU by default, which crashes in
                # some Paddle runtime combinations with oneDNN executor errors.
                init_kwargs["enable_mkldnn"] = False
            return PaddleOCR(**init_kwargs)

    def _load_image(self, input_path: str) -> np.ndarray:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._pdf_page_to_image(path)

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unsupported image or unreadable file: {input_path}")
        return image

    @staticmethod
    def _pdf_page_to_image(path: Path) -> np.ndarray:
        document = fitz.open(path)
        if len(document) == 0:
            raise ValueError("PDF has no pages.")

        page = document[0]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
        document.close()

        if pixmap.n == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def run(self, input_path: str, min_confidence: float = 0.8, preprocess: bool = False) -> list[dict[str, Any]]:
        self.last_error = None
        image = self._load_image(input_path)
        if preprocess:
            image = preprocess_image(image)

        try:
            result = self._predict(image)
        except TypeError as exc:
            # PaddleOCR >= 3.x no longer accepts the cls keyword here.
            if "unexpected keyword argument 'cls'" not in str(exc):
                raise
            try:
                result = self._ocr.ocr(image)
            except Exception as nested_exc:
                self.last_error = f"{type(nested_exc).__name__}: {nested_exc}"
                return []
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []

        parsed = self._parse_result(result)
        if not parsed:
            return []

        blocks: list[dict[str, Any]] = []
        for item in parsed:
            bbox, payload = item
            text, confidence = payload
            if confidence < min_confidence:
                continue
            blocks.append(
                {
                    "text": text.strip(),
                    "confidence": float(confidence),
                    "bbox": [[int(p[0]), int(p[1])] for p in bbox],
                }
            )
        return blocks

    def _predict(self, image: np.ndarray) -> Any:
        predict = getattr(self._ocr, "predict", None)
        if callable(predict):
            return predict(image)
        return self._ocr.ocr(image, cls=self.use_angle_cls)

    @staticmethod
    def _parse_result(result: Any) -> list[tuple[Any, Any]]:
        # PaddleOCR 2.x format: [ [ [bbox, [text, conf]], ... ] ]
        if isinstance(result, list) and result and isinstance(result[0], list):
            first = result[0]
            if first and isinstance(first[0], (list, tuple)) and len(first[0]) == 2:
                return first

        # PaddleOCR 3.x format may return dict-like items.
        if isinstance(result, list) and result and isinstance(result[0], dict):
            normalized: list[tuple[Any, Any]] = []
            for item in result:
                texts = item.get("rec_texts")
                scores = item.get("rec_scores")
                polys = OCREngine._first_present(item, "dt_polys", "rec_polys", "bbox")
                if isinstance(polys, np.ndarray):
                    polys = polys.tolist()

                if isinstance(texts, list):
                    for idx, text in enumerate(texts):
                        if not isinstance(text, str):
                            continue
                        poly = polys[idx] if isinstance(polys, list) and idx < len(polys) else []
                        poly = OCREngine._normalize_poly(poly)
                        score = scores[idx] if isinstance(scores, list) and idx < len(scores) else 0.0
                        normalized.append((poly, [text, float(score)]))
                    continue

                text = item.get("rec_text") or item.get("text") or ""
                score = item.get("rec_score") or item.get("score") or 0.0
                poly = OCREngine._normalize_poly(polys)
                if isinstance(text, str):
                    normalized.append((poly, [text, float(score)]))
            return normalized

        return []

    @staticmethod
    def _first_present(item: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = item.get(key)
            if value is not None:
                return value
        return []

    @staticmethod
    def _normalize_poly(poly: Any) -> Any:
        if isinstance(poly, np.ndarray):
            return poly.tolist()
        return poly

    def save_debug_image(
        self,
        input_path: str,
        blocks: list[dict[str, Any]],
        output_path: str,
        preprocess: bool = False,
    ) -> str:
        image = self._load_image(input_path)
        if preprocess:
            image = preprocess_image(image)

        for block in blocks:
            pts = np.array(block.get("bbox", []), dtype=np.int32)
            if len(pts) == 4:
                cv2.polylines(image, [pts], True, (0, 255, 0), 2)
                cv2.putText(
                    image,
                    block.get("text", "")[:24],
                    tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), image)
        return str(output)
