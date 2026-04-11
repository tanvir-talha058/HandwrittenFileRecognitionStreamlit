from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz
import numpy as np

from ocr.engine import OCREngine


FIELD_LABELS: dict[str, list[str]] = {
    "applicant_name": ["applicant name", "full name", "name"],
    "date_of_birth": ["date of birth", "dob", "birth date"],
    "nid_number": ["nid", "national id", "national id no"],
    "father_name": ["father name", "father's name", "father"],
    "mother_name": ["mother name", "mother's name", "mother"],
    "present_address": ["present address", "current address"],
    "permanent_address": ["permanent address"],
    "mobile_number": ["mobile number", "mobile", "phone", "contact"],
    "occupation": ["occupation", "profession"],
    "monthly_income": ["monthly income", "income", "salary"],
    "loan_amount": ["loan amount", "amount requested", "requested amount"],
    "loan_purpose": ["loan purpose", "purpose of loan", "purpose"],
    "property_address": ["property address", "property location"],
    "land_area": ["land area", "plot area", "sqft", "area"],
    "co_applicant_name": ["co-applicant", "co applicant"],
    "guarantor_name": ["guarantor", "guarantor name"],
    "repayment_period": ["repayment period", "tenure", "term"],
    "signature_date": ["signature date", "date of signature"],
}
TEMPLATE_MIN_CONFIDENCE = 0.35
TEMPLATE_RENDER_SCALE = 1.25


class PDFFiller:
    def __init__(self, template: str | None = None) -> None:
        self.template = template

    def fill(self, form_data: dict[str, str], output_path: str) -> str:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if self.template and Path(self.template).exists():
            doc = fitz.open(self.template)
        else:
            doc = fitz.open()
            doc.new_page(width=595, height=842)

        filled_widget = False
        for page in doc:
            widgets = page.widgets()
            if not widgets:
                continue
            for widget in widgets:
                name = (widget.field_name or "").strip()
                matched_key = self._match_field_key(name, form_data)
                if matched_key:
                    widget.field_value = str(form_data[matched_key])
                    widget.update()
                    filled_widget = True

        if not filled_widget:
            pending = {key: str(value) for key, value in form_data.items() if str(value).strip()}
            self._write_values_next_to_searchable_labels(doc, pending)
            if pending and self.template and Path(self.template).exists():
                self._write_values_using_template_ocr(doc, pending)
            if pending:
                self._append_summary_page(doc, pending)

        doc.save(str(output))
        doc.close()
        return str(output)

    @staticmethod
    def _normalize_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

    def _match_field_key(self, raw_name: str, form_data: dict[str, str]) -> str | None:
        normalized_name = self._normalize_key(raw_name)
        if normalized_name in form_data:
            return normalized_name
        if raw_name in form_data:
            return raw_name

        best_key = None
        for key in form_data:
            normalized_key = self._normalize_key(key)
            if normalized_key == normalized_name:
                return key
            if normalized_key in normalized_name or normalized_name in normalized_key:
                best_key = key
        return best_key

    @staticmethod
    def _search_keyword_rect(page: fitz.Page, keyword: str) -> fitz.Rect | None:
        for candidate in {keyword, keyword.title(), keyword.upper()}:
            rects = page.search_for(candidate)
            if rects:
                return rects[0]
        return None

    def _write_values_next_to_searchable_labels(self, doc: fitz.Document, pending: dict[str, str]) -> None:
        for page in doc:
            for key in list(pending.keys()):
                labels = FIELD_LABELS.get(key, [key.replace("_", " ")])
                anchor = None
                for label in labels:
                    anchor = self._search_keyword_rect(page, label)
                    if anchor is not None:
                        break

                if anchor is None:
                    continue

                self._write_value_near_anchor(page, anchor, pending[key])
                del pending[key]

        if not pending:
            return

    def _write_values_using_template_ocr(self, doc: fitz.Document, pending: dict[str, str]) -> None:
        engine = OCREngine(lang="en", use_gpu=False, use_angle_cls=False)
        for page_number, page in enumerate(doc, start=1):
            page_lines, rendered_size = self._extract_template_page_lines(engine, page, page_number)
            rendered_width, rendered_height = rendered_size
            if not page_lines or rendered_width <= 0 or rendered_height <= 0:
                continue

            scale_x = float(page.rect.width) / float(rendered_width)
            scale_y = float(page.rect.height) / float(rendered_height)

            for key in list(pending.keys()):
                anchor_line = self._find_ocr_anchor_line(page_lines, key)
                if anchor_line is None:
                    continue

                anchor_rect = fitz.Rect(
                    anchor_line["x1"] * scale_x,
                    anchor_line["y1"] * scale_y,
                    anchor_line["x2"] * scale_x,
                    anchor_line["y2"] * scale_y,
                )
                self._write_value_near_anchor(page, anchor_rect, pending[key])
                del pending[key]

            if not pending:
                break

    def _append_summary_page(self, doc: fitz.Document, pending: dict[str, str]) -> None:
        summary_page = doc.new_page(width=595, height=842)
        y = 72
        summary_page.insert_text((72, 48), "Unplaced Extracted Fields", fontsize=14)
        for key in sorted(pending.keys()):
            summary_page.insert_text((72, y), f"{key}: {pending[key]}", fontsize=10)
            y += 16
            if y > 790:
                summary_page = doc.new_page(width=595, height=842)
                y = 72

    def _write_value_near_anchor(self, page: fitz.Page, anchor: fitz.Rect, value: str) -> None:
        if not value.strip():
            return

        target = self._build_value_rect(page.rect, anchor)
        fontsize = 10 if len(value) <= 36 else 9
        remaining = page.insert_textbox(
            target,
            value,
            fontsize=fontsize,
            color=(0, 0, 0),
            align=fitz.TEXT_ALIGN_LEFT,
            overlay=True,
        )
        if remaining < 0:
            fallback = fitz.Point(target.x0, min(target.y1 - 4, page.rect.height - 24))
            page.insert_text(fallback, value, fontsize=fontsize, color=(0, 0, 0), overlay=True)

    @staticmethod
    def _build_value_rect(page_rect: fitz.Rect, anchor: fitz.Rect) -> fitz.Rect:
        right_start = min(anchor.x1 + 10, page_rect.width - 120)
        right_rect = fitz.Rect(
            right_start,
            max(anchor.y0 - 2, 18),
            page_rect.width - 28,
            min(anchor.y1 + 18, page_rect.height - 18),
        )
        if right_rect.width >= 120:
            return right_rect

        below_top = min(anchor.y1 + 4, page_rect.height - 40)
        return fitz.Rect(
            max(anchor.x0, 24),
            below_top,
            page_rect.width - 28,
            min(below_top + 32, page_rect.height - 18),
        )

    def _find_ocr_anchor_line(self, page_lines: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
        labels = FIELD_LABELS.get(key, [key.replace("_", " ")])
        best_line: dict[str, Any] | None = None
        best_score = 0.0

        for line in page_lines:
            normalized_line = self._normalize_for_match(str(line.get("text", "")))
            if not normalized_line:
                continue

            for label in labels:
                score = self._match_label_score(normalized_line, label)
                if score > best_score:
                    best_score = score
                    best_line = line

        if best_score >= 0.72:
            return best_line
        return None

    @staticmethod
    def _normalize_for_match(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def _match_label_score(self, normalized_line: str, label: str) -> float:
        normalized_label = self._normalize_for_match(label)
        if not normalized_label:
            return 0.0
        if normalized_label in normalized_line:
            return 1.0

        label_tokens = normalized_label.split()
        if label_tokens and all(token in normalized_line for token in label_tokens):
            return 0.9

        return self._token_overlap_score(normalized_line, normalized_label)

    @staticmethod
    def _token_overlap_score(normalized_line: str, normalized_label: str) -> float:
        line_tokens = set(normalized_line.split())
        label_tokens = set(normalized_label.split())
        if not line_tokens or not label_tokens:
            return 0.0
        overlap = len(line_tokens & label_tokens)
        return overlap / max(len(label_tokens), 1)

    @staticmethod
    def _extract_template_page_lines(
        engine: OCREngine,
        page: fitz.Page,
        page_number: int,
    ) -> tuple[list[dict[str, Any]], tuple[int, int]]:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(TEMPLATE_RENDER_SCALE, TEMPLATE_RENDER_SCALE), alpha=False)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
        if pixmap.n >= 3:
            image = image[:, :, :3][:, :, ::-1]

        try:
            blocks = engine.run_image_array(
                image,
                min_confidence=TEMPLATE_MIN_CONFIDENCE,
                preprocess=False,
                page_number=page_number,
            )
        except Exception:
            return [], (pixmap.width, pixmap.height)

        return PDFFiller._group_template_lines(PDFFiller._normalize_template_blocks(blocks)), (pixmap.width, pixmap.height)

    @staticmethod
    def _normalize_template_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for block in blocks:
            text = str(block.get("text", "")).strip()
            if not text:
                continue

            bbox = block.get("bbox", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            points: list[list[float]] = []
            for point in bbox:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    points = []
                    break
                try:
                    x_value = float(point[0])
                    y_value = float(point[1])
                except (TypeError, ValueError):
                    points = []
                    break
                points.append([x_value, max(0.0, y_value)])
            if len(points) != 4:
                continue

            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            normalized.append(
                {
                    "page": int(block.get("page", 1) or 1),
                    "text": text,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": (x1 + x2) / 2.0,
                    "cy": (y1 + y2) / 2.0,
                    "height": max(1.0, y2 - y1),
                }
            )
        return normalized

    @staticmethod
    def _group_template_lines(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered_blocks = sorted(blocks, key=lambda item: (item["cy"], item["x1"]))
        lines: list[dict[str, Any]] = []

        for block in ordered_blocks:
            if not lines:
                lines.append(PDFFiller._new_line(block))
                continue

            last_line = lines[-1]
            tolerance = max(last_line["avg_height"] * 0.6, block["height"] * 0.6, 14.0)
            if abs(block["cy"] - last_line["cy"]) <= tolerance:
                PDFFiller._merge_block_into_line(last_line, block)
                continue

            lines.append(PDFFiller._new_line(block))

        for line in lines:
            ordered_line_blocks = sorted(line["blocks"], key=lambda item: item["x1"])
            line["text"] = " ".join(str(item["text"]).strip() for item in ordered_line_blocks if str(item["text"]).strip())
        return lines

    @staticmethod
    def _new_line(block: dict[str, Any]) -> dict[str, Any]:
        return {
            "blocks": [block],
            "text": str(block["text"]),
            "x1": block["x1"],
            "y1": block["y1"],
            "x2": block["x2"],
            "y2": block["y2"],
            "cy": block["cy"],
            "avg_height": block["height"],
        }

    @staticmethod
    def _merge_block_into_line(line: dict[str, Any], block: dict[str, Any]) -> None:
        line["blocks"].append(block)
        line["x1"] = min(line["x1"], block["x1"])
        line["y1"] = min(line["y1"], block["y1"])
        line["x2"] = max(line["x2"], block["x2"])
        line["y2"] = max(line["y2"], block["y2"])
        line["cy"] = (line["y1"] + line["y2"]) / 2.0
        line["avg_height"] = sum(float(item["height"]) for item in line["blocks"]) / len(line["blocks"])
