from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml

from .validators import (
    normalize_date,
    normalize_nid,
    normalize_numeric,
    normalize_phone,
    normalize_whitespace,
)


DEFAULT_FIELDS = {
    "applicant_name": ["applicant name", "applicant's name", "name of applicant", "full name", "name"],
    "date_of_birth": ["date of birth", "dob", "birth date", "birth"],
    "nid_number": ["nid", "national id", "national id no", "national identification", "nid number"],
    "father_name": ["father's name", "father name", "name of father", "father"],
    "mother_name": ["mother's name", "mother name", "name of mother", "mother"],
    "present_address": ["present address", "current address", "mailing address"],
    "permanent_address": ["permanent address", "parmanent address"],
    "mobile_number": ["mobile", "mobile number", "phone", "contact", "telephone"],
    "occupation": ["occupation", "profession", "job"],
    "monthly_income": ["monthly income", "income", "salary"],
    "loan_amount": ["loan amount", "amount requested", "requested amount", "amount"],
    "loan_purpose": ["loan purpose", "purpose", "purpose of loan"],
    "property_address": ["property address", "property location", "address of property"],
    "land_area": ["land area", "sqft", "area", "plot area"],
    "co_applicant_name": ["co-applicant", "co applicant", "co applicant name"],
    "guarantor_name": ["guarantor", "guarantor name"],
    "repayment_period": ["repayment period", "tenure", "term", "repayment term"],
    "signature_date": ["signature date", "date of signature", "date"],
}

MULTILINE_FIELDS = {"present_address", "permanent_address", "property_address", "loan_purpose"}


class FieldMapper:
    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = self._load_config(config_path)

        mapping_cfg = self.config.get("field_mapping", {})
        field_cfg = mapping_cfg.get("fields", {})

        self.date_formats = mapping_cfg.get("date_formats", ["%d/%m/%Y", "%d-%m-%Y"])
        self.conf_threshold = float(mapping_cfg.get("confidence_threshold", 0.6))
        self.fuzzy_match = bool(mapping_cfg.get("fuzzy_match", True))

        self.field_keywords: dict[str, list[str]] = {}
        for field_key, defaults in DEFAULT_FIELDS.items():
            cfg_keywords = field_cfg.get(field_key, {}).get("keywords")
            self.field_keywords[field_key] = cfg_keywords or defaults

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            return {}

        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data if isinstance(data, dict) else {}

    def map(self, ocr_results: list[dict[str, Any]]) -> dict[str, str]:
        extracted, _, _ = self.map_with_context(ocr_results)
        return extracted

    def map_with_context(self, ocr_results: list[dict[str, Any]]) -> tuple[dict[str, str], list[str], list[dict[str, Any]]]:
        blocks = self._normalize_blocks(ocr_results)
        lines = self._group_blocks_into_lines(blocks)
        transcript_lines = [line["text"] for line in lines if line["text"]]
        extracted = self._extract_fields(lines, transcript_lines)
        return extracted, transcript_lines, lines

    def _normalize_blocks(self, ocr_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for block in ocr_results:
            text = normalize_whitespace(str(block.get("text", "")))
            if not text:
                continue

            try:
                confidence = float(block.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < self.conf_threshold:
                continue

            bbox = block.get("bbox", [])
            points = FieldMapper._normalize_bbox(bbox)
            if len(points) != 4:
                continue

            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            normalized.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "bbox": points,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": (x1 + x2) / 2.0,
                    "cy": (y1 + y2) / 2.0,
                    "height": max(1, y2 - y1),
                    "width": max(1, x2 - x1),
                }
            )
        return normalized

    @staticmethod
    def _normalize_bbox(bbox: Any) -> list[list[int]]:
        if not isinstance(bbox, list):
            return []

        points: list[list[int]] = []
        for point in bbox:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    points.append([int(round(float(point[0]))), int(round(float(point[1])))])
                except (TypeError, ValueError):
                    return []
        return points

    def _group_blocks_into_lines(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not blocks:
            return []

        ordered_blocks = sorted(blocks, key=lambda item: (item["cy"], item["x1"]))
        lines: list[dict[str, Any]] = []

        for block in ordered_blocks:
            if not lines:
                lines.append(self._new_line(block))
                continue

            last_line = lines[-1]
            tolerance = max(int(last_line["avg_height"] * 0.6), int(block["height"] * 0.6), 14)
            if abs(block["cy"] - last_line["cy"]) <= tolerance:
                self._merge_block_into_line(last_line, block)
                continue

            lines.append(self._new_line(block))

        for line in lines:
            line["text"] = normalize_whitespace(" ".join(item["text"] for item in sorted(line["blocks"], key=lambda entry: entry["x1"])))

        return lines

    @staticmethod
    def _new_line(block: dict[str, Any]) -> dict[str, Any]:
        return {
            "blocks": [block],
            "text": block["text"],
            "x1": block["x1"],
            "y1": block["y1"],
            "x2": block["x2"],
            "y2": block["y2"],
            "cx": block["cx"],
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
        line["cx"] = (line["x1"] + line["x2"]) / 2.0
        line["cy"] = (line["y1"] + line["y2"]) / 2.0
        line["avg_height"] = sum(item["height"] for item in line["blocks"]) / len(line["blocks"])

    def _extract_fields(self, lines: list[dict[str, Any]], transcript_lines: list[str]) -> dict[str, str]:
        extracted: dict[str, str] = {}
        lower_transcript = "\n".join(transcript_lines).lower()

        for index, line in enumerate(lines):
            line_text = line["text"]
            if not line_text:
                continue

            matched_field, matched_keyword = self._identify_field(line_text)
            if not matched_field or matched_field in extracted:
                continue

            value = self._extract_value_from_line(lines, index, matched_keyword, matched_field)
            if value:
                extracted[matched_field] = value

        fallback_patterns = {
            "date_of_birth": r"(?:dob|date\s*of\s*birth|birth\s*date)[:\s-]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
            "mobile_number": r"(?:mobile|phone|contact|telephone)[:\s-]*([+0-9\-\s]{8,20})",
            "nid_number": r"(?:nid|national\s*id|national\s*id\s*no|national\s*identification)[:\s-]*([0-9\-\s]{8,25})",
            "monthly_income": r"(?:monthly\s*income|income|salary)[:\s-]*([0-9,\.\s]+)",
            "loan_amount": r"(?:loan\s*amount|amount\s*requested|requested\s*amount|amount)[:\s-]*([0-9,\.\s]+)",
            "repayment_period": r"(?:repayment\s*period|tenure|term)[:\s-]*([A-Za-z0-9\s]+)",
        }

        for field, pattern in fallback_patterns.items():
            if field in extracted:
                continue
            match = re.search(pattern, lower_transcript, flags=re.IGNORECASE)
            if match:
                extracted[field] = normalize_whitespace(match.group(1))

        return self._post_process(extracted)

    def _identify_field(self, line_text: str) -> tuple[str | None, str | None]:
        normalized_line = self._normalize_for_match(line_text)
        if not normalized_line:
            return None, None

        best_field: str | None = None
        best_keyword: str | None = None
        best_score = 0.0

        for field, keywords in self.field_keywords.items():
            for keyword in keywords:
                score = self._match_score(normalized_line, keyword)
                if score > best_score:
                    best_score = score
                    best_field = field
                    best_keyword = keyword

        if best_field is None:
            return None, None

        if best_score >= 0.92:
            return best_field, best_keyword
        if self.fuzzy_match and best_score >= 0.78:
            return best_field, best_keyword
        return None, None

    @staticmethod
    def _normalize_for_match(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def _match_score(self, normalized_line: str, keyword: str) -> float:
        normalized_keyword = self._normalize_for_match(keyword)
        if not normalized_keyword:
            return 0.0

        if normalized_keyword in normalized_line:
            return 1.0

        return SequenceMatcher(None, normalized_keyword, normalized_line).ratio()

    def _extract_value_from_line(self, lines: list[dict[str, Any]], index: int, keyword: str | None, field: str) -> str:
        line_text = lines[index]["text"]
        if keyword:
            pattern = self._keyword_value_pattern(keyword)
            match = pattern.search(line_text)
            if match:
                remainder = normalize_whitespace(match.group(1))
                if remainder:
                    return remainder

        if field in MULTILINE_FIELDS:
            return self._collect_following_lines(lines, index, max_lines=3)

        remainder = self._collect_following_lines(lines, index, max_lines=1)
        if remainder:
            return remainder

        return ""

    def _collect_following_lines(self, lines: list[dict[str, Any]], index: int, max_lines: int = 2) -> str:
        collected: list[str] = []
        for next_index in range(index + 1, min(len(lines), index + max_lines + 1)):
            candidate = normalize_whitespace(lines[next_index]["text"])
            if not candidate:
                continue
            if self._identify_field(candidate)[0] is not None:
                break
            if self._looks_like_value_break(candidate):
                break
            collected.append(candidate)

        return normalize_whitespace(" ".join(collected))

    @staticmethod
    def _looks_like_value_break(candidate: str) -> bool:
        normalized = FieldMapper._normalize_for_match(candidate)
        if not normalized:
            return True
        if ":" in candidate:
            return True
        if re.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$", normalized):
            return True
        if re.match(r"^[0-9\s+()-]{8,}$", candidate):
            return True
        return False

    @staticmethod
    def _keyword_value_pattern(keyword: str) -> re.Pattern[str]:
        parts = [re.escape(part) for part in FieldMapper._normalize_for_match(keyword).split()]
        if not parts:
            return re.compile(r"$^")
        joined = r"\W+".join(parts)
        return re.compile(rf"\b{joined}\b(?:\s*[:\-–]?\s*)(.*)$", flags=re.IGNORECASE)

    def _post_process(self, fields: dict[str, str]) -> dict[str, str]:
        cleaned = {key: normalize_whitespace(value) for key, value in fields.items() if value}

        if "date_of_birth" in cleaned:
            cleaned["date_of_birth"] = normalize_date(cleaned["date_of_birth"], self.date_formats)
        if "signature_date" in cleaned:
            cleaned["signature_date"] = normalize_date(cleaned["signature_date"], self.date_formats)
        if "monthly_income" in cleaned:
            cleaned["monthly_income"] = normalize_numeric(cleaned["monthly_income"])
        if "loan_amount" in cleaned:
            cleaned["loan_amount"] = normalize_numeric(cleaned["loan_amount"])
        if "land_area" in cleaned:
            cleaned["land_area"] = normalize_numeric(cleaned["land_area"])
        if "nid_number" in cleaned:
            cleaned["nid_number"] = normalize_nid(cleaned["nid_number"])
        if "mobile_number" in cleaned:
            cleaned["mobile_number"] = normalize_phone(cleaned["mobile_number"])

        return cleaned
