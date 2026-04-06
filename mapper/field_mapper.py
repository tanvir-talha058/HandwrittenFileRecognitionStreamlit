from __future__ import annotations

import re
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
    "applicant_name": ["applicant name", "full name", "name"],
    "date_of_birth": ["date of birth", "dob", "birth"],
    "nid_number": ["nid", "national id", "nid number"],
    "father_name": ["father", "father name"],
    "mother_name": ["mother", "mother name"],
    "present_address": ["present address", "current address"],
    "permanent_address": ["permanent address"],
    "mobile_number": ["mobile", "phone", "contact"],
    "occupation": ["occupation", "profession"],
    "monthly_income": ["monthly income", "income"],
    "loan_amount": ["loan amount", "amount requested", "loan"],
    "loan_purpose": ["loan purpose", "purpose"],
    "property_address": ["property address"],
    "land_area": ["land area", "sqft", "area"],
    "co_applicant_name": ["co-applicant", "co applicant"],
    "guarantor_name": ["guarantor"],
    "repayment_period": ["repayment period", "tenure"],
    "signature_date": ["signature date", "date"],
}


class FieldMapper:
    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = self._load_config(config_path)

        mapping_cfg = self.config.get("field_mapping", {})
        field_cfg = mapping_cfg.get("fields", {})

        self.date_formats = mapping_cfg.get("date_formats", ["%d/%m/%Y", "%d-%m-%Y"])
        self.conf_threshold = float(mapping_cfg.get("confidence_threshold", 0.8))

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
        lines = [
            normalize_whitespace(block.get("text", ""))
            for block in ocr_results
            if block.get("text") and float(block.get("confidence", 0.0)) >= self.conf_threshold
        ]

        combined_text = "\n".join(lines)
        extracted: dict[str, str] = {}

        for line in lines:
            lower_line = line.lower()
            for field, keywords in self.field_keywords.items():
                if field in extracted:
                    continue

                for keyword in keywords:
                    if keyword in lower_line:
                        value = self._extract_labeled_value(line, keyword)
                        if value:
                            extracted[field] = value
                            break

        # Fallback regex extraction for frequent fields.
        fallback_patterns = {
            "date_of_birth": r"(?:dob|date\s*of\s*birth)[:\s-]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
            "mobile_number": r"(?:mobile|phone|contact)[:\s-]*([+0-9\-\s]{10,15})",
            "nid_number": r"(?:nid|national\s*id)[:\s-]*([0-9\-\s]{8,20})",
            "monthly_income": r"(?:monthly\s*income|income)[:\s-]*([0-9,\.\s]+)",
            "loan_amount": r"(?:loan\s*amount|amount\s*requested|amount)[:\s-]*([0-9,\.\s]+)",
            "repayment_period": r"(?:repayment\s*period|tenure)[:\s-]*([A-Za-z0-9\s]+)",
        }

        lower_combined = combined_text.lower()
        for field, pattern in fallback_patterns.items():
            if field in extracted:
                continue
            match = re.search(pattern, lower_combined, flags=re.IGNORECASE)
            if match:
                extracted[field] = normalize_whitespace(match.group(1))

        return self._post_process(extracted)

    @staticmethod
    def _extract_labeled_value(line: str, keyword: str) -> str:
        pattern = re.compile(rf"{re.escape(keyword)}\s*[:\-]?\s*(.+)$", flags=re.IGNORECASE)
        match = pattern.search(line)
        if not match:
            return ""
        return normalize_whitespace(match.group(1))

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
