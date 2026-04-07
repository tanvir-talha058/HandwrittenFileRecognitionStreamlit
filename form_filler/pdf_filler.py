from __future__ import annotations

import re
from pathlib import Path

import fitz


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
            self._write_values_next_to_labels(doc, form_data)

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

    def _write_values_next_to_labels(self, doc: fitz.Document, form_data: dict[str, str]) -> None:
        pending = {key: str(value) for key, value in form_data.items() if str(value).strip()}

        for page in doc:
            page_width = float(page.rect.width)
            for key in list(pending.keys()):
                labels = FIELD_LABELS.get(key, [key.replace("_", " ")])
                anchor = None
                for label in labels:
                    anchor = self._search_keyword_rect(page, label)
                    if anchor is not None:
                        break

                if anchor is None:
                    continue

                value = pending[key]
                x = min(anchor.x1 + 12, page_width - 180)
                y = anchor.y1 + 9
                page.insert_text(
                    (x, y),
                    value,
                    fontsize=10,
                    color=(0, 0, 0),
                    overlay=True,
                )
                del pending[key]

        if not pending:
            return

        summary_page = doc.new_page(width=595, height=842)
        y = 72
        summary_page.insert_text((72, 48), "Unplaced Extracted Fields", fontsize=14)
        for key in sorted(pending.keys()):
            summary_page.insert_text((72, y), f"{key}: {pending[key]}", fontsize=10)
            y += 16
            if y > 790:
                summary_page = doc.new_page(width=595, height=842)
                y = 72
