from __future__ import annotations

import re
from datetime import datetime


def normalize_numeric(value: str) -> str:
    digits = re.sub(r"[^\d.]", "", value)
    if not digits:
        return ""

    if digits.count(".") <= 1:
        return digits

    first_dot = digits.find(".")
    return digits[: first_dot + 1] + digits[first_dot + 1 :].replace(".", "")


def normalize_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if digits.startswith("88") and len(digits) > 11:
        digits = digits[2:]
    return digits


def normalize_nid(value: str) -> str:
    return re.sub(r"\D", "", value)


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_date(value: str, supported_formats: list[str]) -> str:
    compact = normalize_whitespace(value)
    if not compact:
        return ""

    for fmt in supported_formats:
        try:
            return datetime.strptime(compact, fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue

    date_match = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", compact)
    if not date_match:
        return compact

    day, month, year = date_match.groups()
    year = f"20{year}" if len(year) == 2 else year
    try:
        return datetime(int(year), int(month), int(day)).strftime("%d/%m/%Y")
    except ValueError:
        return compact
