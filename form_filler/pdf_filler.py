from __future__ import annotations

from pathlib import Path

import fitz


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
                if name in form_data:
                    widget.field_value = str(form_data[name])
                    widget.update()
                    filled_widget = True

        if not filled_widget:
            self._draw_key_values(doc, form_data)

        doc.save(str(output))
        doc.close()
        return str(output)

    @staticmethod
    def _draw_key_values(doc: fitz.Document, form_data: dict[str, str]) -> None:
        page = doc[0]
        y = 72
        page.insert_text((72, 48), "Home Loan Application (Extracted Data)", fontsize=14)

        for key in sorted(form_data.keys()):
            value = form_data[key]
            page.insert_text((72, y), f"{key}: {value}", fontsize=10)
            y += 16
            if y > 790:
                page = doc.new_page(width=595, height=842)
                y = 72
