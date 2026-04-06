from __future__ import annotations

from pathlib import Path

from docx import Document


class DOCXFiller:
    def __init__(self, template: str | None = None) -> None:
        self.template = template

    def fill(self, form_data: dict[str, str], output_path: str) -> str:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if self.template and Path(self.template).exists():
            document = Document(self.template)
            self._replace_placeholders(document, form_data)
        else:
            document = Document()
            document.add_heading("Home Loan Application (Extracted Data)", level=1)
            table = document.add_table(rows=1, cols=2)
            table.rows[0].cells[0].text = "Field"
            table.rows[0].cells[1].text = "Value"
            for key in sorted(form_data.keys()):
                row = table.add_row().cells
                row[0].text = key
                row[1].text = str(form_data[key])

        document.save(str(output))
        return str(output)

    @staticmethod
    def _replace_placeholders(document: Document, form_data: dict[str, str]) -> None:
        for paragraph in document.paragraphs:
            for key, value in form_data.items():
                placeholder = f"{{{{{key}}}}}"
                if placeholder in paragraph.text:
                    paragraph.text = paragraph.text.replace(placeholder, str(value))

        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    for key, value in form_data.items():
                        placeholder = f"{{{{{key}}}}}"
                        if placeholder in cell.text:
                            cell.text = cell.text.replace(placeholder, str(value))
