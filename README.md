# Home Loan Form Auto-Filler

Handwritten home loan form extractor built with PaddleOCR.

## Web Frontend

You can use a browser UI to upload handwritten forms and your own template format.

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start frontend:

```bash
streamlit run streamlit_app.py
```

3. In the UI:

- Upload handwritten input file (JPG/PNG/PDF)
- Select output format (PDF/DOCX/JSON)
- Optionally upload your own output template (PDF or DOCX)
- Click Process and Fill Form
- Download the generated file

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run OCR pipeline:

```bash
python main.py --input samples/sample_handwritten.jpg --format pdf
```

## CLI

```bash
python main.py \
  --input path/to/form.jpg \
  --output outputs/my_filled_form.pdf \
  --format pdf \
  --show-boxes \
  --confidence 0.85 \
  --lang en
```

Supported output formats: `pdf`, `docx`, `json`.

## Notes

- Add official template files in `templates/`.
- If templates are missing, fallback outputs are still generated.
- PDF input uses the first page.
