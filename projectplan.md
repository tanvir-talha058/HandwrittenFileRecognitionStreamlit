# 🏦 Home Loan Form Auto-Filler using PaddleOCR

> Automatically reads handwritten home loan application forms using PaddleOCR and fills in a clean, structured digital output form.

---

## 📌 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Field Mapping](#field-mapping)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

This tool allows bank staff or applicants to:

1. **Drop** a scanned / photographed handwritten home loan form (JPG, PNG, or PDF)
2. **Automatically extract** all handwritten text using [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
3. **Map** the recognized text to the correct fields of the bank's standard loan form
4. **Output** a clean, filled digital form (PDF / Word / structured JSON)

No manual data entry. No re-typing. Just drop the image and get a filled form.

---

## How It Works

```
┌──────────────────────┐
│  Handwritten Form    │  ← JPG / PNG / PDF scan
│  (dropped by user)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   PaddleOCR Engine   │  ← Detects text regions + recognizes handwriting
│  (Detection + Recog) │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Field Mapper       │  ← Matches OCR output to known form fields
│   (NLP / Regex)      │     (Name, Loan Amount, DOB, Address, etc.)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Form Filler        │  ← Populates the bank's standard digital form
│   (PDF / DOCX)       │
└──────────────────────┘
```

PaddleOCR uses a **3-stage pipeline**:
- **Detection** — finds where text is located (DB model)
- **Direction Classification** — handles rotated/slanted text
- **Recognition** — reads the actual characters (CRNN model)

---

## Project Structure

```
home-loan-ocr/
│
├── README.md                    ← You are here
├── requirements.txt             ← Python dependencies
├── config.yaml                  ← Field mapping and OCR settings
│
├── main.py                      ← Entry point: drop form → get filled output
│
├── ocr/
│   ├── __init__.py
│   ├── engine.py                ← PaddleOCR wrapper (init, run, post-process)
│   └── preprocessor.py         ← Image preprocessing (deskew, denoise, resize)
│
├── mapper/
│   ├── __init__.py
│   ├── field_mapper.py          ← Maps OCR text blocks → form fields
│   └── validators.py            ← Validates field values (dates, amounts, NID)
│
├── form_filler/
│   ├── __init__.py
│   ├── pdf_filler.py            ← Fills PDF form template with extracted data
│   └── docx_filler.py           ← Fills DOCX form template with extracted data
│
├── templates/
│   ├── loan_form_template.pdf   ← Bank's official blank form (PDF)
│   └── loan_form_template.docx  ← Bank's official blank form (Word)
│
├── samples/
│   ├── sample_handwritten.jpg   ← Example handwritten form for testing
│   └── sample_output.pdf        ← Example filled output
│
└── outputs/                     ← Filled forms are saved here (auto-created)
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.8 – 3.11 | 3.12 not yet fully supported by PaddlePaddle |
| pip | latest | `pip install --upgrade pip` |
| OS | Windows 10/11, Ubuntu 20.04+, macOS 12+ | |
| RAM | ≥ 4 GB | 8 GB recommended for large forms |
| GPU (optional) | CUDA 11.2+ | CPU works fine for single forms |

---

## Installation

### Step 1 — Clone this repository

```bash
git clone https://github.com/your-org/home-loan-ocr.git
cd home-loan-ocr
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install PaddlePaddle

Choose the version that matches your system:

**CPU only (most users):**
```bash
pip install paddlepaddle
```

**GPU (CUDA 11.2):**
```bash
pip install paddlepaddle-gpu==2.6.1.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

> For other CUDA versions, see the [official install page](https://www.paddlepaddle.org.cn/install/quick).

### Step 4 — Install PaddleOCR

```bash
pip install paddleocr
```

### Step 5 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` contents:**
```
paddlepaddle>=2.5.0
paddleocr>=2.7.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
PyMuPDF>=1.23.0        # for PDF reading and filling
python-docx>=1.1.0     # for DOCX output
pyyaml>=6.0
regex>=2023.0
```

### Step 6 — Download OCR Models (auto on first run)

PaddleOCR downloads models automatically on first use. To pre-download:

```bash
python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en')"
```

> Models are saved to `~/.paddleocr/` (~200 MB total).

---

## Usage

### Basic Usage — Drop a Form Image

```bash
python main.py --input samples/sample_handwritten.jpg
```

This will:
- Run OCR on the image
- Map extracted text to form fields
- Save a filled PDF to `outputs/filled_loan_form.pdf`

---

### Full Options

```bash
python main.py \
  --input  path/to/handwritten_form.jpg \
  --output outputs/my_filled_form.pdf \
  --format pdf \
  --show-boxes \
  --confidence 0.85
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | Path to scanned/photographed form (JPG, PNG, PDF) |
| `--output` | `outputs/filled_form.pdf` | Where to save the filled form |
| `--format` | `pdf` | Output format: `pdf` or `docx` |
| `--show-boxes` | off | Save a debug image showing detected text regions |
| `--confidence` | `0.80` | Minimum OCR confidence score (0.0 – 1.0) |
| `--lang` | `en` | Language: `en`, `bn` (Bangla), etc. |

---

### Python API

```python
from ocr.engine import OCREngine
from mapper.field_mapper import FieldMapper
from form_filler.pdf_filler import PDFFiller

# 1. Run OCR
engine = OCREngine(lang='en', use_gpu=False)
ocr_results = engine.run("samples/sample_handwritten.jpg")

# 2. Map to form fields
mapper = FieldMapper(config_path="config.yaml")
form_data = mapper.map(ocr_results)

print(form_data)
# {
#   "applicant_name": "Md. Karim Hossain",
#   "loan_amount": "2500000",
#   "date_of_birth": "15/03/1985",
#   "nid_number": "1234567890",
#   "monthly_income": "75000",
#   ...
# }

# 3. Fill the form
filler = PDFFiller(template="templates/loan_form_template.pdf")
filler.fill(form_data, output_path="outputs/filled_form.pdf")
```

---

## Field Mapping

The system recognizes and maps the following home loan form fields:

| Field | Mapped Key | Example Value |
|-------|-----------|---------------|
| Applicant Full Name | `applicant_name` | Md. Rahim Uddin |
| Date of Birth | `date_of_birth` | 12/06/1980 |
| National ID No. | `nid_number` | 19801234567890 |
| Father's Name | `father_name` | Md. Abdul Karim |
| Mother's Name | `mother_name` | Rashida Begum |
| Present Address | `present_address` | House 12, Road 4, Mirpur |
| Permanent Address | `permanent_address` | Village: Bogura |
| Mobile Number | `mobile_number` | 01712345678 |
| Occupation | `occupation` | Service |
| Monthly Income (BDT) | `monthly_income` | 60,000 |
| Loan Amount Requested | `loan_amount` | 25,00,000 |
| Loan Purpose | `loan_purpose` | Purchase of flat |
| Property Address | `property_address` | Bashundhara, Dhaka |
| Land Area (sqft) | `land_area` | 1200 |
| Co-applicant Name | `co_applicant_name` | Fatema Begum |
| Guarantor Name | `guarantor_name` | Md. Salam |
| Repayment Period | `repayment_period` | 20 years |
| Signature Date | `signature_date` | 06/04/2026 |

Field mappings are configured in `config.yaml` and can be customized for your bank's specific form layout.

---

## Configuration

Edit `config.yaml` to customize behavior:

```yaml
ocr:
  lang: en                   # 'en' for English, 'bn' for Bangla
  use_angle_cls: true        # handle rotated text
  use_gpu: false             # set true if CUDA GPU is available
  det_db_thresh: 0.3         # text detection sensitivity (lower = detect more)
  rec_char_type: EN          # character set

field_mapping:
  fuzzy_match: true          # enable fuzzy label matching
  confidence_threshold: 0.80 # ignore OCR results below this score
  date_formats:
    - "%d/%m/%Y"
    - "%d-%m-%Y"
    - "%d %B %Y"

output:
  format: pdf                # 'pdf' or 'docx'
  template: templates/loan_form_template.pdf
  output_dir: outputs/
  save_debug_image: false    # save image with bounding boxes drawn
```

---

## Troubleshooting

### OCR not detecting text

- Ensure the image is **at least 300 DPI** for best results
- Avoid heavy shadows or glare on the form
- Try preprocessing: `python main.py --input form.jpg --preprocess`
- Lower detection threshold: add `det_db_thresh: 0.2` in `config.yaml`

### Wrong fields being mapped

- Run with `--show-boxes` to see what text was detected and where
- Adjust field label keywords in `config.yaml` under `field_mapping`
- Check that your form template matches the expected layout

### PaddlePaddle installation fails

- Ensure Python version is 3.8–3.11
- Try: `pip install paddlepaddle --no-cache-dir`
- On Apple Silicon (M1/M2): use `pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/mac/cpu/develop.html`

### Model download fails

- Check internet connection
- Manually download from [PaddleOCR Model List](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md)
- Place in `~/.paddleocr/whl/`

### Output PDF fields are empty

- Confirm your `loan_form_template.pdf` has **fillable AcroForm fields**
- Or use the DOCX output mode: `--format docx`
- Field names in the PDF must match keys in `config.yaml`

---

## Supported Input Formats

| Format | Notes |
|--------|-------|
| `.jpg` / `.jpeg` | Recommended for photos |
| `.png` | Best quality scans |
| `.pdf` | First page extracted automatically |
| `.tiff` | High-res scanner output |

---

## References

| Resource | Link |
|----------|------|
| PaddleOCR GitHub | https://github.com/PaddlePaddle/PaddleOCR |
| PaddleOCR Documentation | https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html |
| AI Studio Demo | https://aistudio.baidu.com/paddleocr |
| PaddlePaddle Install | https://www.paddlepaddle.org.cn/install/quick |
| Model Zoo | https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md |

---

## License

This project is for internal bank use. PaddleOCR is licensed under the [Apache 2.0 License](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE).

---

> **Note:** For Bangla (Bengali) handwriting support, set `lang: 'en'` and include a fine-tuned Bangla model. PaddleOCR supports multilingual OCR — see the [multilingual model list](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/multi_languages_en.md) for details.