from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from form_filler.docx_filler import DOCXFiller
from form_filler.pdf_filler import PDFFiller
from mapper.field_mapper import FieldMapper
from ocr.engine import OCREngine


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def write_uploaded_file(uploaded, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(uploaded.getbuffer())
    return target_path


def run_pipeline(
    input_path: Path,
    output_format: str,
    confidence: float,
    preprocess: bool,
    lang: str,
    use_gpu: bool,
    template_path: Path | None,
    show_boxes: bool,
) -> tuple[Path, dict[str, str], Path | None, list[dict[str, Any]], str | None]:
    cfg = load_config("config.yaml")
    out_cfg = cfg.get("output", {})

    engine = OCREngine(lang=lang, use_gpu=use_gpu)
    ocr_results = engine.run(str(input_path), min_confidence=confidence, preprocess=preprocess)

    mapper = FieldMapper(config_path="config.yaml")
    mapper.conf_threshold = confidence
    form_data = mapper.map(ocr_results)

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    debug_path = None
    if show_boxes:
        debug_path = outputs_dir / "ui_ocr_debug.jpg"
        engine.save_debug_image(str(input_path), ocr_results, str(debug_path), preprocess=preprocess)

    if output_format == "pdf":
        output_path = outputs_dir / "ui_filled_form.pdf"
        fallback_template = out_cfg.get("template_pdf")
        chosen_template = template_path if template_path else (Path(fallback_template) if fallback_template else None)
        filler = PDFFiller(template=str(chosen_template) if chosen_template else None)
        filler.fill(form_data, str(output_path))
        return output_path, form_data, debug_path, ocr_results, engine.last_error

    if output_format == "docx":
        output_path = outputs_dir / "ui_filled_form.docx"
        fallback_template = out_cfg.get("template_docx")
        chosen_template = template_path if template_path else (Path(fallback_template) if fallback_template else None)
        filler = DOCXFiller(template=str(chosen_template) if chosen_template else None)
        filler.fill(form_data, str(output_path))
        return output_path, form_data, debug_path, ocr_results, engine.last_error

    output_path = outputs_dir / "ui_filled_form.json"
    output_path.write_text(json.dumps(form_data, indent=2), encoding="utf-8")
    return output_path, form_data, debug_path, ocr_results, engine.last_error


def main() -> None:
    st.set_page_config(page_title="Home Loan OCR Auto-Filler", page_icon="📝", layout="wide")

    st.title("Home Loan OCR Auto-Filler")
    st.write("Upload a handwritten form, optionally upload your template format, and download the filled output.")

    with st.sidebar:
        st.header("Settings")
        output_format = st.selectbox("Output format", ["pdf", "docx", "json"], index=0)
        confidence = st.slider("OCR confidence threshold", min_value=0.1, max_value=1.0, value=0.8, step=0.05)
        lang = st.text_input("OCR language", value="en")
        preprocess = st.checkbox("Preprocess image", value=False)
        use_gpu = st.checkbox("Use GPU (if available)", value=False)
        show_boxes = st.checkbox("Show OCR boxes", value=False)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1) Upload Handwritten File")
        input_file = st.file_uploader(
            "Accepted formats: JPG, JPEG, PNG, PDF",
            type=["jpg", "jpeg", "png", "pdf"],
            key="input_file",
        )

    with col2:
        st.subheader("2) Upload Your Template (Optional)")
        template_types = ["pdf"] if output_format == "pdf" else (["docx"] if output_format == "docx" else ["pdf", "docx"])
        template_file = st.file_uploader(
            "Use this when you want output filled into your own format/template",
            type=template_types,
            key="template_file",
        )

    run_clicked = st.button("Process and Fill Form", type="primary", use_container_width=True)

    if not run_clicked:
        st.info("Upload your files and click Process and Fill Form.")
        return

    if not input_file:
        st.error("Please upload a handwritten input file.")
        return

    with st.spinner("Running OCR and filling template..."):
        with tempfile.TemporaryDirectory(prefix="loan_ocr_ui_") as temp_dir:
            temp_path = Path(temp_dir)
            input_path = write_uploaded_file(input_file, temp_path / input_file.name)

            template_path = None
            if template_file is not None:
                template_path = write_uploaded_file(template_file, temp_path / template_file.name)

            try:
                output_path, form_data, debug_path, ocr_results, ocr_error = run_pipeline(
                    input_path=input_path,
                    output_format=output_format,
                    confidence=confidence,
                    preprocess=preprocess,
                    lang=lang,
                    use_gpu=use_gpu,
                    template_path=template_path,
                    show_boxes=show_boxes,
                )
            except Exception as exc:
                st.error(f"Processing failed: {exc}")
                st.stop()

    st.success("Processing complete.")

    if ocr_error:
        st.warning(
            "OCR hit a runtime issue and returned no text blocks. "
            f"Details: `{ocr_error}`"
        )
    elif not ocr_results:
        st.warning("OCR completed, but no text regions were detected in the uploaded file.")
    elif not form_data:
        st.info(
            "OCR detected text, but none of it matched the configured field mappings. "
            "Try lowering the confidence threshold or enabling OCR boxes to inspect the detected text."
        )

    st.caption(f"OCR text blocks detected: {len(ocr_results)}")

    st.subheader("Extracted Fields")
    st.json(form_data)

    file_bytes = output_path.read_bytes()
    mime = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "json": "application/json",
    }[output_format]

    st.download_button(
        label=f"Download Filled {output_format.upper()}",
        data=file_bytes,
        file_name=output_path.name,
        mime=mime,
        use_container_width=True,
    )

    if debug_path and debug_path.exists():
        st.subheader("OCR Debug Boxes")
        st.image(str(debug_path), caption="Detected text regions")


if __name__ == "__main__":
    main()
