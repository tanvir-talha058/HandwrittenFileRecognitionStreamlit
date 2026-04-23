from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np
import streamlit as st
import yaml

from form_filler.docx_filler import DOCXFiller
from form_filler.pdf_filler import PDFFiller
from mapper.field_mapper import FieldMapper
from ocr.engine import OCREngine


UPLOADS_DIR = Path("outputs/uploads")
PAGE_STRIDE = 10000


def _paddle_runtime_summary() -> str | None:
    try:
        import paddle

        compiled_with_cuda = bool(getattr(paddle, "is_compiled_with_cuda", lambda: False)())
        try:
            device = paddle.device.get_device()
        except Exception:
            device = "unknown"
        return f"Paddle {paddle.__version__} · device: {device} · CUDA build: {compiled_with_cuda}"
    except Exception:
        return None


def _paddle_cuda_available() -> bool:
    try:
        import paddle

        return bool(getattr(paddle, "is_compiled_with_cuda", lambda: False)())
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def _cached_ocr_engine(lang: str, use_gpu: bool, use_angle_cls: bool, det_db_thresh: float) -> OCREngine:
    return OCREngine(lang=lang, use_gpu=use_gpu, use_angle_cls=use_angle_cls, det_db_thresh=det_db_thresh)


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


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_filename(name: str) -> str:
    keep = []
    for ch in name.strip():
        if ch.isalnum() or ch in {".", "_", "-"}:
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    return "".join(keep) or "upload"


def _sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _init_session_state() -> None:
    st.session_state.setdefault("recent_files", [])
    st.session_state.setdefault("selected_file_id", None)
    st.session_state.setdefault("results_by_key", {})
    st.session_state.setdefault("uploader_nonce", 0)
    st.session_state.setdefault("last_uploaded_digest", None)
    st.session_state.setdefault("last_template_digest", None)
    st.session_state.setdefault("template_entry", None)


def _reset_parsing_state() -> None:
    st.session_state["results_by_key"] = {}
    st.session_state["selected_file_id"] = None
    st.session_state["uploader_nonce"] = int(st.session_state.get("uploader_nonce", 0)) + 1
    st.session_state["last_uploaded_digest"] = None
    st.session_state["last_template_digest"] = None
    st.session_state["template_entry"] = None


def _save_upload_to_outputs(uploaded) -> dict[str, Any]:
    data = uploaded.getvalue()
    digest = _sha1_bytes(data)
    safe_name = _safe_filename(uploaded.name)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    stored_path = UPLOADS_DIR / f"{digest[:12]}_{safe_name}"
    if not stored_path.exists():
        stored_path.write_bytes(data)
    return {
        "id": digest[:12],
        "digest": digest,
        "name": uploaded.name,
        "path": str(stored_path),
        "size": len(data),
        "uploaded_at": _now_iso(),
    }


def _upsert_recent_file(entry: dict[str, Any]) -> None:
    recents: list[dict[str, Any]] = list(st.session_state.get("recent_files", []))
    recents = [item for item in recents if item.get("id") != entry.get("id")]
    recents.insert(0, entry)
    st.session_state["recent_files"] = recents[:20]
    st.session_state["selected_file_id"] = entry.get("id")


def _get_selected_entry() -> dict[str, Any] | None:
    selected_id = st.session_state.get("selected_file_id")
    if not selected_id:
        return None
    for item in st.session_state.get("recent_files", []):
        if item.get("id") == selected_id:
            return item
    return None


def _file_suffix(path: Path) -> str:
    return path.suffix.lower().lstrip(".")


def _render_pdf_page_bgr(path: Path, page_number: int) -> tuple[np.ndarray, int]:
    doc = fitz.open(path)
    try:
        n_pages = len(doc)
        page_number = max(1, min(page_number, n_pages))
        page = doc.load_page(page_number - 1)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
        if pixmap.n == 4:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return bgr, n_pages
    finally:
        doc.close()


def _render_source_preview(path: Path, page_number: int) -> tuple[Any, int, str]:
    suffix = _file_suffix(path)
    if suffix == "pdf":
        bgr, n_pages = _render_pdf_page_bgr(path, page_number)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, n_pages, "image"

    if suffix in {"jpg", "jpeg", "png"}:
        bgr = cv2.imread(str(path))
        if bgr is None:
            return "Unsupported image file.", 1, "text"
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, 1, "image"

    if suffix == "docx":
        try:
            blocks = OCREngine._extract_docx_text_lines(path)  # type: ignore[attr-defined]
        except Exception as exc:
            return f"Failed to read DOCX: {exc}", 1, "text"
        return "\n".join(blocks) if blocks else "DOCX contains no readable text.", 1, "text"

    if suffix == "txt":
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            return f"Failed to read TXT: {exc}", 1, "text"
        return text or "TXT is empty.", 1, "text"

    return f"Unsupported file type: .{suffix}", 1, "text"


def _draw_ocr_overlay(
    source_path: Path,
    ocr_results: list[dict[str, Any]],
    page_number: int,
) -> Any | None:
    suffix = _file_suffix(source_path)
    if suffix == "pdf":
        base_bgr, _ = _render_pdf_page_bgr(source_path, page_number)
        y_offset = (page_number - 1) * PAGE_STRIDE
        blocks = [
            item
            for item in ocr_results
            if int(item.get("page", 1)) == page_number and isinstance(item.get("bbox"), list)
        ]
    elif suffix in {"jpg", "jpeg", "png"}:
        base_bgr = cv2.imread(str(source_path))
        if base_bgr is None:
            return None
        y_offset = 0
        blocks = [item for item in ocr_results if isinstance(item.get("bbox"), list)]
    else:
        return None

    overlay = base_bgr.copy()
    for block in blocks:
        raw_bbox = block.get("bbox", [])
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            continue
        pts: list[list[int]] = []
        for point in raw_bbox:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                pts = []
                break
            try:
                x = int(round(float(point[0])))
                y = int(round(float(point[1]) - y_offset))
            except (TypeError, ValueError):
                pts = []
                break
            pts.append([x, y])
        if len(pts) != 4:
            continue
        np_pts = np.array(pts, dtype=np.int32)
        cv2.polylines(overlay, [np_pts], True, (156, 39, 176), 2)
        label = str(block.get("text", ""))[:24].strip()
        if label:
            cv2.putText(
                overlay,
                label,
                tuple(np_pts[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (225, 29, 72),
                1,
                cv2.LINE_AA,
            )
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def run_pipeline(
    input_path: Path,
    output_format: str,
    confidence: float,
    preprocess: bool,
    lang: str,
    use_gpu: bool,
    template_path: Path | None,
    show_boxes: bool,
    use_angle_cls: bool = True,
    run_id: str = "ui",
) -> tuple[Path, dict[str, str], Path | None, list[dict[str, Any]], list[str], str | None]:
    cfg = load_config("config.yaml")
    out_cfg = cfg.get("output", {})
    ocr_cfg = cfg.get("ocr", {})
    det_db_thresh = float(ocr_cfg.get("det_db_thresh", 0.3))

    engine = _cached_ocr_engine(
        lang=lang,
        use_gpu=use_gpu,
        use_angle_cls=use_angle_cls,
        det_db_thresh=det_db_thresh,
    )
    ocr_results = engine.run(str(input_path), min_confidence=confidence, preprocess=preprocess)

    mapper = FieldMapper(config_path="config.yaml")
    mapper.conf_threshold = confidence
    form_data, transcript_lines, _ = mapper.map_with_context(ocr_results)

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    debug_path = None
    if show_boxes and input_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        debug_path = outputs_dir / f"{run_id}_ocr_debug.jpg"
        engine.save_debug_image(str(input_path), ocr_results, str(debug_path), preprocess=preprocess)

    if output_format == "pdf":
        output_path = outputs_dir / f"{run_id}_filled_form.pdf"
        fallback_template = out_cfg.get("template_pdf")
        chosen_template = template_path if template_path else (Path(fallback_template) if fallback_template else None)
        filler = PDFFiller(template=str(chosen_template) if chosen_template else None)
        filler.fill(form_data, str(output_path))
        return output_path, form_data, debug_path, ocr_results, transcript_lines, engine.last_error

    if output_format == "docx":
        output_path = outputs_dir / f"{run_id}_filled_form.docx"
        fallback_template = out_cfg.get("template_docx")
        chosen_template = template_path if template_path else (Path(fallback_template) if fallback_template else None)
        filler = DOCXFiller(template=str(chosen_template) if chosen_template else None)
        filler.fill(form_data, str(output_path))
        return output_path, form_data, debug_path, ocr_results, transcript_lines, engine.last_error

    output_path = outputs_dir / f"{run_id}_filled_form.json"
    output_path.write_text(json.dumps(form_data, indent=2), encoding="utf-8")
    return output_path, form_data, debug_path, ocr_results, transcript_lines, engine.last_error


def main() -> None:
    st.set_page_config(page_title="Form OCR Parser", page_icon="🧾", layout="wide")

    _init_session_state()

    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] .stButton > button { width: 100%; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### PaddleOCR-style Parsing")
        if st.button("+ New Parsing", use_container_width=True):
            _reset_parsing_state()

        uploader_key = f"source_uploader_{st.session_state.get('uploader_nonce', 0)}"
        input_file = st.file_uploader(
            "Upload source file",
            type=["jpg", "jpeg", "png", "pdf", "docx", "txt"],
            key=uploader_key,
            label_visibility="collapsed",
        )

        if input_file is not None:
            entry = _save_upload_to_outputs(input_file)
            if entry["digest"] != st.session_state.get("last_uploaded_digest"):
                st.session_state["last_uploaded_digest"] = entry["digest"]
                _upsert_recent_file(entry)

        st.divider()

        st.markdown("#### System Settings")
        output_format = st.selectbox("Output format", ["pdf", "docx", "json"], index=0)
        confidence = st.slider("OCR confidence", min_value=0.1, max_value=1.0, value=0.55, step=0.05)
        lang = st.text_input("OCR language", value="en")
        preprocess = st.checkbox("Preprocess image", value=False)
        use_gpu = st.checkbox("Use GPU", value=False)
        runtime = _paddle_runtime_summary()
        if runtime:
            st.caption(runtime)
        show_boxes = st.checkbox("Show overlay boxes", value=True)

        st.divider()

        st.markdown("#### Recents")
        recents = st.session_state.get("recent_files", [])
        if recents:
            options = {f"{item['name']}  ·  {item['uploaded_at']}": item["id"] for item in recents}
            labels = list(options.keys())
            selected_id = st.session_state.get("selected_file_id")
            selected_index = 0
            if selected_id:
                for idx, label in enumerate(labels):
                    if options[label] == selected_id:
                        selected_index = idx
                        break
            selected_label = st.radio(
                "Recent uploads",
                labels,
                index=selected_index,
                label_visibility="collapsed",
            )
            st.session_state["selected_file_id"] = options[selected_label]

            cols = st.columns(2)
            with cols[0]:
                if st.button("Remove", use_container_width=True):
                    selected_id = st.session_state.get("selected_file_id")
                    st.session_state["recent_files"] = [item for item in recents if item.get("id") != selected_id]
                    if st.session_state["recent_files"]:
                        st.session_state["selected_file_id"] = st.session_state["recent_files"][0]["id"]
                    else:
                        st.session_state["selected_file_id"] = None
                        st.session_state["results_by_key"] = {}
            with cols[1]:
                if st.button("Clear all", use_container_width=True):
                    st.session_state["recent_files"] = []
                    st.session_state["selected_file_id"] = None
                    st.session_state["results_by_key"] = {}
        else:
            st.caption("No recent files yet.")

        st.divider()

        st.markdown("#### Template (Optional)")
        template_types = ["pdf"] if output_format == "pdf" else (["docx"] if output_format == "docx" else ["pdf", "docx"])
        template_key = f"template_uploader_{st.session_state.get('uploader_nonce', 0)}"
        template_file = st.file_uploader(
            "Upload template",
            type=template_types,
            key=template_key,
            label_visibility="collapsed",
        )

        template_entry = st.session_state.get("template_entry")
        if template_file is not None:
            template_entry = _save_upload_to_outputs(template_file)
            if template_entry["digest"] != st.session_state.get("last_template_digest"):
                st.session_state["last_template_digest"] = template_entry["digest"]
            st.session_state["template_entry"] = template_entry

        if template_entry:
            st.caption(f"Template: `{template_entry['name']}`")
            if st.button("Clear template", use_container_width=True):
                st.session_state["template_entry"] = None
                st.session_state["last_template_digest"] = None
                template_entry = None

    selected = _get_selected_entry()
    if not selected:
        st.title("Form OCR Parser")
        st.info("Upload a file from the sidebar to start parsing.")
        return

    source_path = Path(selected["path"])
    st.caption(f"Selected: `{selected['name']}`  ·  {_file_suffix(source_path).upper()}  ·  {selected['size']:,} bytes")

    left, right = st.columns([0.56, 0.44], gap="large")

    with left:
        st.markdown("### Source File")
        page_number = 1
        n_pages = 1

        if _file_suffix(source_path) == "pdf":
            try:
                _, n_pages = _render_pdf_page_bgr(source_path, 1)
            except Exception as exc:
                st.error(f"Failed to open PDF: {exc}")
                return
            page_number = int(st.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1))

        preview, n_pages, preview_kind = _render_source_preview(source_path, page_number)
        if preview_kind == "image":
            st.image(preview, use_container_width=True)
        else:
            st.text(preview)

    with right:
        st.markdown("### Parsing")
        parsing_model = st.selectbox("Parsing model", ["PP-OCR (Default)", "PP-OCR (No Angle CLS)"], index=0)

        run_clicked = st.button("Run parsing", type="primary", use_container_width=True)

        run_key = (
            selected["id"],
            output_format,
            float(confidence),
            bool(preprocess),
            str(lang),
            bool(use_gpu),
            bool(show_boxes),
            parsing_model,
            template_entry["id"] if template_entry else None,
        )

        if run_clicked:
            with st.spinner("Running OCR and mapping fields..."):
                with tempfile.TemporaryDirectory(prefix="loan_ocr_ui_") as temp_dir:
                    temp_path = Path(temp_dir)
                    temp_input_path = temp_path / source_path.name
                    temp_input_path.write_bytes(source_path.read_bytes())

                    template_path = None
                    if template_entry:
                        template_source = Path(template_entry["path"])
                        template_path = temp_path / template_source.name
                        template_path.write_bytes(template_source.read_bytes())

                    use_angle_cls = parsing_model == "PP-OCR (Default)"
                    engine_lang = lang.strip() or "en"
                    effective_use_gpu = bool(use_gpu and _paddle_cuda_available())
                    if use_gpu and not effective_use_gpu:
                        st.warning("GPU requested but Paddle is CPU-only here; running OCR on CPU.")

                    try:
                        output_path, form_data, debug_path, ocr_results, transcript_lines, ocr_error = run_pipeline(
                            input_path=temp_input_path,
                            output_format=output_format,
                            confidence=confidence,
                            preprocess=preprocess,
                            lang=engine_lang,
                            use_gpu=effective_use_gpu,
                            template_path=template_path,
                            show_boxes=show_boxes,
                            use_angle_cls=use_angle_cls,
                            run_id=f"{selected['id']}_{template_entry['id'] if template_entry else 'no_template'}",
                        )
                    except Exception as exc:
                        st.error(f"Processing failed: {exc}")
                        st.stop()

            st.session_state["results_by_key"][run_key] = {
                "output_path": str(output_path),
                "form_data": form_data,
                "debug_path": str(debug_path) if debug_path else None,
                "ocr_results": ocr_results,
                "transcript_lines": transcript_lines,
                "ocr_error": ocr_error,
            }

        results = st.session_state.get("results_by_key", {}).get(run_key)

        tabs = st.tabs(["Document parsing", "JSON"])
        with tabs[0]:
            if not results:
                st.info("Click Run parsing to generate results.")
            else:
                ocr_error = results.get("ocr_error")
                ocr_results = results.get("ocr_results") or []
                form_data = results.get("form_data") or {}

                if ocr_error:
                    st.warning(f"OCR runtime issue: `{ocr_error}`")

                st.caption(f"OCR blocks: {len(ocr_results)}")

                if show_boxes:
                    overlay = _draw_ocr_overlay(source_path, ocr_results, page_number)
                    if overlay is not None:
                        st.image(overlay, caption="Overlay preview", use_container_width=True)

                st.subheader("Extracted Fields")
                st.json(form_data)

                with st.expander("OCR transcript", expanded=False):
                    transcript_lines = results.get("transcript_lines") or []
                    if transcript_lines:
                        st.text("\n".join(transcript_lines))
                    else:
                        st.write("No transcript assembled.")

                with st.expander("OCR blocks table", expanded=False):
                    rows = []
                    for item in ocr_results:
                        rows.append(
                            {
                                "page": int(item.get("page", 1)),
                                "confidence": float(item.get("confidence", 0.0)),
                                "text": str(item.get("text", "")),
                            }
                        )
                    if rows:
                        st.dataframe(rows, use_container_width=True, hide_index=True)
                    else:
                        st.write("No OCR blocks.")

                output_path = Path(results["output_path"])
                if output_path.exists():
                    file_bytes = output_path.read_bytes()
                    mime = {
                        "pdf": "application/pdf",
                        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "json": "application/json",
                    }[output_format]
                    st.download_button(
                        label=f"Download {output_format.upper()}",
                        data=file_bytes,
                        file_name=output_path.name,
                        mime=mime,
                        use_container_width=True,
                    )

                debug_path = results.get("debug_path")
                if debug_path and Path(debug_path).exists():
                    with st.expander("Debug image (first page only)", expanded=False):
                        st.image(str(debug_path), use_container_width=True)

        with tabs[1]:
            if not results:
                st.info("No JSON yet.")
            else:
                st.json(
                    {
                        "file": {"id": selected["id"], "name": selected["name"]},
                        "settings": {
                            "output_format": output_format,
                            "confidence": confidence,
                            "lang": lang,
                            "preprocess": preprocess,
                            "use_gpu": use_gpu,
                            "show_boxes": show_boxes,
                            "parsing_model": parsing_model,
                        },
                        "extracted_fields": results.get("form_data") or {},
                        "ocr_results": results.get("ocr_results") or [],
                    }
                )


if __name__ == "__main__":
    main()
