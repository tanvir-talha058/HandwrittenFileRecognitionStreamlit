from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from form_filler.docx_filler import DOCXFiller
from form_filler.pdf_filler import PDFFiller
from mapper.field_mapper import FieldMapper
from ocr.engine import OCREngine


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Home Loan Form Auto-Filler using PaddleOCR")
    parser.add_argument("--input", required=True, help="Path to handwritten form (JPG, PNG, or PDF)")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--format", choices=["pdf", "docx", "json"], help="Output format")
    parser.add_argument("--show-boxes", action="store_true", help="Save debug image with OCR boxes")
    parser.add_argument("--confidence", type=float, default=None, help="Minimum OCR confidence")
    parser.add_argument("--lang", default=None, help="OCR language, e.g., en, bn")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU for PaddleOCR")
    parser.add_argument("--preprocess", action="store_true", help="Enable image preprocessing")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    return parser.parse_args()


def resolve_output_path(args: argparse.Namespace, cfg: dict[str, Any], output_format: str) -> Path:
    if args.output:
        return Path(args.output)

    out_dir = Path(cfg.get("output", {}).get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = output_format if output_format != "json" else "json"
    return out_dir / f"filled_form.{suffix}"


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    ocr_cfg = cfg.get("ocr", {})
    mapping_cfg = cfg.get("field_mapping", {})
    out_cfg = cfg.get("output", {})

    output_format = args.format or out_cfg.get("format", "pdf")
    confidence = args.confidence if args.confidence is not None else float(mapping_cfg.get("confidence_threshold", 0.6))

    engine = OCREngine(
        lang=args.lang or ocr_cfg.get("lang", "en"),
        use_gpu=bool(args.use_gpu or ocr_cfg.get("use_gpu", False)),
        use_angle_cls=bool(ocr_cfg.get("use_angle_cls", True)),
        det_db_thresh=float(ocr_cfg.get("det_db_thresh", 0.3)),
    )

    ocr_results = engine.run(args.input, min_confidence=confidence, preprocess=args.preprocess)

    mapper = FieldMapper(config_path=args.config)
    mapper.conf_threshold = confidence
    form_data = mapper.map(ocr_results)

    output_path = resolve_output_path(args, cfg, output_format)

    if output_format == "json":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(form_data, indent=2), encoding="utf-8")
        print(f"Saved: {output_path}")
    elif output_format == "docx":
        filler = DOCXFiller(template=out_cfg.get("template_docx"))
        saved_path = filler.fill(form_data, str(output_path))
        print(f"Saved: {saved_path}")
    else:
        filler = PDFFiller(template=out_cfg.get("template_pdf"))
        saved_path = filler.fill(form_data, str(output_path))
        print(f"Saved: {saved_path}")

    if args.show_boxes or bool(out_cfg.get("save_debug_image", False)):
        debug_path = output_path.parent / "ocr_debug_boxes.jpg"
        engine.save_debug_image(args.input, ocr_results, str(debug_path), preprocess=args.preprocess)
        print(f"Debug image: {debug_path}")

    if engine.last_error:
        print(f"OCR warning: {engine.last_error}", file=sys.stderr)
    elif not ocr_results:
        print("OCR warning: no text regions were detected in the input file.", file=sys.stderr)
    elif not form_data:
        print("OCR warning: text was detected but no configured fields were mapped.", file=sys.stderr)

    print(json.dumps(form_data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
