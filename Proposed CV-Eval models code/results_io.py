"""
results_io.py

Small shared helpers for writing pipeline results to disk as JSON, with a
consistent, filesystem-safe naming scheme.
"""

import re
import json
from pathlib import Path


def safe_filename_part(text: str) -> str:
    """Turn arbitrary text (e.g. a job description's filename stem) into
    something safe to use inside a filename."""
    return re.sub(r"[^\w\-]+", "_", text)


def save_json(results, output_dir, filename: str) -> Path:
    """Write results to <output_dir>/<filename> as indented JSON, creating
    output_dir (and any parent folders) if needed. Returns the full path
    written, so callers can print/log it."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return filepath
