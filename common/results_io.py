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


def safe_path_part(name: str) -> str:
    """
    Turn a job title, candidate name or model tag into a single path
    component that is safe on disk, keeping it readable. Unlike
    safe_filename_part, spaces, dots and parentheses are kept.

    rewriter_v2.save_results names its folders and rewrite files with this,
    and rewriter_v2_judge finds those files again with it, so the writer and
    the reader cannot disagree on a name.
    """
    # "." is deliberately kept: it's safe mid-name on every OS we care
    # about (Windows/macOS/Linux) and this sanitizer is also applied to
    # agent_model, where version-numbered tags like "qwen2.5" or
    # "qwen2.5:14b" need their dot to survive - only the trailing-dot
    # case (stripped below) and characters like ":" are actually unsafe.
    keep = "-_(). "
    cleaned = "".join(c if (c.isalnum() or c in keep) else "_" for c in name)
    return cleaned.strip().strip(".") or "unnamed"


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
