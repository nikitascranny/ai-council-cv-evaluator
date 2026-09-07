"""Which PDFs the app offers, on the layout main actually has.

Both corpora are committed under data/: job listings in data/job-listings,
CVs in data/cvs/<batch>. The CVs were local-only until 2026-09-06, when they
were committed to data/cvs, so every clone now has them and the app needs no
local setup.

An absent corpus is still reported rather than raised. A partial checkout is
possible, and a page that explains itself beats an empty dropdown.
"""

import re
from pathlib import Path

# ui/pipeline/<module>.py -> ui/pipeline -> ui -> repo root.
ROOT = Path(__file__).resolve().parent.parent.parent
JOB_SUBDIR = Path("data") / "job-listings"
CV_SUBDIRS = (Path("data") / "cvs",)

# Folders whose name adds nothing to a label, because every file is in one.
# The batch folders under data/cvs are NOT in here on purpose: batch3 holds
# bad_ and professional_ versions of the same three people, so the folder has
# to appear or entries read identically.
_FLAT_FOLDERS = {"cvs", "job-listings"}


def _pdfs_under(*dirs) -> list:
    found = []
    for directory in dirs:
        if directory.is_dir():
            found.extend(sorted(directory.rglob("*.pdf")))
    return found


def discover_jobs(root: Path = ROOT) -> list:
    return _pdfs_under(Path(root) / JOB_SUBDIR)


def discover_cvs(root: Path = ROOT) -> list:
    return _pdfs_under(*(Path(root) / sub for sub in CV_SUBDIRS))


def corpus_status(root: Path = ROOT) -> dict:
    """Enough to render an honest empty state."""
    root = Path(root)
    return {"jobs": len(discover_jobs(root)),
            "cvs": len(discover_cvs(root)),
            "cv_dir_present": any((root / sub).is_dir() for sub in CV_SUBDIRS),
            "cv_dirs": [str(sub) for sub in CV_SUBDIRS]}


def label_for(path) -> str:
    path = Path(path)
    parent = path.parent.name
    if parent in _FLAT_FOLDERS:
        return path.stem
    return "{}/{}".format(parent, path.stem)


def title_from_slug(stem: str) -> str:
    """job_listing_startup_fullstack_engineer -> "Startup Fullstack Engineer"."""
    return re.sub(r"^job[_ ]listing[_ ]", "", stem).replace("_", " ").strip().title()
