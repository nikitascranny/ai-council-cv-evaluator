"""Candidate mode's seam onto rewriter v2.

v2 is another team member's work: it takes a CV and a job title and produces a
.docx aimed at that listing. It does not exist yet, so this module raises
NotAvailable and the UI says so plainly rather than shipping a stub that looks
like it worked.

When v2 lands, set BACKEND to a callable taking (cv_path, job_title) and
returning a path to the generated document. Nothing else in the app changes -
including if the input format later becomes .docx rather than PDF.
"""

from dataclasses import dataclass, field

BACKEND = None


class NotAvailable(RuntimeError):
    """Rewriter v2 is not wired up."""


@dataclass
class Result:
    path: str
    notes: list = field(default_factory=list)


def generate(cv_path, job_title: str, backend=...) -> Result:
    backend = BACKEND if backend is ... else backend
    if backend is None:
        raise NotAvailable(
            "rewriter v2 is not available yet. Candidate mode generates the "
            "improved CV, and that component is still being built.")
    path = backend(cv_path, job_title)
    if not path:
        raise NotAvailable("rewriter v2 returned no document.")
    return Result(path=path)
