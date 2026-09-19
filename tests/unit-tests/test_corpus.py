"""Finding the PDFs on the committed repo layout.

Both corpora live under data/: job listings in data/job-listings, CVs in
data/cvs/<batch>. Both are committed, so every clone has them and no local
setup is needed. An absent corpus is still
reported rather than raised - a shallow or partial checkout is possible.
"""

from ui.pipeline import corpus


def _make_corpus(tmp_path, jobs=(), batches=()):
    """batches is (folder, filename) pairs under data/cvs/, as committed."""
    (tmp_path / "data" / "job-listings").mkdir(parents=True)
    for name in jobs:
        (tmp_path / "data" / "job-listings" / name).write_bytes(b"%PDF-")
    for folder, name in batches:
        d = tmp_path / "data" / "cvs" / folder
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(b"%PDF-")
    return tmp_path


def test_jobs_come_from_the_committed_data_directory(tmp_path):
    root = _make_corpus(tmp_path, jobs=["job_listing_secretary.pdf"])
    assert [p.name for p in corpus.discover_jobs(root)] == ["job_listing_secretary.pdf"]


def test_cvs_span_every_batch_folder(tmp_path):
    """data/cvs holds batch1 through batch4; all of them are candidates."""
    root = _make_corpus(tmp_path, batches=[("batch1", "ALEX PARK.pdf"),
                                           ("batch3", "bad_alex.pdf"),
                                           ("batch4", "Ralph_Farm_CV.pdf")])
    assert {p.name for p in corpus.discover_cvs(root)} == {
        "ALEX PARK.pdf", "bad_alex.pdf", "Ralph_Farm_CV.pdf"}


def test_missing_cv_corpus_is_reported_not_raised(tmp_path):
    """A fresh clone of main has job listings and no CVs."""
    root = _make_corpus(tmp_path, jobs=["job_listing_secretary.pdf"])
    assert corpus.discover_cvs(root) == []
    status = corpus.corpus_status(root)
    assert status["cv_dir_present"] is False
    assert status["jobs"] == 1 and status["cvs"] == 0


def test_label_keeps_the_batch_folder_so_duplicates_are_distinguishable(tmp_path):
    """batch3 holds bad_ and professional_ versions of the same three people."""
    root = _make_corpus(tmp_path, batches=[("batch1", "ALEX PARK.pdf"),
                                           ("batch3", "bad_alex.pdf")])
    labels = sorted(corpus.label_for(p) for p in corpus.discover_cvs(root))
    assert labels == ["batch1/ALEX PARK", "batch3/bad_alex"]


def test_job_listing_slug_becomes_a_title():
    assert corpus.title_from_slug(
        "job_listing_startup_fullstack_engineer") == "Startup Fullstack Engineer"
