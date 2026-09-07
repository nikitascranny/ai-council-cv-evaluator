"""What the CV states, pulled out of the rewrite guard's extractors.

These functions were built to reject invented content. Read forwards instead of
backwards they answer a reviewer's question: what does this CV actually claim?
"""

from ui import facts


def test_numbers_are_listed_as_stated():
    got = facts.facts_of("Reduced latency by 40% across 3 services.")
    assert "40%" in got["numbers"]


def test_a_percentage_is_not_flattened_to_a_bare_number():
    """"72.8%" is a different claim from "72.8"."""
    got = facts.facts_of("Coverage rose to 72.8%.")
    assert "72.8%" in got["numbers"]
    assert "72.8" not in got["numbers"]


def test_organisations_come_through_as_capitalised_runs():
    # extract_caps_runs normalises through normalise(), which lowercases - by
    # design, so a rephrasing that only re-cases never reads as a new fact
    # (see rewriter_v1.normalise). That is right for the guard but wrong for
    # a reviewer, so facts_of recovers the CV's own casing before returning.
    got = facts.facts_of("Worked at Cape Town Software before joining Acme Corp.")
    assert any("Cape Town Software" in o for o in got["organisations"])


def test_a_qualification_keeps_the_casing_the_cv_used():
    """The guard normalises "BSc" to "bsc" to match case-insensitively; a
    reviewer reading "bsc" reasonably concludes the tool cannot read the
    document properly, so the panel must show the CV's own wording."""
    got = facts.facts_of("Holds a BSc in Computer Science.")
    assert "BSc" in got["qualifications"]
    assert "bsc" not in got["qualifications"]


def test_every_list_is_sorted_so_the_panel_is_stable():
    got = facts.facts_of("BSc from UCT, MSc from Wits, 5 years, 3 teams.")
    for key in ("numbers", "qualifications", "organisations"):
        assert got[key] == sorted(got[key])


def test_empty_cv_gives_empty_lists_not_an_error():
    assert facts.facts_of("") == {"numbers": [], "qualifications": [],
                                  "organisations": []}


def test_the_panel_escapes_cv_text():
    """CV text is not trusted markup."""
    out = facts.render_facts_panel("Led <script>alert(1)</script> 5 projects")
    assert "<script>" not in out
