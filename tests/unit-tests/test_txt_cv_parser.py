"""txt_cv_parser: plain-text section segmentation for rewritten CVs.

Covers the markdown-emphasis bug found by running real rewriter output
through segment_text(): a bold or italic heading ("**Experience**",
"*Umoja Bank*") starts with the same character as a bullet marker
("* Managed the budget"), and the original bullet check could not tell them
apart - every asterisk-wrapped heading in a phi4 rewrite was read as a
bullet and folded into whatever section was already open, so
sections_named() came back empty for a CV that plainly had an Experience
and an Education section.
"""

from txt_cv_parser import (ParsedCV, _is_bullet, _unwrap_emphasis, is_heading_plain,
                           segment_text)


def test_a_real_bullet_is_still_a_bullet():
    assert _is_bullet("- Managed the budget")
    assert _is_bullet("* Managed the budget")
    assert _is_bullet("• Managed the budget")


def test_markdown_emphasis_is_not_a_bullet():
    assert not _is_bullet("**Experience**")
    assert not _is_bullet("*Umoja Bank*")


def test_unwrap_emphasis_strips_matching_wrapping_only():
    assert _unwrap_emphasis("**Experience**") == "Experience"
    assert _unwrap_emphasis("*Umoja Bank*") == "Umoja Bank"
    assert _unwrap_emphasis("Plain heading") == "Plain heading"
    # Asymmetric wrapping (an opening marker with no matching close) is not a
    # wrapper - stripping it would eat real content off one end only.
    assert _unwrap_emphasis("*Umoja Bank") == "*Umoja Bank"


def test_a_bold_heading_is_heading_shaped_but_a_real_bullet_is_not():
    assert is_heading_plain("**Experience**")
    assert is_heading_plain("*Skills*")
    assert not is_heading_plain("* Led the redesign project")
    assert not is_heading_plain("- Led the redesign project")


def test_segment_text_recognises_bold_wrapped_section_headings():
    """The GEOFF NITSCHKE regression: markdown-bold headings must open their
    own sections rather than being swallowed into whatever section preceded
    them."""
    rewrite = (
        "**Geoff Nitschke**\n"
        "Portfolio\n\n"
        "**Professional Summary**\n"
        "A short summary paragraph.\n\n"
        "**Experience**\n"
        "**Senior UX Researcher**\n"
        "*Umoja Bank*\n"
        "- Led comprehensive customer research.\n\n"
        "**Education**\n"
        "Bachelor of Arts, Cultural Anthropology\n"
    )
    cv = ParsedCV(path="rewrite.txt", sections=segment_text(rewrite),
                 extractor="plaintext")

    assert "experience" in [s.label for s in cv.sections]
    assert "education" in [s.label for s in cv.sections]

    scoped = cv.sections_named("experience", "education", "certifications")
    assert "Led comprehensive customer research" in scoped
    assert "Bachelor of Arts, Cultural Anthropology" in scoped
    # Excluded content must stay excluded - the fix should not widen scoping
    # into a "when in doubt, include everything" fallback.
    assert "A short summary paragraph" not in scoped


def test_a_bold_heading_stores_the_unwrapped_text():
    """The heading text becomes part of what is sent to a model - it should
    read "Experience", not "**Experience**"."""
    sections = segment_text("**Experience**\nFive years as a technician.\n")
    experience = next(s for s in sections if s.label == "experience")
    assert experience.heading == "Experience"
