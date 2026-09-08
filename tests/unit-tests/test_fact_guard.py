"""
Unit tests for common/fact_guard.py.

Grouped by function. Each group tries to cover: the happy path, the
specific false-positive cases mentioned in the module's own docstrings
(THE DANIEL's "Hons"->"Honours", ALEX PARK's trailing page number "2"),
and the documented one-directional / narrow-scope behaviour so a future
change that silently widens or narrows the guard gets caught here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import fact_guard as fg


# ---------------- normalise ----------------

class TestNormalise:
    def test_lowercases(self):
        assert fg.normalise("Python Developer") == "python developer"

    def test_strips_punctuation(self):
        assert fg.normalise("C++, Java; SQL!") == "c java sql"

    def test_collapses_whitespace(self):
        assert fg.normalise("a   b\t\nc") == "a b c"

    def test_keeps_percent_attached_to_number(self):
        assert "72.8%" in fg.normalise("Scored 72.8% overall")

    def test_keeps_decimal_point_between_digits(self):
        # "72.8" must stay one token, not become "72 8"
        result = fg.normalise("72.8")
        assert result == "72.8"

    def test_drops_sentence_final_period(self):
        # A period NOT between two digits is punctuation, not a decimal.
        result = fg.normalise("Managed the team.")
        assert "." not in result

    def test_ordinal_suffix_becomes_separate_word(self):
        # Documented in _new_against's docstring: extractor splits "3rd"
        # into fact "3", while normalise() keeps the surrounding text
        # intact for the substring-fallback check.
        assert "3rd" in fg.normalise("Came 3rd in the regional final")


# ---------------- extract_hard_facts / extract_caps_runs / extract_facts ----------------

class TestExtraction:
    def test_hard_facts_catches_numbers(self):
        facts = fg.extract_hard_facts("Increased sales by 42%")
        assert "42%" in facts

    def test_hard_facts_catches_qualifications(self):
        facts = fg.extract_hard_facts("Holds a BSc in Computer Science")
        assert "bsc" in facts

    def test_hard_facts_ignores_caps_runs(self):
        # "Cape Town University" is a multi-word caps run, not a number or
        # qualification token - extract_hard_facts must not pick it up.
        facts = fg.extract_hard_facts("Studied at Cape Town University")
        assert not any("cape" in f or "university" in f for f in facts)

    def test_caps_run_requires_two_or_more_words(self):
        # A single capitalised word starts nearly every sentence; the
        # module docstring is explicit that a lone one must not be
        # collected, or every rewrite would be flagged.
        single = fg.extract_caps_runs("Managed the Sales team")
        multi = fg.extract_caps_runs("Managed the Sales Department team")
        assert not any(f == "sales" for f in single)
        assert any("sales" in f and "department" in f for f in multi)

    def test_extract_facts_is_union_of_hard_and_caps(self):
        text = "Certified Software Test Analyst, scored 90%"
        hard = fg.extract_hard_facts(text)
        caps = fg.extract_caps_runs(text)
        allf = fg.extract_facts(text)
        assert hard <= allf
        assert caps <= allf

    def test_qualification_abbreviation_and_expansion_are_canonicalised(self):
        # The specific false-positive fixed after THE DANIEL.pdf: "Hons"
        # expanding to "Honours" must resolve to the SAME fact set, or an
        # honest rewrite gets rejected for "inventing" a qualification.
        # ("BCom" is itself a recognised qualification token, so both
        # sides also contain "bcom" - that's expected, not the thing
        # under test here.)
        hons = fg._extract("BCom Hons", (fg._QUALIFICATION,))
        honours = fg._extract("BCom Honours", (fg._QUALIFICATION,))
        assert hons == honours
        assert "honours" in hons

    def test_matriculation_canonicalises_to_matric(self):
        a = fg._extract("Matriculation certificate", (fg._QUALIFICATION,))
        b = fg._extract("Matric certificate", (fg._QUALIFICATION,))
        assert a == b
        assert "matric" in a


# ---------------- load_bearing_numbers ----------------

class TestLoadBearingNumbers:
    def test_excludes_standalone_page_number_line(self):
        # The ALEX PARK.pdf case from the docstring: a lone "2" on its
        # own line is a page number, not a fact worth rejecting over.
        text = "Managed compliance policies.\n2"
        numbers = fg.load_bearing_numbers(text)
        assert "2" not in numbers

    def test_includes_inline_number_even_if_short(self):
        # A single digit that's part of a sentence (not its own line)
        # still isn't load-bearing per the "< 2 chars" rule - only
        # numbers of length >= 2 count.
        text = "Led a team of 12 engineers"
        assert "12" in fg.load_bearing_numbers(text)

    def test_excludes_single_digit_inline(self):
        text = "Rated it a 5 out of ten"
        numbers = fg.load_bearing_numbers(text)
        assert "5" not in numbers

    def test_includes_years(self):
        text = "Employed there since 2015"
        assert "2015" in fg.load_bearing_numbers(text)

    def test_includes_percentages(self):
        text = "Improved throughput by 42%"
        assert "42%" in fg.load_bearing_numbers(text)


# ---------------- numbers_lost ----------------

class TestNumbersLost:
    def test_flags_dropped_year(self):
        original = "Worked at Acme Corp since 2015."
        rewritten = "Worked at Acme Corp."
        assert "2015" in fg.numbers_lost(original, rewritten)

    def test_does_not_flag_retained_number(self):
        original = "Managed a team of 12."
        rewritten = "Led a team of 12 people."
        assert fg.numbers_lost(original, rewritten) == []

    def test_does_not_flag_dropped_page_number(self):
        original = "Managed compliance policies.\n2"
        rewritten = "Managed compliance policies."
        assert fg.numbers_lost(original, rewritten) == []

    def test_reformatted_number_still_matches_via_substring_fallback(self):
        # "72.8%" re-punctuated as "72.8 %" should still be found in the
        # rewritten haystack via the substring fallback in _new_against.
        original = "Scored 72.8% on the assessment."
        rewritten = "Scored 72.8% on the assessment, a strong result."
        assert fg.numbers_lost(original, rewritten) == []


# ---------------- title_fabricated ----------------

class TestTitleFabricated:
    def test_true_when_title_appears_only_in_rewrite(self):
        original = "Assisted the IT department with helpdesk tickets."
        rewritten = "Worked as a Computer Repair Technician fixing hardware."
        assert fg.title_fabricated(original, rewritten, "Computer Repair Technician")

    def test_false_when_title_already_in_original(self):
        original = "Worked as a Computer Repair Technician for two years."
        rewritten = "Served as a Computer Repair Technician handling repairs."
        assert not fg.title_fabricated(original, rewritten, "Computer Repair Technician")

    def test_false_when_title_absent_from_both(self):
        original = "Assisted with general IT support."
        rewritten = "Provided general IT support to staff."
        assert not fg.title_fabricated(original, rewritten, "Computer Repair Technician")

    def test_false_for_empty_job_title(self):
        assert not fg.title_fabricated("anything", "anything else", "")


# ---------------- check_no_new_facts ----------------

class TestCheckNoNewFacts:
    def test_empty_when_pure_rephrasing(self):
        original = "Assisted with support tickets. Holds a BSc."
        rewritten = "Provided assistance with support tickets. Holds a BSc."
        assert fg.check_no_new_facts(original, rewritten) == []

    def test_flags_invented_number(self):
        original = "Assisted with support tickets."
        rewritten = "Resolved 500 support tickets."
        assert "500" in fg.check_no_new_facts(original, rewritten)

    def test_flags_invented_qualification(self):
        original = "Studied computer science."
        rewritten = "Holds a BSc in computer science."
        assert "bsc" in fg.check_no_new_facts(original, rewritten)

    def test_hons_expansion_is_not_flagged(self):
        # Regression test for the exact bug the canon map fixes.
        original = "BCom Hons in Finance."
        rewritten = "BCom Honours in Finance."
        assert fg.check_no_new_facts(original, rewritten) == []

    def test_recased_caps_run_is_not_flagged(self):
        # Caps runs are advisory-only; check_no_new_facts must not reject
        # on a re-cased or re-worded organisation/title alone.
        original = "certified software test analyst"
        rewritten = "Certified Software Test Analyst"
        assert fg.check_no_new_facts(original, rewritten) == []

    def test_does_not_flag_dropped_facts(self):
        # One-directional: losing a fact is not this function's job.
        original = "Resolved 500 tickets. Holds a BSc."
        rewritten = "Resolved tickets."
        assert fg.check_no_new_facts(original, rewritten) == []


# ---------------- advisory_caps_runs ----------------

class TestAdvisoryCapsRuns:
    def test_flags_new_multiword_entity(self):
        original = "Worked in retail."
        rewritten = "Previously at Google Cloud Platform."
        result = fg.advisory_caps_runs(original, rewritten)
        assert any("google" in r for r in result)

    def test_does_not_flag_entity_present_in_original(self):
        original = "Worked at Google Cloud Platform as an intern."
        rewritten = "Interned at Google Cloud Platform."
        assert fg.advisory_caps_runs(original, rewritten) == []


# ---------------- rejection_reasons (integration of all three checks) ----------------

class TestRejectionReasons:
    def test_no_reasons_for_clean_rewrite(self):
        original = "Assisted with support tickets. Employed since 2015."
        rewritten = "Provided assistance with support tickets since 2015."
        assert fg.rejection_reasons(original, rewritten, "Support Technician") == []

    def test_combines_new_fact_and_title_claim(self):
        original = "Assisted with support tickets."
        rewritten = "Worked as a Support Technician, resolving 500 tickets."
        reasons = fg.rejection_reasons(original, rewritten, "Support Technician")
        assert any("500" in r for r in reasons)
        assert any("job title claimed but not held" in r for r in reasons)

    def test_flags_dropped_number(self):
        original = "Managed a team of 12 in 2019."
        rewritten = "Managed a team in 2019."
        reasons = fg.rejection_reasons(original, rewritten, "Team Lead")
        assert any("number dropped from the CV" in r and "12" in r for r in reasons)

    def test_title_already_held_is_not_a_reason(self):
        original = "Worked as a Team Lead managing 12 staff in 2019."
        rewritten = "Served as Team Lead, managing 12 staff since 2019."
        assert fg.rejection_reasons(original, rewritten, "Team Lead") == []
