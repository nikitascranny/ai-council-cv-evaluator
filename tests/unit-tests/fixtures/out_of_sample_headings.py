"""Section headings from CVs outside the Batch1 tuning corpus.

Batch1's ten CVs tuned every threshold in this parser, so they cannot show that
it generalises. These headings come from documents the lexicon had never seen.
Keep the strings, not the documents - the source CVs carry personal contact
details and do not belong in version control.
"""

OUT_OF_SAMPLE_HEADINGS = [
    ("PROFILE", "summary"),
    ("EDUCATION", "education"),
    ("TUTORING EXPERIENCE", "experience"),
    ("PROJECTS & TECHNICAL EXPERIENCE", "experience"),
    ("TECHNICAL SKILLS", "skills"),
    ("ADDITIONAL", None),
]
