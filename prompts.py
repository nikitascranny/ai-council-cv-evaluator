"""Versioned prompts.

Every vote records which version produced it. Changing the prompt and the context
window in the same run makes the result unattributable, so variants are explicit
and never edited in place - add a new one instead.

v1_plain    Reproduces the minimal schema of the original Batch1 runs. Used to
            isolate the effect of num_ctx: it is the ONLY variable that differs
            from the original results.
v2_anchored Adds explicit rubric anchors and forces an essentials check. Expected
            to fix score compression, but must be compared against v1 at the SAME
            num_ctx, never against the original truncated runs.
"""

SCHEMA = """Reply with ONLY a JSON object, no other text:
{"score": <int 1-10>, "strengths": [<string>, ...], "weaknesses": [<string>, ...],
 "summary": "<string>", "candidate": "<the candidate's full name, uppercase>"}"""

V1_PLAIN = """You are screening a CV against a job listing.

JOB LISTING:
{job}

CV:
{cv}

Score how well this candidate fits the job listing above, from 1 to 10.

""" + SCHEMA

V2_ANCHORED = """You are screening a CV against a job listing. Score fit to THIS
job only. A strong candidate for a different job is a weak candidate here.

JOB LISTING:
{job}

CV:
{cv}

First check the listing's Essential Requirements against the CV. Then score:

  1-2   Fails one or more essential requirements outright. No relevant evidence.
  3-4   Fails an essential, but shows transferable skills that could be trained up.
  5-6   Meets some essentials; clear gaps remain in others.
  7-8   Meets all essential requirements with evidence in the work history.
  9-10  Meets all essentials and several desirables, with strong evidence.

Score only what the CV evidences. A skill listed in a skills section but absent
from the work history is a claim, not evidence. Never credit the candidate with a
technology that does not appear in their CV.

Use the full range. Most candidates for a specialised role score 1-4.

Reminder - the job being scored is the one in the JOB LISTING above, not whatever
role this candidate currently holds.

""" + SCHEMA

CRITERIA_SCHEMA = """Reply with ONLY a JSON object, no other text:
{"required_skills": <1-5>, "evidence": <1-5>, "transferability": <1-5>,
 "findability": <1-5>, "customisation": <1-5>, "score": <int 1-10>,
 "strengths": [<string>, ...], "weaknesses": [<string>, ...],
 "summary": "<string>", "candidate": "<full name, uppercase>"}"""

# v3 scores the five criteria named in Description.txt separately instead of
# collapsing them into one number. A single 1-10 scale cannot separate ten
# candidates who all fail the same essential requirement - they correctly bunch
# at the bottom and the ranking vanishes. The criteria do separate them: on the
# aquarium job the reference scores NDLOVU evidence 5 / transferability 4 and
# LERATO 1 / 1, while their blended scores are only 4 and 1 apart.
# Still one CV against one job - no cross-candidate comparison.
V3_CRITERIA = """You are screening one CV against one job listing.

JOB LISTING:
{job}

CV:
{cv}

Rate this candidate on five separate criteria, each from 1 to 5.

1. required_skills - does the CV show the listing's Essential Requirements?
     1 none  2 marginal  3 partial  4 most  5 all, plus desirables

2. evidence - are the relevant skills backed by the work history, or only
   asserted in a skills list? Judge the QUALITY of evidence for whatever
   relevant skills exist, independently of how many there are. A skill listed
   but never used in any role scores 1 here no matter how impressive it sounds.
     1 claims only  3 some backing  5 specific, dated, with outcomes

3. transferability - if this person does not already meet the requirements,
   how plausibly could they be trained into the role? Look for skills whose
   SHAPE matches the work even if the domain differs: routine procedural
   logging, physical or repetitive work, following exact specifications,
   working inside the relevant kind of team.
     1 no usable foundation  3 partial  5 close, needs only domain training

4. findability - how fast could a screener locate the relevant information?
     1 buried in prose  3 workable  5 immediately scannable

5. customisation - how far does the CV's emphasis match THIS listing?
     1 written for a different role  5 reads as written for this one

Then give an overall score from 1 to 10 for fit to this specific job.

Score only what the CV evidences. Never credit a technology or qualification
that does not appear in the CV.

""" + CRITERIA_SCHEMA

VERSIONS = {"v1_plain": V1_PLAIN, "v2_anchored": V2_ANCHORED, "v3_criteria": V3_CRITERIA}


def build(version: str, job_text: str, cv_text: str) -> str:
    # Plain replacement, not str.format: the JSON schema below contains literal
    # braces that format() would try to interpret as fields.
    return (VERSIONS[version]
            .replace("{job}", job_text)
            .replace("{cv}", cv_text))
