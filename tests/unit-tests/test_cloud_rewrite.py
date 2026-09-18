"""The cloud rewriter against the real Ollama Cloud.

Deselected by default (pytest.ini's addopts), because it spends real usage
and needs OLLAMA_API_KEY. Run it with:

    python -m pytest -m cloud

The CV is made up and written here, so running this sends nobody's real CV
to ollama.com.
"""

import os

import pytest

import rewriter_v2

pytestmark = pytest.mark.cloud

_CV = """SAM EXAMPLE
Cape Town | sam.example@example.com

PROFILE
Office administrator with 4 years of experience in scheduling, filing and
front-desk support.

EXPERIENCE
Office Administrator, Harbour Logistics (Pty) Ltd - 2021 to 2025
- Managed calendars for 6 managers and booked 30 meetings a week.
- Kept the filing system for 1,200 client records.
- Answered phones and welcomed visitors at reception.

Receptionist, Bayview Dental - 2019 to 2021
- Booked patient appointments and handled payments.

EDUCATION
National Senior Certificate (Matric), 2018

SKILLS
Microsoft Word, Excel, Outlook; typing 55 words per minute.
"""


@pytest.fixture(autouse=True)
def _needs_a_key():
    if not os.environ.get(rewriter_v2.CLOUD_API_KEY_ENV, "").strip():
        pytest.skip("OLLAMA_API_KEY is not set")


def test_both_cloud_models_rewrite_a_cv_without_falling_back():
    results = rewriter_v2.rewrite_cv(_CV, "Secretary", "Sam Example",
                                     profile=rewriter_v2.CLOUD)

    for result in results:
        print(f"{result.agent_model}: {result.tokens_in} tokens in, "
              f"{result.tokens_out} out, retried={result.retried}, "
              f"rejected={result.rejected}, reasons={result.reasons}")

    # A stand-in would mean the cloud failed - say why rather than just "False".
    assert [r.replaced_cloud_model for r in results] == [None, None], \
        [r.cloud_error for r in results]
    assert [r.agent_model for r in results] == [c for c, _ in rewriter_v2.CLOUD_AGENTS]
    for result in results:
        assert result.provider == rewriter_v2.CLOUD
        assert result.error is None
        assert result.rewritten_text.strip()
        assert result.tokens_in and result.tokens_out
