"""What a run reports while it happens.

The console app printed progress inline, which made it impossible to drive from
anything but a terminal. Runners now yield these instead, and the front end
decides how to render them.
"""

from dataclasses import dataclass, field

# "run" is the envelope event carrying the finished payload; "rank" is emitted
# by batch runs as the ordering firms up. "rewrite_v2" is candidate mode's
# rewrite - distinct from "rewrite", which is reviewer mode's
# phrasing-sensitivity probe and measures something else entirely.
# "feedback" is rewriter_v2_judge's verdict on those rewrites, distinct again
# from "judge", which scores a CV.
STAGES = ("parse", "council", "essential", "judge", "rewrite", "rewrite_v2",
          "feedback", "rank", "run")
STATES = ("start", "progress", "done", "failed")


@dataclass(frozen=True)
class Event:
    stage: str
    state: str
    detail: dict = field(default_factory=dict)
    secs: float = 0.0
    candidate: str = ""

    def __post_init__(self):
        if self.stage not in STAGES:
            raise ValueError("unknown stage: {}".format(self.stage))
        if self.state not in STATES:
            raise ValueError("unknown state: {}".format(self.state))

    def to_dict(self) -> dict:
        return {"stage": self.stage, "state": self.state,
                "detail": self.detail, "secs": round(self.secs, 1),
                "candidate": self.candidate}
