"""What a run reports while it happens.

The console app printed progress inline, which made it impossible to drive from
anything but a terminal. Runners now yield these instead, and the front end
decides how to render them.
"""

from dataclasses import dataclass, field

# "run" is the envelope event carrying the finished payload; "rank" is emitted
# by batch runs as the ordering firms up.
STAGES = ("parse", "council", "essential", "judge", "rewrite", "rank", "run")
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
