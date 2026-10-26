"""Result type shared by the grader container and the host. Stdlib only."""

from dataclasses import dataclass, field


@dataclass
class GradeResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    # None when the submission is invalid or no private labels exist (live competition).
    score: float | None = None
    metric: str | None = None
