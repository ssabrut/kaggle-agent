"""Result type shared by the grader container and the host. Stdlib only."""

from dataclasses import dataclass, field


@dataclass
class GradeResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    # Held-out score. None when the submission is invalid or no private labels exist.
    # Evaluation only: the search loop must never select on this.
    score: float | None = None
    metric: str | None = None
    # Cross-validation on train, computed here from the script's out-of-fold predictions.
    # None when oof.csv is absent or invalid (see cv_errors).
    cv_score: float | None = None
    cv_std: float | None = None
    cv_errors: list[str] = field(default_factory=list)
