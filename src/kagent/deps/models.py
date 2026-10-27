"""Dependency requests: what the agent asks for and what the pre-checks found."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["pending", "approved", "denied", "expired"]


class PreCheck(BaseModel):
    """Facts gathered automatically before a human sees the request."""

    exists_on_index: bool = False
    age_days: int | None = None  # days since the first upload
    latest_release_age_days: int | None = None
    near_miss_of: str | None = None  # a well-known name this one is suspiciously close to
    vulns: list[str] = Field(default_factory=list)  # advisories on the latest release
    new_transitive: list[str] = Field(default_factory=list)  # packages a throwaway resolve adds
    resolved_version: str | None = None
    resolve_error: str | None = None  # why the throwaway resolve failed, if it did
    checked: bool = True  # False when the index could not be reached

    def flags(self) -> list[str]:
        """Short warnings for the approval screen."""
        out = []
        if not self.checked:
            out.append("PRE-CHECKS INCOMPLETE")
        elif not self.exists_on_index:
            out.append("NOT ON INDEX")
        if self.near_miss_of:
            out.append(f"NEAR-MISS OF {self.near_miss_of}")
        if self.age_days is not None and self.age_days < 90:
            out.append(f"NEW PACKAGE ({self.age_days}d)")
        if self.vulns:
            out.append(f"{len(self.vulns)} ADVISORIES")
        if self.new_transitive:
            out.append(f"+{len(self.new_transitive)} TRANSITIVE")
        if self.resolve_error:
            out.append("DOES NOT RESOLVE")
        return out


class DependencyRequest(BaseModel):
    id: str
    package: str
    constraint: str = ""  # a proposal, for example ">=1.4,<2"; the resolver decides
    reason: str  # the error class or task need
    node_id: str  # the search node waiting on it
    precheck: PreCheck = Field(default_factory=PreCheck)
    status: Status = "pending"
    decision_reason: str = ""
    resolved_version: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    decided_at: str | None = None

    @property
    def requirement(self) -> str:
        return f"{self.package}{self.constraint}"
