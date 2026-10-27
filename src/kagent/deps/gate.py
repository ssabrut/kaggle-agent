"""What the search loop sees of the dependency flow: ask for a package, then wait for a human.

The agent never installs anything. A run that fails on a missing import becomes a request with
automatic pre-checks; the node is parked while other branches keep running.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from kagent.config import Settings
from kagent.deps.detect import missing_module, needs_package
from kagent.deps.lockfile import parse_lock
from kagent.deps.models import DependencyRequest
from kagent.deps.prechecks import IndexClient, PyPIClient, Resolver, run_prechecks, uv_resolver
from kagent.deps.store import DependencyStore


class DepGate:
    def __init__(
        self,
        store: DependencyStore,
        settings: Settings,
        *,
        index: IndexClient | None = None,
        resolver: Resolver | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.store = store
        self.settings = settings
        self.index = index or PyPIClient()
        self.resolver = resolver or uv_resolver(settings.dep_cooldown)
        self._now = now
        self._created = 0  # requests made this session, capped by max_dep_requests

    def _docker(self, name: str) -> Path:
        return self.settings.docker_dir / name

    def installed(self) -> set[str]:
        return set(parse_lock(self._docker("sandbox.lock.txt").read_text()))

    def request_for_failure(self, node_id: str, stderr: str) -> DependencyRequest | None:
        """A request (new or existing) when the failure is a missing package, else None."""
        package = needs_package(stderr, self.installed())
        if package is None:
            return None

        # Already asked: reuse a pending one, replay a denial; an approved package that is
        # still missing means the mapping or the image is wrong, which a human should look at.
        for existing in self.store.find(package):
            if existing.status in ("pending", "denied"):
                return existing
            if existing.status == "approved":
                return None
        if self._created >= self.settings.max_dep_requests:
            self.store.log("request_cap_reached", package=package, node=node_id)
            return None

        module = missing_module(stderr)
        request = DependencyRequest(
            id=self.store.next_id(),
            package=package,
            reason=f"ModuleNotFoundError: No module named '{module}'",
            node_id=node_id,
            created_at=self._now().isoformat(),
            precheck=run_prechecks(
                package,
                "",
                index=self.index,
                resolver=self.resolver,
                sandbox_in_text=self._docker("sandbox.in").read_text(),
                installed=self.installed(),
                now=self._now(),
            ),
        )
        self._created += 1
        self.store.save(request)
        self.store.log(
            "requested", request=request.id, package=package, node=node_id,
            flags=request.precheck.flags(),
        )  # fmt: skip

        if request.precheck.checked and not request.precheck.exists_on_index:
            request.status = "denied"
            request.decision_reason = "not found on the package index (a hallucinated name?)"
            request.decided_at = self._now().isoformat()
            self.store.save(request)
            self.store.log("auto_denied", request=request.id, package=package)
        return request

    def current(self, request_id: str) -> DependencyRequest:
        """The request as it stands on disk, including a human's decision since the last look."""
        self.store.reload()
        return self.store.get(request_id)

    def is_stale(self, request: DependencyRequest) -> bool:
        created = datetime.fromisoformat(request.created_at)
        return (self._now() - created).total_seconds() > self.settings.dep_request_timeout_s

    def expire(self, request: DependencyRequest) -> DependencyRequest:
        request.status = "expired"
        request.decided_at = self._now().isoformat()
        self.store.save(request)
        self.store.log("expired", request=request.id, package=request.package)
        return request

    def pending(self) -> list[DependencyRequest]:
        self.store.reload()
        return self.store.all("pending")


def denial_message(request: DependencyRequest, error: str) -> str:
    """Tell the next debug step what happened, and that it must avoid the package."""
    why = request.decision_reason or "no reason given"
    return (
        f"{error.rstrip()}\n\nThe package '{request.package}' is not installed and its request "
        f"was denied ({why}). Rewrite the script without it; use only the installed libraries."
    )
