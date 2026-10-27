"""Human decisions on dependency requests, and the install that follows an approval.

Approval edits sandbox.in, recompiles the hash-locked sandbox.lock.txt under the release-age
cooldown, and rebuilds the sandbox image from it. If any step fails, both files are restored
and the request stays pending. Nothing here runs inside the sandbox.
"""

import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from kagent.config import REPO_ROOT, Settings
from kagent.deps.lockfile import add_requirement, normalize, parse_lock
from kagent.deps.models import DependencyRequest
from kagent.deps.store import DependencyStore
from kagent.images import ImageSpec, build_image, specs

CompileFn = Callable[..., None]
BuildFn = Callable[[ImageSpec, Settings], None]


class InstallError(RuntimeError):
    """Approval could not be carried out; the environment is unchanged."""


def compile_lock(
    in_path: Path,
    out_path: Path,
    *,
    cooldown: str,
    cooldown_exempt: str | None = None,
    python_version: str = "3.11",
) -> None:
    """Recompile the lock with hashes. Existing pins in out_path are kept where possible."""
    command = [
        "uv", "pip", "compile", _rel(in_path), "--universal", "--generate-hashes",
        "--exclude-newer", cooldown, "--python-version", python_version, "-o", _rel(out_path),
        "--quiet",
    ]  # fmt: skip
    if cooldown_exempt:
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        command += ["--exclude-newer-package", f"{cooldown_exempt}={now}"]
    proc = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise InstallError((proc.stderr or proc.stdout).strip()[-600:])


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _decide(store: DependencyStore, request_id: str) -> DependencyRequest:
    request = store.get(request_id)
    if request.status != "pending":
        raise InstallError(f"{request_id} is already {request.status}")
    return request


def deny_request(store: DependencyStore, request_id: str, reason: str) -> DependencyRequest:
    request = _decide(store, request_id)
    request.status, request.decision_reason = "denied", reason
    request.decided_at = datetime.now(UTC).isoformat()
    store.save(request)
    store.log("denied", request=request_id, package=request.package, reason=reason)
    return request


def approve_request(
    store: DependencyStore,
    request_id: str,
    settings: Settings,
    *,
    skip_cooldown: bool = False,
    compile_fn: CompileFn = compile_lock,
    build_fn: BuildFn = build_image,
) -> DependencyRequest:
    request = _decide(store, request_id)
    in_path = settings.docker_dir / "sandbox.in"
    lock_path = settings.docker_dir / "sandbox.lock.txt"
    old_in, old_lock = in_path.read_text(), lock_path.read_text()

    try:
        in_path.write_text(add_requirement(old_in, request.package, request.constraint))
        compile_fn(
            in_path,
            lock_path,
            cooldown=settings.dep_cooldown,
            cooldown_exempt=request.package if skip_cooldown else None,
        )
        version = parse_lock(lock_path.read_text()).get(normalize(request.package))
        if version is None:
            raise InstallError(f"'{request.package}' is missing from the lock after resolving")
        sandbox = next(spec for spec in specs(settings) if spec.kind == "sandbox")
        build_fn(sandbox, settings)
    except (InstallError, subprocess.CalledProcessError, OSError) as exc:
        in_path.write_text(old_in)
        lock_path.write_text(old_lock)
        store.log("install_failed", request=request_id, package=request.package, error=str(exc))
        raise InstallError(f"{exc}") from exc

    request.status, request.resolved_version = "approved", version
    request.decided_at = datetime.now(UTC).isoformat()
    store.save(request)
    store.log(
        "approved",
        request=request_id,
        package=request.package,
        version=version,
        cooldown_skipped=skip_cooldown,
    )
    return request
