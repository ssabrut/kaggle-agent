"""Read and edit the hash-locked requirement files that define the sandbox environment."""

import hashlib
import re
from pathlib import Path

from packaging.markers import Marker
from packaging.requirements import Requirement

# The sandbox is linux on Python 3.11; evaluate the lock's environment markers for that.
SANDBOX_ENV = {
    "python_full_version": "3.11.0",
    "python_version": "3.11",
    "sys_platform": "linux",
    "platform_system": "Linux",
    "os_name": "posix",
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
}

_PINNED = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)\s*(?:;\s*(.*?))?\s*\\?\s*$")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(text: str) -> dict[str, str]:
    """Map normalized package name -> version for the sandbox environment."""
    pins: dict[str, str] = {}
    for line in text.splitlines():
        match = _PINNED.match(line)
        if not match:
            continue
        name, version, marker = match.groups()
        if marker and not Marker(marker).evaluate(SANDBOX_ENV):
            continue
        pins[normalize(name)] = version
    return pins


def requirement_lines(in_text: str) -> list[str]:
    return [
        line.strip() for line in in_text.splitlines() if line.strip() and not line.startswith("#")
    ]


def requirement_names(in_text: str) -> list[str]:
    return [normalize(Requirement(line).name) for line in requirement_lines(in_text)]


def add_requirement(in_text: str, package: str, constraint: str) -> str:
    """Add `package<constraint>` to the .in text, replacing any existing line for it."""
    wanted = normalize(package)
    new_line = f"{package}{constraint}".strip()
    kept = [
        line
        for line in in_text.splitlines()
        if not line.strip() or line.startswith("#") or normalize(Requirement(line).name) != wanted
    ]
    return "\n".join([*kept, new_line]).rstrip("\n") + "\n"


def sandbox_versions(docker_dir: Path) -> dict[str, str]:
    """Top-level sandbox packages with the versions the lock installs."""
    pins = parse_lock((docker_dir / "sandbox.lock.txt").read_text())
    names = requirement_names((docker_dir / "sandbox.in").read_text())
    return {name: pins[name] for name in names if name in pins}


def lock_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
