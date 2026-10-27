"""Build the sandbox and grader images from their hash-locked requirements."""

import subprocess
from dataclasses import dataclass

from kagent.config import REPO_ROOT, Settings
from kagent.deps.lockfile import lock_sha256

LABEL = "kagent.lock-sha256"


@dataclass(frozen=True)
class ImageSpec:
    kind: str
    dockerfile: str
    lock: str
    tag: str


def specs(settings: Settings) -> list[ImageSpec]:
    return [
        ImageSpec("sandbox", "sandbox.Dockerfile", "sandbox.lock.txt", settings.sandbox_image),
        ImageSpec("grader", "grader.Dockerfile", "grader.lock.txt", settings.grader_image),
    ]


def build_image(spec: ImageSpec, settings: Settings) -> None:
    """Build from the repo root; the lock file's hash is stored on the image as a label."""
    digest = lock_sha256(settings.docker_dir / spec.lock)
    subprocess.run(
        [
            "docker", "build",
            "-f", str(settings.docker_dir / spec.dockerfile),
            "--build-arg", f"LOCK_SHA256={digest}",
            "-t", spec.tag, ".",
        ],
        cwd=REPO_ROOT,
        check=True,
    )  # fmt: skip


def image_lock_label(tag: str) -> str | None:
    proc = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            f'{{{{ index .Config.Labels "{LABEL}" }}}}',
            tag,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    value = proc.stdout.strip()
    return value if proc.returncode == 0 and value and value != "<no value>" else None


def stale_images(settings: Settings) -> list[str]:
    """Images that are missing or were built from a different lock than the one on disk."""
    stale = []
    for spec in specs(settings):
        current = lock_sha256(settings.docker_dir / spec.lock)
        if image_lock_label(spec.tag) != current:
            stale.append(spec.kind)
    return stale
