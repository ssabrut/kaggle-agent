import subprocess

from kagent import images
from kagent.deps.lockfile import lock_sha256


def test_build_passes_the_lock_hash_and_builds_from_the_repo_root(settings, monkeypatch):
    seen = []
    monkeypatch.setattr(images.subprocess, "run", lambda argv, **kw: seen.append((argv, kw)))

    spec = images.specs(settings)[0]
    images.build_image(spec, settings)

    argv, kwargs = seen[0]
    digest = lock_sha256(settings.docker_dir / "sandbox.lock.txt")
    assert f"LOCK_SHA256={digest}" in argv
    assert argv[:2] == ["docker", "build"] and argv[-2:] == ["kagent-sandbox:latest", "."]
    assert kwargs["cwd"] == images.REPO_ROOT and kwargs["check"] is True


def test_image_with_the_current_lock_is_not_stale(settings, monkeypatch):
    current = {s.tag: lock_sha256(settings.docker_dir / s.lock) for s in images.specs(settings)}
    monkeypatch.setattr(images, "image_lock_label", lambda tag: current[tag])
    assert images.stale_images(settings) == []


def test_image_built_from_an_older_lock_or_missing_is_stale(settings, monkeypatch):
    labels = {"kagent-sandbox:latest": "old-hash", "kagent-grader:latest": None}
    monkeypatch.setattr(images, "image_lock_label", labels.get)
    assert images.stale_images(settings) == ["sandbox", "grader"]


def test_label_reader_handles_docker_output(monkeypatch):
    def fake(stdout, code=0):
        return lambda argv, **kw: subprocess.CompletedProcess(argv, code, stdout=stdout)

    monkeypatch.setattr(images.subprocess, "run", fake("abc123\n"))
    assert images.image_lock_label("img") == "abc123"
    monkeypatch.setattr(images.subprocess, "run", fake("<no value>\n"))
    assert images.image_lock_label("img") is None
    monkeypatch.setattr(images.subprocess, "run", fake("", code=1))
    assert images.image_lock_label("img") is None
