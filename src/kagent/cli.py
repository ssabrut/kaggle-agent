import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated

import typer

from kagent import harness
from kagent.config import REPO_ROOT, CompetitionConfig, Settings
from kagent.docker import docker_available
from kagent.grader import metrics
from kagent.submit import SubmitRefused, submit
from kagent.workspace import Workspace

app = typer.Typer(no_args_is_help=True, help="Autonomous Kaggle agent.")

Competition = Annotated[str, typer.Option("--competition", "-c", help="Kaggle competition slug.")]


def _workspace(slug: str) -> Workspace:
    return Workspace.for_slug(Settings(), slug)


def _copy_contents(src: Path, dst: Path) -> None:
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _kaggle_download(slug: str, dest: Path) -> None:
    if shutil.which("kaggle") is None:
        raise typer.BadParameter("kaggle CLI not found; run via `uv run kagent ...`")
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            ["kaggle", "competitions", "download", "-c", slug, "-p", tmp],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise typer.BadParameter(
                f"kaggle download failed: {(proc.stderr or proc.stdout).strip()}\n"
                "A 403 usually means the competition rules are not accepted on kaggle.com yet."
            )
        for archive in Path(tmp).glob("*.zip"):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest)


@app.command()
def init(
    competition: Competition,
    metric: Annotated[str, typer.Option(help=f"One of: {', '.join(metrics.available())}")],
    id_column: Annotated[str, typer.Option(help="Id column in sample_submission.csv.")] = "id",
    data_dir: Annotated[
        Path | None, typer.Option(help="Copy public data from here instead of downloading.")
    ] = None,
    private_dir: Annotated[
        Path | None, typer.Option(help="Copy held-out labels.csv from here (grader only).")
    ] = None,
) -> None:
    """Create the workspace, fetch data, write config."""
    try:
        config = CompetitionConfig(slug=competition, metric=metric, id_column=id_column)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    ws = _workspace(competition)
    ws.create()
    if data_dir:
        _copy_contents(data_dir, ws.public)
    else:
        _kaggle_download(competition, ws.public)
    if private_dir:
        _copy_contents(private_dir, ws.private)
    ws.save_config(config)
    typer.echo(f"workspace ready at {ws.root}")


@app.command("build-images")
def build_images() -> None:
    """Build the sandbox and grader Docker images from docker/."""
    settings = Settings()
    if not docker_available():
        typer.echo("docker daemon not reachable; start Docker and retry", err=True)
        raise typer.Exit(1)
    for dockerfile, tag in (
        ("sandbox.Dockerfile", settings.sandbox_image),
        ("grader.Dockerfile", settings.grader_image),
    ):
        subprocess.run(
            ["docker", "build", "-f", f"docker/{dockerfile}", "-t", tag, "."],
            cwd=REPO_ROOT,
            check=True,
        )


@app.command()
def run(
    competition: Competition,
    script: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Script to run.")],
) -> None:
    """Run a script in the sandbox, check integrity, grade the submission."""
    record = harness.run_script(_workspace(competition), script, Settings())
    typer.echo(record.model_dump_json(indent=2))
    raise typer.Exit(0 if record.status == "ok" else 1)


@app.command("submit")
def submit_cmd(
    competition: Competition,
    run_id: Annotated[str, typer.Option("--run", help="Run id from `kagent run`.")],
    message: Annotated[str, typer.Option("--message", "-m")] = "kagent submission",
    approve: Annotated[
        bool, typer.Option("--approve", help="Really submit. Without it this is a dry run.")
    ] = False,
) -> None:
    """Dry-run a Kaggle submission; add --approve to send it for real."""
    try:
        result = submit(_workspace(competition), run_id, message, approve)
    except SubmitRefused as exc:
        typer.echo(f"refused: {exc}", err=True)
        raise typer.Exit(1) from exc
    if result.dry_run:
        typer.echo("dry run, would execute:\n  " + " ".join(result.command))
    else:
        typer.echo(result.output or "submitted")
