import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated

import typer

from kagent import harness
from kagent.config import REPO_ROOT, CompetitionConfig, Settings
from kagent.docker import DockerError, docker_available
from kagent.eda import write_profile
from kagent.folds import make_folds
from kagent.grader import metrics
from kagent.llm.client import ChatClient, LLMError
from kagent.search.loop import IntegrityAbort, OperatorFailures, open_tree, run_search
from kagent.search.operators import DebugOperator, DraftOperator, ImproveOperator
from kagent.search.policies import GreedyPolicy
from kagent.search.select import CHOICES, select_best, selected_run_id
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
    description: Annotated[
        Path | None, typer.Option(help="Text or markdown file describing the competition.")
    ] = None,
) -> None:
    """Create the workspace, fetch data, write config and CV folds."""
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
    if description:
        shutil.copyfile(description, ws.description_path)
    ws.save_config(config)
    if make_folds(ws, config):
        typer.echo("wrote data/public/folds.csv (5 folds)")
    else:
        typer.echo("no train.csv found, folds not written", err=True)
    typer.echo(f"workspace ready at {ws.root}")


@app.command()
def eda(competition: Competition) -> None:
    """Profile the data and write eda.md (used in prompts)."""
    ws = _workspace(competition)
    typer.echo(write_profile(ws, ws.load_config()))


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
    try:
        record = harness.run_script(_workspace(competition), script, Settings())
    except DockerError as exc:
        typer.echo(f"docker failed, the script did not run: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(record.model_dump_json(indent=2))
    raise typer.Exit(0 if record.status == "ok" else 1)


@app.command()
def search(
    competition: Competition,
    policy: Annotated[str, typer.Option(help="Search policy (only 'greedy' so far).")] = "greedy",
    budget_hours: Annotated[float, typer.Option(help="Wall-clock budget for this session.")] = 6.0,
    max_nodes: Annotated[int | None, typer.Option(help="Stop after this many new nodes.")] = None,
    num_drafts: Annotated[int, typer.Option(help="Independent drafts before improving.")] = 3,
    max_debug: Annotated[int, typer.Option(help="Debug attempts before a node is dead.")] = 3,
) -> None:
    """Run the draft/debug/improve loop. Resumes an existing tree in the workspace."""
    if policy != "greedy":
        raise typer.BadParameter("only 'greedy' is implemented")
    settings = Settings()
    if not docker_available():
        typer.echo("docker daemon not reachable; start Docker and retry", err=True)
        raise typer.Exit(2)
    ws = Workspace.for_slug(settings, competition)
    config = ws.load_config()
    if not (ws.public / "folds.csv").exists():
        raise typer.BadParameter("folds.csv missing; run `kagent init` with data first")

    description = ws.description_path.read_text() if ws.description_path.exists() else ""
    eda_text = ws.eda_path.read_text() if ws.eda_path.exists() else write_profile(ws, config)

    trajectory = ws.search_dir / "trajectory.jsonl"
    planner = ChatClient(
        settings.planner_base_url,
        settings.planner_model,
        role="planner",
        timeout_s=settings.llm_timeout_s,
        max_tokens=settings.llm_max_tokens,
        temperature=0.7,
        log_path=trajectory,
    )
    coder = ChatClient(
        settings.coder_base_url,
        settings.coder_model,
        role="coder",
        timeout_s=settings.llm_timeout_s,
        max_tokens=settings.llm_max_tokens,
        temperature=0.2,
        log_path=trajectory,
    )
    operators = {
        "draft": DraftOperator(planner, coder),
        "improve": ImproveOperator(planner, coder),
        "debug": DebugOperator(coder),
    }
    tree = open_tree(ws, config)
    try:
        run_search(
            ws,
            config,
            settings,
            tree,
            GreedyPolicy(num_drafts=num_drafts),
            operators,
            planner,
            description=description,
            eda=eda_text,
            budget_s=budget_hours * 3600,
            max_nodes=max_nodes,
            max_debug=max_debug,
            log=typer.echo,
        )
    except (LLMError, IntegrityAbort, OperatorFailures, DockerError) as exc:
        typer.echo(f"search stopped: {exc}", err=True)
        if isinstance(exc, DockerError):
            typer.echo("no node was recorded for that step; rerun to resume", err=True)
        raise typer.Exit(1) from exc

    best = tree.best(robust=True)
    if best:
        typer.echo(f"best so far: {best.id} cv={best.cv_score:.5f} +/- {best.cv_std or 0:.5f}")
    else:
        typer.echo("no node has a cross-validation score yet", err=True)


@app.command()
def select(
    competition: Competition,
    by: Annotated[str, typer.Option(help=f"Selection rule: {', '.join(CHOICES)}")] = "robust_cv",
) -> None:
    """Pick the best node by cross-validation and record it for `submit`."""
    ws = _workspace(competition)
    try:
        node = select_best(ws, open_tree(ws, ws.load_config()), by)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"selected {node.id} (run {node.run_id}) cv={node.cv_score:.5f} by {by}")


@app.command("submit")
def submit_cmd(
    competition: Competition,
    run_id: Annotated[
        str | None, typer.Option("--run", help="Run id; defaults to the selected node's run.")
    ] = None,
    message: Annotated[str, typer.Option("--message", "-m")] = "kagent submission",
    approve: Annotated[
        bool, typer.Option("--approve", help="Really submit. Without it this is a dry run.")
    ] = False,
) -> None:
    """Dry-run a Kaggle submission; add --approve to send it for real."""
    ws = _workspace(competition)
    try:
        result = submit(ws, run_id or selected_run_id(ws), message, approve)
    except (SubmitRefused, FileNotFoundError) as exc:
        typer.echo(f"refused: {exc}", err=True)
        raise typer.Exit(1) from exc
    if result.dry_run:
        typer.echo("dry run, would execute:\n  " + " ".join(result.command))
    else:
        typer.echo(result.output or "submitted")
