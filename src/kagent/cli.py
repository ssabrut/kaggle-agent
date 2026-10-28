import shutil
import signal
import subprocess
import tempfile
import threading
import webbrowser
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from kagent import harness
from kagent.competition_page import (
    PageFetchError,
    PageInfo,
    fetch_pages,
    parse_overview,
    save_pages,
)
from kagent.config import CompetitionConfig, Settings
from kagent.dashboard.demo import DemoRun
from kagent.dashboard.launch import Launcher
from kagent.dashboard.reader import WorkspaceReader
from kagent.dashboard.server import is_loopback, make_server
from kagent.deps.gate import DepGate
from kagent.deps.install import InstallError, approve_request, deny_request
from kagent.deps.store import DependencyStore
from kagent.docker import DockerError, docker_available
from kagent.eda import write_profile
from kagent.folds import make_folds
from kagent.grader import metrics
from kagent.images import build_image, specs, stale_images
from kagent.llm.client import ChatClient, LLMError
from kagent.search.loop import IntegrityAbort, OperatorFailures, open_tree, run_search
from kagent.search.operators import DebugOperator, DraftOperator, ImproveOperator
from kagent.search.policies import GreedyPolicy
from kagent.search.progress import Progress
from kagent.search.select import CHOICES, select_best, selected_run_id
from kagent.submit import SubmitRefused, submit
from kagent.workspace import Workspace

app = typer.Typer(no_args_is_help=True, help="Autonomous Kaggle agent.")

Competition = Annotated[str, typer.Option("--competition", "-c", help="Kaggle competition slug.")]


def _workspace(slug: str) -> Workspace:
    return Workspace.for_slug(Settings(), slug)


def _interrupt(signum: int, frame: object) -> None:
    """Turn a stop request into the same clean shutdown as Ctrl-C."""
    raise KeyboardInterrupt


def _deps_dir(ws: Workspace) -> Path:
    return ws.root / "deps"


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


def _read_competition_page(ws: Workspace, slug: str, headless: bool) -> PageInfo | None:
    """Fetch and save the public pages. A failure is reported and init carries on."""
    try:
        fetched = fetch_pages(slug, headless=headless)
    except PageFetchError as exc:
        typer.echo(f"could not read the competition page: {exc}", err=True)
        return None
    info = parse_overview(fetched.overview_container, fetched.overview_body)
    save_pages(ws, fetched, info)
    typer.echo(f"page: {info.kind or 'competition kind not found'}")
    if info.tags:
        typer.echo(f"page tags: {', '.join(info.tags)}")
    if info.prize_warning:
        typer.echo(f"warning: {info.prize_warning}", err=True)
    typer.echo(
        f"saved {ws.page_dir}; read rules.txt before running, since some competitions "
        "restrict automated tools"
    )
    return info


@app.command()
def init(
    competition: Competition,
    metric: Annotated[
        str | None,
        typer.Option(
            help=f"One of: {', '.join(metrics.available())}. Needed unless --fetch-page finds it."
        ),
    ] = None,
    id_column: Annotated[
        str | None, typer.Option(help="Defaults to the first column of sample_submission.csv.")
    ] = None,
    data_dir: Annotated[
        Path | None, typer.Option(help="Copy public data from here instead of downloading.")
    ] = None,
    private_dir: Annotated[
        Path | None, typer.Option(help="Copy held-out labels.csv from here (grader only).")
    ] = None,
    description: Annotated[
        Path | None, typer.Option(help="Text or markdown file describing the competition.")
    ] = None,
    fetch_page: Annotated[
        bool,
        typer.Option("--fetch-page", help="Read the competition's public pages (opens a browser)."),
    ] = False,
    headless: Annotated[
        bool, typer.Option(help="Hide the browser window. Only the visible mode has been tried.")
    ] = False,
) -> None:
    """Create the workspace, fetch data, write config and CV folds."""
    if metric is not None and metric not in metrics.METRICS:
        raise typer.BadParameter(f"unknown metric '{metric}', choose from {metrics.available()}")

    ws = _workspace(competition)
    ws.create()
    if data_dir:
        _copy_contents(data_dir, ws.public)
    else:
        _kaggle_download(competition, ws.public)
    if private_dir:
        _copy_contents(private_dir, ws.private)

    info = _read_competition_page(ws, competition, headless) if fetch_page else None
    chosen = metric or (info.metric if info else None)
    if chosen is None:
        tags = f" (page tags: {', '.join(info.tags)})" if info and info.tags else ""
        raise typer.BadParameter(
            f"no metric given and none could be read from the page{tags}; "
            f"pass --metric, one of {metrics.available()}"
        )
    if metric is None and info is not None:
        typer.echo(f"metric from page: {chosen} (tag '{info.metric_text}')")

    try:
        config = CompetitionConfig(
            slug=competition, metric=chosen, id_column=id_column or ws.infer_id_column()
        )
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if description:
        shutil.copyfile(description, ws.description_path)
    elif info is not None:
        ws.description_path.write_text(info.description + "\n")
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
    """Build the sandbox and grader images from their hash-locked requirements."""
    settings = Settings()
    if not docker_available():
        typer.echo("docker daemon not reachable; start Docker and retry", err=True)
        raise typer.Exit(1)
    for spec in specs(settings):
        build_image(spec, settings)


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
    noise_k: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="A node must beat the best by this many fold std devs to be improved next "
            "(default: KAGENT_PROMOTION_NOISE_K, 0.5).",
        ),
    ] = None,
) -> None:
    """Run the draft/debug/improve loop. Resumes an existing tree in the workspace."""
    if policy != "greedy":
        raise typer.BadParameter("only 'greedy' is implemented")
    signal.signal(signal.SIGTERM, _interrupt)  # a Stop from the dashboard ends like Ctrl-C
    settings = Settings()
    if not docker_available():
        typer.echo("docker daemon not reachable; start Docker and retry", err=True)
        raise typer.Exit(2)
    if stale := stale_images(settings):
        typer.echo(
            f"{' and '.join(stale)} image missing or built from an older lock; "
            "run `kagent build-images`",
            err=True,
        )
        raise typer.Exit(2)
    ws = Workspace.for_slug(settings, competition)
    config = ws.load_config()
    if not (ws.public / "folds.csv").exists():
        raise typer.BadParameter("folds.csv missing; run `kagent init` with data first")

    description = ws.description_path.read_text() if ws.description_path.exists() else ""
    eda_text = ws.eda_path.read_text() if ws.eda_path.exists() else write_profile(ws, config)

    noise = settings.promotion_noise_k if noise_k is None else noise_k
    progress = Progress(
        ws,
        budget_s=budget_hours * 3600,
        params={
            "policy": policy,
            "noise_k": noise,
            "num_drafts": num_drafts,
            "max_nodes": max_nodes,
            "max_debug": max_debug,
            "planner_model": settings.planner_model,
            "coder_model": settings.coder_model,
            "planner_thinking": settings.planner_thinking,
            "planner_temperature": settings.planner_temperature,
            "coder_temperature": settings.coder_temperature,
        },
    )
    typer.echo(f"follow along: kagent dashboard -c {competition}")

    trajectory = ws.search_dir / "trajectory.jsonl"
    planner = ChatClient(
        settings.planner_base_url,
        settings.planner_model,
        role="planner",
        timeout_s=settings.llm_timeout_s,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.planner_temperature,
        top_p=settings.planner_top_p,
        top_k=settings.planner_top_k,
        template_kwargs=settings.planner_template_kwargs(),
        log_path=trajectory,
        observer=progress,
    )
    coder = ChatClient(
        settings.coder_base_url,
        settings.coder_model,
        role="coder",
        timeout_s=settings.llm_timeout_s,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.coder_temperature,
        top_p=settings.coder_top_p,
        top_k=settings.coder_top_k,
        log_path=trajectory,
        observer=progress,
    )
    operators = {
        "draft": DraftOperator(planner, coder),
        "improve": ImproveOperator(planner, coder),
        "debug": DebugOperator(coder),
    }
    tree = open_tree(ws, config)
    gate = DepGate(DependencyStore(_deps_dir(ws)), settings)
    try:
        run_search(
            ws,
            config,
            settings,
            tree,
            GreedyPolicy(num_drafts=num_drafts, noise_k=noise),
            operators,
            planner,
            description=description,
            eda=eda_text,
            budget_s=budget_hours * 3600,
            max_nodes=max_nodes,
            max_debug=max_debug,
            deps=gate,
            log=typer.echo,
            progress=progress,
        )
    except (LLMError, IntegrityAbort, OperatorFailures, DockerError) as exc:
        typer.echo(f"search stopped: {exc}", err=True)
        if isinstance(exc, DockerError):
            typer.echo("no node was recorded for that step; rerun to resume", err=True)
        raise typer.Exit(1) from exc

    if pending := gate.pending():
        typer.echo(
            f"{len(pending)} dependency request(s) waiting for a decision: "
            f"`kagent deps list -c {competition}`"
        )
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


@app.command()
def dashboard(
    competition: Annotated[
        str | None, typer.Option("--competition", "-c", help="Kaggle competition slug.")
    ] = None,
    host: Annotated[
        str,
        typer.Option(
            help="Interface to listen on. Anything but loopback shows your code and prompts "
            "to the network."
        ),
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on; 0 picks a free one.")] = 8765,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Open the page in a browser.")
    ] = True,
    demo: Annotated[
        bool, typer.Option(help="Show an invented search that advances in real time.")
    ] = False,
    read_only: Annotated[
        bool, typer.Option("--read-only", help="Hide the Start and Stop buttons.")
    ] = False,
) -> None:
    """Live dashboard of a search, with a form to start one. Run it beside `kagent search`."""
    settings = Settings()
    if demo:
        with tempfile.TemporaryDirectory(prefix="kagent-demo-") as tmp:
            run = DemoRun(Workspace(Path(tmp) / "demo"))
            run.populate(6)
            _serve(run.ws, settings, host, port, open_browser, then=run.play)
        return
    if competition is None:
        raise typer.BadParameter("pass --competition, or --demo to see an invented search")
    ws = Workspace.for_slug(settings, competition)
    if not ws.config_path.exists():
        raise typer.BadParameter(f"{ws.config_path} missing; run `kagent init` first")
    launcher = None
    if not read_only:
        if is_loopback(host):
            launcher = Launcher(ws, settings)
        else:
            typer.echo("run controls are off: only offered on a loopback address", err=True)
    _serve(ws, settings, host, port, open_browser, launcher=launcher)


def _serve(
    ws: Workspace,
    settings: Settings,
    host: str,
    port: int,
    open_browser: bool,
    then: Callable[[], None] | None = None,
    launcher: Launcher | None = None,
) -> None:
    try:
        server = make_server(WorkspaceReader(ws, settings), host, port, launcher)
    except OSError as exc:
        typer.echo(f"cannot listen on {host}:{port}: {exc}; try --port 0", err=True)
        raise typer.Exit(1) from exc
    url = f"http://{host}:{server.server_address[1]}/"
    mode = "can start and stop searches" if launcher else "read-only"
    typer.echo(f"dashboard for {ws.root.name} at {url} ({mode}; Ctrl-C to stop)")
    if not is_loopback(host):
        typer.echo("warning: reachable by the network; shows code, logs, prompts", err=True)
    if open_browser:
        webbrowser.open(url)
    if then is not None:
        threading.Thread(target=then, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


deps_app = typer.Typer(no_args_is_help=True, help="Review the agent's package requests.")
app.add_typer(deps_app, name="deps")


@deps_app.command("list")
def deps_list(
    competition: Competition,
    all_: Annotated[bool, typer.Option("--all", help="Include decided requests.")] = False,
) -> None:
    """Show dependency requests (pending ones unless --all)."""
    store = DependencyStore(_deps_dir(_workspace(competition)))
    requests = store.all() if all_ else store.all("pending")
    if not requests:
        typer.echo("no pending dependency requests" if not all_ else "no dependency requests")
        return
    for r in requests:
        flags = ", ".join(r.precheck.flags()) or "no warnings"
        typer.echo(f"{r.id}  {r.status:<8} {r.requirement:<24} node {r.node_id}  [{flags}]")


@deps_app.command("show")
def deps_show(competition: Competition, request_id: Annotated[str, typer.Argument()]) -> None:
    """Everything known about one request."""
    try:
        request = DependencyStore(_deps_dir(_workspace(competition))).get(request_id)
    except KeyError as exc:
        raise typer.BadParameter(str(exc.args[0])) from exc
    typer.echo(request.model_dump_json(indent=2))


@deps_app.command("approve")
def deps_approve(
    competition: Competition,
    request_id: Annotated[str, typer.Argument()],
    skip_cooldown: Annotated[
        bool, typer.Option(help="Allow a release newer than the cooldown, for this package only.")
    ] = False,
) -> None:
    """Add the package to the lock, rebuild the sandbox image, and let waiting nodes rerun."""
    settings = Settings()
    if not docker_available():
        typer.echo("docker daemon not reachable; start Docker and retry", err=True)
        raise typer.Exit(2)
    store = DependencyStore(_deps_dir(Workspace.for_slug(settings, competition)))
    try:
        request = approve_request(store, request_id, settings, skip_cooldown=skip_cooldown)
    except KeyError as exc:
        raise typer.BadParameter(str(exc.args[0])) from exc
    except InstallError as exc:
        typer.echo(f"not approved: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"approved {request.package} {request.resolved_version}; sandbox image rebuilt")


@deps_app.command("deny")
def deps_deny(
    competition: Competition,
    request_id: Annotated[str, typer.Argument()],
    reason: Annotated[str, typer.Option(help="Shown to the agent so it picks another approach.")],
) -> None:
    """Refuse a request. The waiting node is returned to debugging with the reason."""
    store = DependencyStore(_deps_dir(_workspace(competition)))
    try:
        deny_request(store, request_id, reason)
    except KeyError as exc:
        raise typer.BadParameter(str(exc.args[0])) from exc
    except InstallError as exc:
        typer.echo(f"not denied: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"denied {request_id}")
