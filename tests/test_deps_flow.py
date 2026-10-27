"""Gate, approve/deny, and the search loop's handling of parked nodes (no network, no Docker)."""

import shutil
import subprocess
from datetime import UTC, datetime, timedelta

import pytest

from fakes import FakeChat, code_block
from kagent.config import REPO_ROOT
from kagent.deps.gate import DepGate
from kagent.deps.install import InstallError, approve_request, deny_request
from kagent.deps.lockfile import normalize
from kagent.deps.store import DependencyStore
from kagent.grader.result import GradeResult
from kagent.harness import RunRecord
from kagent.search.loop import open_tree, run_search
from kagent.search.operators import DebugOperator, DraftOperator, ImproveOperator
from kagent.search.policies import GreedyPolicy

NOW = datetime(2026, 10, 27, 12, tzinfo=UTC)
MISSING = "Traceback (most recent call last):\nModuleNotFoundError: No module named '{}'"


class FakeIndex:
    def __init__(self, unknown=()):
        self.unknown = {normalize(u) for u in unknown}

    def fetch(self, package):
        if normalize(package) in self.unknown:
            return None
        return {
            "releases": {"1": [{"upload_time_iso_8601": "2020-01-01T00:00:00Z"}]},
            "vulnerabilities": [],
        }


def fake_resolver(text):
    return {"numpy": "2.4.6", "catboost": "1.2.8", "graphviz": "0.2"}


@pytest.fixture
def docker_dir(tmp_path):
    target = tmp_path / "docker"
    target.mkdir()
    for name in ("sandbox.in", "sandbox.lock.txt"):
        shutil.copy(REPO_ROOT / "docker" / name, target / name)
    return target


@pytest.fixture
def cfg(settings, docker_dir):
    return settings.model_copy(update={"docker_dir": docker_dir, "max_dep_requests": 3})


@pytest.fixture
def store(ws):
    return DependencyStore(ws.root / "deps")


def make_gate(store, cfg, index=None, now=NOW):
    return DepGate(store, cfg, index=index or FakeIndex(), resolver=fake_resolver, now=lambda: now)


# --- gate -----------------------------------------------------------------------------------


def test_missing_package_becomes_a_pending_request_with_prechecks(store, cfg):
    request = make_gate(store, cfg).request_for_failure("n002", MISSING.format("catboost"))
    assert request.status == "pending" and request.package == "catboost"
    assert request.node_id == "n002" and "catboost" in request.reason
    assert (
        request.precheck.exists_on_index
        and request.precheck.new_transitive == ["catboost", "graphviz"][1:]
    )
    assert DependencyStore(store.dir).get(request.id).package == "catboost"  # persisted
    events = [line for line in store.events_path.read_text().splitlines() if "requested" in line]
    assert len(events) == 1


def test_other_failures_make_no_request(store, cfg):
    gate = make_gate(store, cfg)
    assert gate.request_for_failure("n001", "KeyError: 'zip'") is None
    assert gate.request_for_failure("n001", MISSING.format("sklearn.bogus")) is None
    assert store.all() == []


def test_hallucinated_name_is_denied_without_a_human(store, cfg):
    gate = make_gate(store, cfg, index=FakeIndex(unknown=["catboostt"]))
    request = gate.request_for_failure("n001", MISSING.format("catboostt"))
    assert request.status == "denied" and "not found" in request.decision_reason
    assert store.all("pending") == []


def test_same_package_reuses_pending_and_replays_denial(store, cfg):
    gate = make_gate(store, cfg)
    first = gate.request_for_failure("n001", MISSING.format("catboost"))
    again = gate.request_for_failure("n005", MISSING.format("catboost"))
    assert again.id == first.id and len(store.all()) == 1

    deny_request(store, first.id, "unknown maintainer")
    replay = gate.request_for_failure("n006", MISSING.format("catboost"))
    assert replay.status == "denied" and replay.decision_reason == "unknown maintainer"
    assert len(store.all()) == 1  # a denied package is never requested again


def test_approved_package_that_is_still_missing_is_an_ordinary_failure(store, cfg):
    gate = make_gate(store, cfg)
    request = gate.request_for_failure("n001", MISSING.format("catboost"))
    request.status = "approved"
    store.save(request)
    assert gate.request_for_failure("n002", MISSING.format("catboost")) is None


def test_requests_per_session_are_capped(store, cfg):
    gate = make_gate(store, cfg)
    for pkg in ("aaa", "bbb", "ccc"):
        assert gate.request_for_failure("n001", MISSING.format(pkg)) is not None
    assert gate.request_for_failure("n001", MISSING.format("ddd")) is None
    assert "request_cap_reached" in store.events_path.read_text()


def test_requests_expire_after_the_timeout(store, cfg):
    request = make_gate(store, cfg).request_for_failure("n001", MISSING.format("catboost"))
    later = NOW + timedelta(seconds=cfg.dep_request_timeout_s + 1)
    gate = make_gate(store, cfg, now=later)
    assert gate.is_stale(request) and not make_gate(store, cfg).is_stale(request)
    assert gate.expire(request).status == "expired"


# --- approve / deny -------------------------------------------------------------------------


def fake_compile(version="1.2.8"):
    calls = []

    def compile_fn(in_path, lock_path, **kwargs):
        calls.append(kwargs)
        lock_path.write_text(
            lock_path.read_text() + f"catboost=={version} \\\n    --hash=sha256:ab\n"
        )

    compile_fn.calls = calls
    return compile_fn


def pending(store, cfg):
    return make_gate(store, cfg).request_for_failure("n001", MISSING.format("catboost"))


def test_approve_edits_the_in_file_recompiles_and_rebuilds(store, cfg, docker_dir):
    request = pending(store, cfg)
    built = []
    compile_fn = fake_compile()

    done = approve_request(
        store,
        request.id,
        cfg,
        compile_fn=compile_fn,
        build_fn=lambda spec, s: built.append(spec.kind),
    )

    assert done.status == "approved" and done.resolved_version == "1.2.8"
    assert "catboost" in (docker_dir / "sandbox.in").read_text()
    assert built == ["sandbox"]
    assert compile_fn.calls[0] == {"cooldown": "7 days", "cooldown_exempt": None}
    assert DependencyStore(store.dir).get(request.id).status == "approved"
    assert "approved" in store.events_path.read_text()


def test_skip_cooldown_exempts_only_that_package(store, cfg):
    request = pending(store, cfg)
    compile_fn = fake_compile()
    approve_request(
        store, request.id, cfg, skip_cooldown=True, compile_fn=compile_fn, build_fn=lambda *a: None
    )
    assert compile_fn.calls[0]["cooldown_exempt"] == "catboost"


def test_failed_resolve_restores_files_and_keeps_the_request_pending(store, cfg, docker_dir):
    request = pending(store, cfg)
    before = {n: (docker_dir / n).read_text() for n in ("sandbox.in", "sandbox.lock.txt")}

    def broken(in_path, lock_path, **kwargs):
        lock_path.write_text("half written")
        raise InstallError("no versions older than 7 days")

    with pytest.raises(InstallError, match="older than 7 days"):
        approve_request(store, request.id, cfg, compile_fn=broken, build_fn=lambda *a: None)

    assert {n: (docker_dir / n).read_text() for n in before} == before
    assert store.get(request.id).status == "pending"
    assert "install_failed" in store.events_path.read_text()


def test_failed_image_build_restores_files(store, cfg, docker_dir):
    request = pending(store, cfg)
    before = (docker_dir / "sandbox.lock.txt").read_text()

    def bad_build(spec, settings):
        raise subprocess.CalledProcessError(1, "docker build")

    with pytest.raises(InstallError):
        approve_request(store, request.id, cfg, compile_fn=fake_compile(), build_fn=bad_build)
    assert (docker_dir / "sandbox.lock.txt").read_text() == before
    assert store.get(request.id).status == "pending"


def test_package_missing_from_the_lock_after_resolving_is_an_error(store, cfg, docker_dir):
    request = pending(store, cfg)
    with pytest.raises(InstallError, match="missing from the lock"):
        approve_request(
            store, request.id, cfg, compile_fn=lambda *a, **k: None, build_fn=lambda *a: None
        )
    assert "catboost" not in (docker_dir / "sandbox.in").read_text()


def test_only_pending_requests_can_be_decided(store, cfg):
    request = pending(store, cfg)
    deny_request(store, request.id, "no")
    with pytest.raises(InstallError, match="already denied"):
        deny_request(store, request.id, "again")
    with pytest.raises(InstallError, match="already denied"):
        approve_request(store, request.id, cfg, compile_fn=fake_compile(), build_fn=lambda *a: None)
    with pytest.raises(KeyError):
        store.get("dep-999")


# --- search loop ----------------------------------------------------------------------------


def make_run(installed):
    """Scripts containing '# NEEDS catboost' fail until `installed` has 'catboost'."""

    def run(ws, script, settings):
        text = script.read_text()
        base = {
            "run_id": f"run-{script.stem}-{len(installed)}",
            "slug": "toy",
            "duration_s": 1.0,
            "script_sha256": "x",
        }
        if "# NEEDS catboost" in text and "catboost" not in installed:
            return RunRecord(
                **base, exit_code=1, status="failed", stderr_tail=MISSING.format("catboost")
            )
        return RunRecord(
            **base,
            exit_code=0,
            status="ok",
            grade=GradeResult(valid=True, cv_score=0.8, cv_std=0.01),
        )

    return run


def search(ws, cfg, gate, coder_replies, installed, **kwargs):
    config = ws.load_config()
    planner = FakeChat(["- plan"] * 20, role="planner")
    coder = FakeChat(coder_replies, role="coder")
    tree = open_tree(ws, config)
    operators = {
        "draft": DraftOperator(planner, coder),
        "improve": ImproveOperator(planner, coder),
        "debug": DebugOperator(coder),
    }
    logs: list[str] = []
    run_search(
        ws, config, cfg, tree, GreedyPolicy(num_drafts=2, debug_prob=1.0), operators, planner,
        description="d", eda="e", budget_s=3600, run=make_run(installed), deps=gate,
        log=logs.append, **kwargs,
    )  # fmt: skip
    return tree, coder, logs


NEEDS = code_block("# NEEDS catboost\nprint('x')")
FINE = code_block("print('fine')")


def test_a_missing_package_parks_the_node_and_the_search_moves_on(ws, cfg, store):
    tree, _, logs = search(ws, cfg, make_gate(store, cfg), [NEEDS, FINE], set(), max_nodes=2)

    n1, n2 = tree.nodes
    assert n1.status == "blocked" and n1.dep_request == "dep-001"
    assert n2.status == "ok"  # the other branch ran while the request waits
    assert "waiting on dep-001" in logs[0]
    assert tree.buggy_leaves() == []  # a parked node is not offered to the debug operator


def test_approval_reruns_the_parked_node_in_place(ws, cfg, store):
    installed: set[str] = set()
    gate = make_gate(store, cfg)
    search(ws, cfg, gate, [NEEDS, FINE], installed, max_nodes=2)

    approve_request(store, "dep-001", cfg, compile_fn=fake_compile(), build_fn=lambda *a: None)
    installed.add("catboost")  # the rebuilt image now has the package

    tree, _, logs = search(ws, cfg, gate, [FINE], installed, max_nodes=1)
    n1 = tree.get("n001")
    assert n1.status == "ok" and n1.cv_score == 0.8 and n1.dep_request is None
    assert len(tree.nodes) == 3  # the rerun updated n001; only the new step added a node
    assert any("catboost 1.2.8 installed, rerunning" in line for line in logs)


def test_denial_returns_the_node_to_debugging_with_the_reason(ws, cfg, store):
    gate = make_gate(store, cfg)
    search(ws, cfg, gate, [NEEDS, FINE], set(), max_nodes=2)
    deny_request(store, "dep-001", "unknown maintainer")

    tree, coder, _ = search(ws, cfg, gate, [FINE], set(), max_nodes=1)
    n1 = tree.get("n001")
    assert n1.status == "buggy" and n1.dep_request is None
    assert "denied (unknown maintainer)" in n1.error and "installed libraries" in n1.error
    debug_child = next(n for n in tree.nodes if n.parent == "n001")
    assert debug_child.operator == "debug"
    assert "denied (unknown maintainer)" in coder.user_text(0)  # the coder sees why


def test_a_request_left_waiting_too_long_kills_the_node(ws, cfg, store):
    search(ws, cfg, make_gate(store, cfg), [NEEDS, FINE], set(), max_nodes=2)
    later = make_gate(store, cfg, now=NOW + timedelta(seconds=cfg.dep_request_timeout_s + 5))

    tree, _, logs = search(ws, cfg, later, [FINE], set(), max_nodes=1)
    assert tree.get("n001").status == "dead" and "expired" in tree.get("n001").error
    assert store.get("dep-001").status == "expired"
    assert any("expired" in line for line in logs)


def test_a_second_node_needing_a_denied_package_goes_straight_to_debug(ws, cfg, store):
    gate = make_gate(store, cfg)
    search(ws, cfg, gate, [NEEDS, FINE], set(), max_nodes=2)
    deny_request(store, "dep-001", "no")

    tree, _, _ = search(ws, cfg, gate, [NEEDS], set(), max_nodes=1)
    newest = tree.nodes[-1]
    assert newest.id != "n001" and newest.status == "buggy" and newest.dep_request is None
    assert len(store.all()) == 1


def test_search_without_a_gate_treats_missing_modules_as_plain_failures(ws, cfg):
    tree, _, _ = search(ws, cfg, None, [NEEDS], set(), max_nodes=1)
    assert tree.nodes[0].status == "buggy" and "ModuleNotFoundError" in tree.nodes[0].error
