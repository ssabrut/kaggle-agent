from typer.testing import CliRunner

from kagent.cli import app
from kagent.deps.models import DependencyRequest, PreCheck
from kagent.deps.store import DependencyStore

runner = CliRunner()


def seed(ws, **precheck):
    store = DependencyStore(ws.root / "deps")
    store.save(
        DependencyRequest(
            id="dep-001",
            package="catboost",
            reason="ModuleNotFoundError",
            node_id="n002",
            precheck=PreCheck(exists_on_index=True, **precheck),
        )
    )
    return store


def invoke(settings, *args):
    env = {"KAGENT_WORKSPACES_DIR": str(settings.workspaces_dir)}
    return runner.invoke(app, ["deps", *args, "-c", "toy"], env=env)


def test_list_shows_pending_requests_with_their_warnings(ws, settings):
    seed(ws, age_days=12, new_transitive=["graphviz"])
    result = invoke(settings, "list")
    assert result.exit_code == 0
    assert (
        "dep-001" in result.output and "catboost" in result.output and "node n002" in result.output
    )
    assert "NEW PACKAGE (12d)" in result.output and "+1 TRANSITIVE" in result.output


def test_list_is_empty_message_and_all_includes_decided(ws, settings):
    assert "no pending" in invoke(settings, "list").output
    store = seed(ws)
    request = store.get("dep-001")
    request.status = "denied"
    store.save(request)
    assert "no pending" in invoke(settings, "list").output
    assert "denied" in invoke(settings, "list", "--all").output


def test_deny_records_the_reason_the_agent_will_see(ws, settings):
    seed(ws)
    result = invoke(settings, "deny", "dep-001", "--reason", "unknown maintainer")
    assert result.exit_code == 0 and "denied dep-001" in result.output
    stored = DependencyStore(ws.root / "deps").get("dep-001")
    assert stored.status == "denied" and stored.decision_reason == "unknown maintainer"


def test_deny_needs_a_reason(ws, settings):
    seed(ws)
    assert invoke(settings, "deny", "dep-001").exit_code != 0


def test_deciding_twice_or_an_unknown_id_fails_cleanly(ws, settings):
    seed(ws)
    invoke(settings, "deny", "dep-001", "--reason", "x")
    again = invoke(settings, "deny", "dep-001", "--reason", "y")
    assert again.exit_code == 1 and "already denied" in again.output
    unknown = invoke(settings, "show", "dep-999")
    assert unknown.exit_code != 0 and "no dependency request" in unknown.output


def test_show_prints_the_full_record(ws, settings):
    seed(ws, vulns=["PYSEC-1"])
    result = invoke(settings, "show", "dep-001")
    assert '"package": "catboost"' in result.output and "PYSEC-1" in result.output
