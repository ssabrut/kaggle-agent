# kagent

Autonomous Kaggle agent: an LLM proposes solution scripts, a sandbox runs them, an isolated grader scores them, and a search policy picks what to expand next. Design and evidence: the research spec (sections 3-9).

Status: **Milestone 1 (harness)**. Sandbox runner, isolated grader, integrity hashing, dry-run submit. No LLM or search loop yet.

## Setup

```bash
uv sync
uv run kagent build-images     # needs a running Docker daemon
uv run pytest                  # docker tests skip if daemon or images are missing
```

## Commands

```bash
uv run kagent init   -c <slug> --metric auc [--id-column id]   # kaggle download, or --data-dir/--private-dir
uv run kagent run    -c <slug> path/to/solution.py             # sandbox -> integrity check -> grade
uv run kagent submit -c <slug> --run <run_id>                  # dry run; add --approve to really submit
```

Downloading needs Kaggle credentials and the competition rules accepted on kaggle.com (otherwise 403).

## How isolation works

| Zone | Sees | Does not see |
|---|---|---|
| Sandbox container | `data/public` (read-only), its own `out/` | private labels, grader code, network, host env |
| Grader container | `data/public`, `data/private`, the one `submission.csv` (all read-only) | the agent's script, network |
| Host | everything | |

Around every run the harness hashes `data/public`, `data/private` and the grader source before and after; any difference marks the run `integrity_violation`. Submissions that are not plain files (symlinks) are rejected, and `submit` re-checks the file hash recorded at grading time. Real submissions need `--approve`.

Scripts read from `/data` and must write `/work/out/submission.csv`.

## Layout

```
src/kagent/
  cli.py  config.py  workspace.py  harness.py  submit.py  docker.py  grading.py
  sandbox/runner.py     # agent container
  grader/               # self-contained; copied into the grader image
  integrity/hashing.py
docker/                 # sandbox and grader images
workspaces/<slug>/      # data/{public,private}, runs/<id>/  (gitignored)
```
