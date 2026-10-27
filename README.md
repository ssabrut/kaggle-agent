# kagent

Autonomous Kaggle agent: an LLM proposes solution scripts, a sandbox runs them, an isolated grader scores them, and a search policy picks what to expand next. Design and evidence: the research spec (sections 3-9).

Status: **Milestones 1-2**. Harness (sandbox, isolated grader, integrity hashing, dry-run submit) plus a greedy AIDE-style draft/debug/improve loop. Not built yet: retrieval, ablation-guided refinement, UCT/evolutionary policies, merge/ensemble, skill store.

## Setup

```bash
uv sync
uv run kagent build-images     # needs a running Docker daemon
uv run pytest                  # docker tests skip if daemon or images are missing
```

LLM endpoints are plain OpenAI-compatible servers (`mlx_lm.server`), one per role. Defaults are in [config.py](src/kagent/config.py); override with `KAGENT_PLANNER_BASE_URL`, `KAGENT_PLANNER_MODEL`, `KAGENT_CODER_BASE_URL`, `KAGENT_CODER_MODEL`.

## Workflow

```bash
uv run kagent init   -c <slug> --metric auc [--id-column id] [--description desc.md]
                                # kaggle download, or --data-dir/--private-dir for local data
uv run kagent eda    -c <slug>                       # data profile used in prompts
uv run kagent search -c <slug> --budget-hours 6      # draft/debug/improve loop; resumes the tree
uv run kagent select -c <slug> --by robust_cv        # pick the node to submit
uv run kagent submit -c <slug>                       # dry run; add --approve to really submit
uv run kagent run    -c <slug> path/to/solution.py   # run one hand-written script through the harness
```

Downloading needs Kaggle credentials and the competition rules accepted on kaggle.com (otherwise 403).

## Script contract

Every generated script runs once in the sandbox. It reads `/data/{train,test,sample_submission,folds}.csv`, cross-validates on the provided `folds.csv`, and writes `/work/out/oof.csv` (out-of-fold predictions for every train row) and `/work/out/submission.csv`.

The grader computes the CV score from `oof.csv` against the train labels it holds, so the script cannot report a score it did not earn. It can still write dishonest predictions into `oof.csv` (for example by using the target); the held-out `score` in the run record (only when you supply `--private-dir` labels) exposes that as a gap between CV and held-out score.

## How isolation works

| Zone | Sees | Does not see |
|---|---|---|
| Sandbox container | `data/public` (read-only), its own `out/` | private labels, grader code, network, host env |
| Grader container | `data/public`, `data/private`, the run's `submission.csv` and `oof.csv` (all read-only) | the agent's script, network |
| Host | everything | |

Around every run the harness hashes `data/public`, `data/private` and the grader source before and after; any difference marks the run `integrity_violation` and stops the search. Output files that are not plain files (symlinks) are rejected, and `submit` re-checks the file hash recorded at grading time. Real submissions need `--approve`.

The search selects on cross-validation only. The held-out `score` is recorded for evaluation and never fed to the loop.

## Search loop

```text
policy.select(tree) -> Action(draft | debug | improve, parent)
operator.apply(parent, ctx) -> new script    planner proposes, coder writes (debug: coder only)
harness.run_script(...) -> RunRecord          sandbox -> integrity check -> grader (CV from oof.csv)
tree.add(node)                                ok / buggy / dead; summary kept for sibling context
```

Greedy policy: `--num-drafts` independent drafts (the first is a simple baseline), then debug a failed leaf (probability 0.5) or improve the best node. A node dies after `--max-debug` consecutive failed fixes. Prompts carry the parent script, a summary, sibling results and a log tail, never the full history. Every LLM call is appended to `workspaces/<slug>/search/trajectory.jsonl`.

## Layout

```text
src/kagent/
  cli.py  config.py  workspace.py  harness.py  submit.py  docker.py  grading.py  eda.py  folds.py
  llm/client.py         # OpenAI-compatible chat client, retries, trajectory log
  search/               # tree.py policies.py operators.py prompts.py loop.py select.py
  sandbox/runner.py     # agent container
  grader/               # self-contained; copied into the grader image
  integrity/hashing.py
docker/                 # sandbox and grader images
workspaces/<slug>/      # data/{public,private}, runs/<id>/, search/  (gitignored)
```
