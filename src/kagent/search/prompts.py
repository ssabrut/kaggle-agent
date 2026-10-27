"""Prompt builders. Each returns a chat message list; none of them include full history."""

from collections.abc import Sequence

from kagent.llm.client import Message

SYSTEM_PLANNER = (
    "You are a Kaggle grandmaster planning solutions for a tabular machine learning "
    "competition. Be concrete and brief."
)
SYSTEM_CODER = (
    "You are an expert Kaggle data scientist who writes complete, correct, runnable Python "
    "scripts. Reply with exactly one fenced ```python block containing the whole script."
)

PREDICTION_HINT = {
    "auc": "the predicted probability of the positive class, a float in [0, 1]",
    "logloss": "the predicted probability of the positive class, clipped to [1e-15, 1 - 1e-15]",
    "accuracy": "the predicted class label, with the same values as the training target",
    "f1_macro": "the predicted class label, with the same values as the training target",
    "rmse": "the predicted numeric value",
    "mae": "the predicted numeric value",
}

# What a failed run had finished, keyed by kagent.stages. Worded to fit a timeout as well as a crash.
STAGE_PROGRESS = {
    "start": "It reported no finished step, so it stopped while loading the data or before "
    "its first STAGE line.",
    "load": "The data loaded; it stopped before the feature matrix was built.",
    "features": "The features were built; it stopped before cross-validation finished.",
    "cv": "Cross-validation finished and oof.csv was written; it stopped before submission.csv "
    "was written.",
    "predict": "submission.csv was written; the problem comes after that or is in the "
    "submission itself.",
}

CLIP_CODE = 12_000
CLIP_TEXT = 6_000
CLIP_LOG = 3_000


def clip(text: str, limit: int) -> str:
    """Keep the head and tail of long text."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} characters cut] ...\n{text[-half:]}"


def direction(higher_is_better: bool) -> str:
    return "higher is better" if higher_is_better else "lower is better"


def available_libraries(libraries: dict[str, str]) -> str:
    """Tell the planner what is installed and what leaving that set costs."""
    names = ", ".join(libraries)
    return (
        f"The script can import the Python standard library plus {names}. Prefer these. Any other "
        "package makes the script fail until a human approves an install request, which can take "
        "hours, so import something else only when it is clearly worth the wait."
    )


def contract(
    *,
    metric: str,
    id_column: str,
    target: str,
    step_timeout_s: int,
    cpus: float,
    memory: str,
    libraries: dict[str, str],
) -> str:
    libs = ", ".join(f"{name} {version}" for name, version in libraries.items())
    return f"""Environment contract (the script is run exactly as written, once):
- Python 3.11 with {libs} (plus their dependencies). No internet, no pip install.
  These versions are newer than your training data: do not use APIs removed in recent releases
  (for example CalibratedClassifierCV(cv="prefit"), use FrozenEstimator; DataFrame.append).
- Read /data/train.csv, /data/test.csv, /data/sample_submission.csv and /data/folds.csv
  (columns: {id_column}, fold). Never write to /data. Do not read any other path.
- Cross-validate with the provided folds only: for each fold k, fit on rows with fold != k and
  predict rows with fold == k. Do not create your own split.
- Write /work/out/oof.csv with columns [{id_column}, {target}]: the out-of-fold prediction for
  every train row.
- Write /work/out/submission.csv with exactly the columns, order and rows of
  sample_submission.csv. Use models fitted on all train data, or the average of the fold models.
- Predictions in both files are {PREDICTION_HINT[metric]}. The metric is {metric}.
- Commit to ONE final model or blend. oof.csv, submission.csv and the printed score must all
  come from that same model; if you compare several models, keep the best one's predictions and
  do not overwrite them with a later model's. The grader recomputes the score from oof.csv alone.
- Print the cross-validated {metric} to stdout. Also print a line `STAGE: <name>` as each step
  finishes, only after it has really finished: `STAGE: load` once the data is read,
  `STAGE: features` once the feature matrix is built, `STAGE: cv` once every fold is done and
  oof.csv is written, `STAGE: predict` once submission.csv is written. A failed run is
  diagnosed from these lines.
- Hard limits per run: {step_timeout_s} s wall clock, {cpus:g} CPUs, {memory} RAM. Keep the
  script well inside them (subsample or limit estimators if needed)."""


def _tried_block(items: list[tuple[str, str]], empty: str) -> str:
    if not items:
        return empty
    return "\n".join(f"- {name}: {text}" for name, text in items)


def _blocklist_block(blocklist: Sequence[str]) -> str:
    """Empty when there is nothing to avoid, so a clean search sees no extra section."""
    if not blocklist:
        return ""
    items = "\n".join(f"- {entry}" for entry in blocklist)
    return (
        "\nFailed or did not help earlier in the search (do not repeat these or close "
        f"variants):\n{items}\n"
    )


def plan_draft(
    *,
    description: str,
    eda: str,
    metric: str,
    higher_is_better: bool,
    tried: list[tuple[str, str]],
    first: bool,
    remaining_min: float,
    libraries: dict[str, str],
    blocklist: Sequence[str] = (),
) -> list[Message]:
    style = (
        "This is the first draft. Plan a simple, robust baseline that is very likely to run "
        "correctly, for example one gradient boosting or linear model with basic preprocessing."
        if first
        else "Plan an approach that differs meaningfully from the ones already tried."
    )
    user = f"""Competition description:
{clip(description or "(none provided)", CLIP_TEXT)}

Data profile:
{clip(eda, CLIP_TEXT)}

Metric: {metric} ({direction(higher_is_better)}). About {remaining_min:.0f} minutes of search time remain.
{available_libraries(libraries)}

Approaches already tried:
{_tried_block(tried, "(none yet)")}
{_blocklist_block(blocklist)}
{style}
Write a short plan of 3-6 bullets: preprocessing, features, model, and how to use the folds. No code."""
    return [{"role": "system", "content": SYSTEM_PLANNER}, {"role": "user", "content": user}]


def code_from_plan(*, contract_text: str, plan: str) -> list[Message]:
    user = f"""{contract_text}

Plan:
{plan}

Write the complete script implementing this plan. Begin the file with the plan as '# PLAN:'
comment lines."""
    return [{"role": "system", "content": SYSTEM_CODER}, {"role": "user", "content": user}]


def plan_improvement(
    *,
    code: str,
    metric: str,
    higher_is_better: bool,
    cv_score: float,
    cv_std: float | None,
    summary: str,
    tried: list[tuple[str, str]],
    remaining_min: float,
    libraries: dict[str, str],
    blocklist: Sequence[str] = (),
) -> list[Message]:
    spread = f" +/- {cv_std:.5f}" if cv_std is not None else ""
    user = f"""Metric: {metric} ({direction(higher_is_better)}). About {remaining_min:.0f} minutes of search time remain.
{available_libraries(libraries)}

Current best solution, cross-validated {metric} = {cv_score:.5f}{spread}:
```python
{clip(code, CLIP_CODE)}
```
What happened when it ran: {summary or "(no summary)"}

Changes already tried from this solution, with results:
{_tried_block(tried, "(none yet)")}
{_blocklist_block(blocklist)}
Propose exactly ONE specific change likely to improve the cross-validated {metric}, such as a
new feature, a hyperparameter change, a different model, or a blend. Do not repeat a change
that was already tried. Reply with 2-4 sentences and no code."""
    return [{"role": "system", "content": SYSTEM_PLANNER}, {"role": "user", "content": user}]


def code_improve(*, contract_text: str, code: str, idea: str) -> list[Message]:
    user = f"""{contract_text}

Current script:
```python
{clip(code, CLIP_CODE)}
```

Change to make: {idea}

Return the complete modified script. Keep everything that already works and change only what
this idea requires. Update the '# PLAN:' header to mention the change."""
    return [{"role": "system", "content": SYSTEM_CODER}, {"role": "user", "content": user}]


def progress_note(stage: str) -> str:
    """Where the failed run stopped; empty when the stage is unknown or nothing is left to fix."""
    text = STAGE_PROGRESS.get(stage)
    return f"\nHow far it got: {text}\n" if text else ""


def code_debug(*, contract_text: str, code: str, error: str, stage: str = "") -> list[Message]:
    user = f"""{contract_text}

This script failed.
```python
{clip(code, CLIP_CODE)}
```

Error output:
```
{clip(error, CLIP_LOG)}
```
{progress_note(stage)}
Fix the problem without changing the overall approach. Return the complete corrected script."""
    return [{"role": "system", "content": SYSTEM_CODER}, {"role": "user", "content": user}]


def summarize_run(
    *, code: str, log: str, metric: str, cv_score: float, cv_std: float | None
) -> list[Message]:
    spread = f" +/- {cv_std:.5f}" if cv_std is not None else ""
    user = f"""Summarize this experiment in 2-3 sentences for use as memory: model and features,
key hyperparameters, the result, and anything notable in the log such as warnings or slow
steps. No code.

Cross-validated {metric} = {cv_score:.5f}{spread}

Script:
```python
{clip(code, CLIP_CODE)}
```

Log tail:
{clip(log, CLIP_LOG)}"""
    return [{"role": "system", "content": SYSTEM_PLANNER}, {"role": "user", "content": user}]
