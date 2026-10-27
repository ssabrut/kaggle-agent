"""Compact data profile for prompts: shapes, column types, target distribution."""

import pandas as pd

from kagent.config import CompetitionConfig
from kagent.workspace import Workspace

MAX_COLUMNS = 60
MAX_ROWS_READ = 200_000
OTHER_FILES = 20


def _describe_columns(df: pd.DataFrame) -> list[str]:
    lines = []
    for col in list(df.columns)[:MAX_COLUMNS]:
        s = df[col]
        examples = ", ".join(map(str, s.dropna().unique()[:3]))
        lines.append(
            f"  - {col}: {s.dtype}, {s.nunique()} unique, {s.isna().mean():.1%} missing, e.g. {examples}"
        )
    if len(df.columns) > MAX_COLUMNS:
        lines.append(f"  - ... {len(df.columns) - MAX_COLUMNS} more columns")
    return lines


def _describe_target(series: pd.Series) -> list[str]:
    if series.nunique() <= 20:
        dist = series.value_counts(normalize=True).round(3).to_dict()
        return [f"  distribution: {dist}"]
    d = series.describe()
    return [f"  min {d['min']:.4g}, mean {d['mean']:.4g}, max {d['max']:.4g}, std {d['std']:.4g}"]


def profile(ws: Workspace, config: CompetitionConfig) -> str:
    target = ws.target_column(config)
    lines = [f"Metric: {config.metric}. Id column: {config.id_column}. Target column: {target}."]

    for path in sorted(ws.public.glob("*.csv")):
        df = pd.read_csv(path, nrows=MAX_ROWS_READ)
        truncated = len(df) == MAX_ROWS_READ
        lines.append(
            f"\n{path.name}: {len(df)}{'+' if truncated else ''} rows x {len(df.columns)} cols"
        )
        if path.name == "folds.csv":
            continue
        lines += _describe_columns(df)
        if path.name == "train.csv" and target in df:
            lines.append(f"Target '{target}':")
            lines += _describe_target(df[target])

    others = sorted(p.name for p in ws.public.iterdir() if p.suffix != ".csv")
    if others:
        shown = ", ".join(others[:OTHER_FILES])
        lines.append(
            f"\nOther files in /data: {shown}{' ...' if len(others) > OTHER_FILES else ''}"
        )
    return "\n".join(lines) + "\n"


def write_profile(ws: Workspace, config: CompetitionConfig) -> str:
    text = profile(ws, config)
    ws.eda_path.write_text(text)
    return text
