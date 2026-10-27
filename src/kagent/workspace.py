from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from kagent.config import CompetitionConfig, Settings


@dataclass(frozen=True)
class Workspace:
    """Directory layout for one competition.

    data/public   train, test, sample_submission, folds; mounted read-only into the sandbox
    data/private  held-out labels; mounted only into the grader container
    runs/<id>     script, sandbox output, logs and the run record
    search/       solution tree (nodes.jsonl), node scripts, LLM trajectory log
    """

    root: Path

    @classmethod
    def for_slug(cls, settings: Settings, slug: str) -> "Workspace":
        return cls(settings.workspaces_dir / slug)

    @property
    def public(self) -> Path:
        return self.root / "data" / "public"

    @property
    def private(self) -> Path:
        return self.root / "data" / "private"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def search_dir(self) -> Path:
        return self.root / "search"

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def description_path(self) -> Path:
        return self.root / "description.md"

    @property
    def eda_path(self) -> Path:
        return self.root / "eda.md"

    @property
    def selected_path(self) -> Path:
        return self.root / "selected.json"

    @property
    def page_dir(self) -> Path:
        """What `init --fetch-page` read from the competition website."""
        return self.root / "page"

    def infer_id_column(self) -> str:
        """The first column of sample_submission.csv, which Kaggle uses for the row id."""
        sample = self.public / "sample_submission.csv"
        if not sample.exists():
            raise FileNotFoundError(f"{sample} not found; pass --id-column")
        return str(pd.read_csv(sample, nrows=0).columns[0])

    def run_dir(self, run_id: str) -> Path:
        return self.runs / run_id

    def create(self) -> None:
        for path in (self.public, self.private, self.runs):
            path.mkdir(parents=True, exist_ok=True)

    def load_config(self) -> CompetitionConfig:
        if not self.config_path.exists():
            raise FileNotFoundError(f"{self.config_path} missing; run `kagent init` first")
        return CompetitionConfig.model_validate_json(self.config_path.read_text())

    def save_config(self, config: CompetitionConfig) -> None:
        self.config_path.write_text(config.model_dump_json(indent=2) + "\n")

    def target_column(self, config: CompetitionConfig) -> str:
        """The single prediction column of sample_submission.csv."""
        header = pd.read_csv(self.public / "sample_submission.csv", nrows=0).columns
        targets = [c for c in header if c != config.id_column]
        if len(targets) != 1:
            raise ValueError(f"only one target column is supported, found {targets}")
        return targets[0]
