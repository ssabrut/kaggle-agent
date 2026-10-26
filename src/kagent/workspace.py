from dataclasses import dataclass
from pathlib import Path

from kagent.config import CompetitionConfig, Settings


@dataclass(frozen=True)
class Workspace:
    """Directory layout for one competition.

    data/public   train, test, sample_submission; mounted read-only into the sandbox
    data/private  held-out labels; mounted only into the grader container
    runs/<id>     script, sandbox output, logs and the run record
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
    def config_path(self) -> Path:
        return self.root / "config.json"

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
