from pathlib import Path

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from kagent.grader import metrics

# Valid for a source checkout (uv sync); used to find docker/ and the default workspaces dir.
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Process-wide knobs, overridable with KAGENT_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="KAGENT_")

    workspaces_dir: Path = REPO_ROOT / "workspaces"
    sandbox_image: str = "kagent-sandbox:latest"
    grader_image: str = "kagent-grader:latest"
    step_timeout_s: int = 600
    sandbox_memory: str = "8g"
    sandbox_cpus: float = 4
    grader_timeout_s: int = 120
    grader_memory: str = "2g"


class CompetitionConfig(BaseModel):
    """Per-competition settings, stored as config.json in the workspace."""

    slug: str
    metric: str
    id_column: str = "id"

    @field_validator("metric")
    @classmethod
    def _known_metric(cls, value: str) -> str:
        if value not in metrics.METRICS:
            raise ValueError(f"unknown metric '{value}', choose from {metrics.available()}")
        return value
