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

    # Search (spec section 13): a node is promoted only if it beats the incumbent by more than
    # this many fold std devs, since smaller gains are validation noise. Tune per task.
    promotion_noise_k: float = 0.5

    # Dependency gate (spec section 11): the agent requests packages, a human approves them.
    docker_dir: Path = REPO_ROOT / "docker"
    dep_cooldown: str = "7 days"  # newest release the resolver may pick
    dep_request_timeout_s: int = 6 * 3600  # a node waiting longer than this is marked dead
    max_dep_requests: int = 5  # per search session

    # OpenAI-compatible servers (mlx_lm.server), one per role.
    planner_base_url: str = "http://jupyter-02.aml1.id.iosda.org:11000/v1"
    coder_base_url: str = "http://jupyter-02.aml1.id.iosda.org:10000/v1"
    planner_model: str = "mlx-community/Qwen3.8-Flash-Next-4bit"
    coder_model: str = "mlx-community/Qwen3-Coder-Next-4bit"
    llm_timeout_s: float = 900
    llm_max_tokens: int = 8192

    @field_validator("planner_base_url", "coder_base_url")
    @classmethod
    def _base_url(cls, value: str) -> str:
        """Accept either the base URL or the full .../chat/completions URL."""
        return value.rstrip("/").removesuffix("/chat/completions")


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
