"""Model sampling and thinking settings."""

import pytest
from pydantic import ValidationError

from kagent.config import Settings


def test_defaults_follow_the_model_cards():
    settings = Settings()
    assert (settings.coder_temperature, settings.coder_top_p) == (1.0, 0.95)
    assert settings.coder_top_k == 40
    assert (settings.planner_temperature, settings.planner_top_p) == (1.0, 0.95)
    assert settings.planner_top_k == 20


@pytest.mark.parametrize(
    ("thinking", "kwargs"),
    [
        ("medium", {"reasoning_effort": "medium"}),
        ("low", {"reasoning_effort": "low"}),
        ("xhigh", {"reasoning_effort": "xhigh"}),
        ("off", {"enable_thinking": False}),
        ("default", {}),
    ],
)
def test_planner_thinking_becomes_chat_template_kwargs(thinking, kwargs):
    assert Settings(planner_thinking=thinking).planner_template_kwargs() == kwargs


def test_the_planner_defaults_to_full_effort_thinking():
    assert Settings().planner_template_kwargs() == {"reasoning_effort": "xhigh"}


def test_an_unknown_thinking_level_is_refused(monkeypatch):
    monkeypatch.setenv("KAGENT_PLANNER_THINKING", "max")
    with pytest.raises(ValidationError):
        Settings()
