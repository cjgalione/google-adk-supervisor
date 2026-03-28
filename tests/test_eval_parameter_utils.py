from braintrust.logger import Prompt
from braintrust.prompt import PromptCompletionBlock, PromptData
from pydantic import BaseModel, Field

from evals.parameter_utils import (
    extract_prompt_text_and_model,
    get_hook_parameters,
    resolve_agent_config_overrides,
    resolve_prompt_and_model,
    unwrap_parameter_value,
    unwrap_parameters,
)
from src.config import AgentConfig


class _ValueParam(BaseModel):
    value: str = Field(default="default-value")


class _MultiFieldParam(BaseModel):
    a: int = 1
    b: int = 2


class _Hooks:
    def __init__(self, parameters):
        self.parameters = parameters


class _NonStringPromptObject:
    pass


def test_get_hook_parameters_returns_empty_for_missing_hooks():
    assert get_hook_parameters(None) == {}


def test_get_hook_parameters_returns_dict_for_valid_hooks():
    params = {"key": "value"}
    assert get_hook_parameters(_Hooks(params)) == params


def test_unwrap_parameter_value_supports_instances_and_classes():
    assert unwrap_parameter_value(_ValueParam(value="x")) == "x"
    assert unwrap_parameter_value(_ValueParam) == "default-value"


def test_unwrap_parameter_value_falls_back_for_non_value_models():
    multi = _MultiFieldParam()
    assert unwrap_parameter_value(multi) == multi


def test_unwrap_parameters_discards_none_values():
    unwrapped = unwrap_parameters(
        {
            "present": _ValueParam(value="ok"),
            "none_value": None,
        }
    )
    assert unwrapped == {"present": "ok"}


def test_extract_prompt_text_and_model_supports_braintrust_prompt_objects():
    prompt = Prompt.from_prompt_data(
        "research_agent_prompt",
        PromptData(
            prompt=PromptCompletionBlock(content="Use the web carefully."),
            options={"model": "gemini-2.5-flash"},
        ),
    )

    prompt_text, model = extract_prompt_text_and_model(prompt)

    assert prompt_text == "Use the web carefully."
    assert model == "gemini-2.5-flash"


def test_resolve_prompt_and_model_prefers_embedded_prompt_object_values():
    prompt = Prompt.from_prompt_data(
        "math_agent_prompt",
        PromptData(
            prompt=PromptCompletionBlock(content="Do the math."),
            options={"model": "gemini-2.5-pro"},
        ),
    )

    prompt_text, model = resolve_prompt_and_model(
        {
            "math_agent_prompt": prompt,
            "math_model": _ValueParam(value="legacy-model"),
        },
        prompt_key="math_agent_prompt",
        model_key="math_model",
        default_model="default-model",
    )

    assert prompt_text == "Do the math."
    assert model == "gemini-2.5-pro"


def test_resolve_agent_config_overrides_expands_prompt_objects_into_config_fields():
    prompt = Prompt.from_prompt_data(
        "system_prompt",
        PromptData(
            prompt=PromptCompletionBlock(content="Delegate when needed."),
            options={"model": "gemini-2.5-mini"},
        ),
    )

    overrides = resolve_agent_config_overrides(
        {
            "system_prompt": prompt,
            "prompt_modification": _ValueParam(value="Be extra strict."),
        }
    )

    assert overrides["system_prompt"] == "Delegate when needed."
    assert overrides["supervisor_model"] == "gemini-2.5-mini"
    assert overrides["prompt_modification"] == "Be extra strict."


def test_resolve_agent_config_overrides_drops_non_string_prompt_overrides():
    overrides = resolve_agent_config_overrides(
        {
            "research_agent_prompt": _NonStringPromptObject(),
            "prompt_modification": _ValueParam(value="Respond in Italian."),
        }
    )

    assert "research_agent_prompt" not in overrides
    assert overrides["prompt_modification"] == "Respond in Italian."


def test_research_prompt_object_can_round_trip_into_agent_config():
    prompt = Prompt.from_prompt_data(
        "research_agent_prompt",
        PromptData(
            prompt=PromptCompletionBlock(
                content="Research in Italian and include sources."
            ),
            options={"model": "gpt-4o-mini"},
        ),
    )

    overrides = resolve_agent_config_overrides({"research_agent_prompt": prompt})
    config = AgentConfig.from_env(**overrides)

    assert config.research_agent_prompt == "Research in Italian and include sources."
    assert isinstance(config.research_agent_prompt, str)
