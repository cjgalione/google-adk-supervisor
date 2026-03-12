from pydantic import BaseModel, Field

from evals.parameter_utils import (
    get_hook_parameters,
    unwrap_parameter_value,
    unwrap_parameters,
)


class _ValueParam(BaseModel):
    value: str = Field(default="default-value")


class _MultiFieldParam(BaseModel):
    a: int = 1
    b: int = 2


class _Hooks:
    def __init__(self, parameters):
        self.parameters = parameters


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
