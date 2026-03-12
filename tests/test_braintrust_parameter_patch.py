from pydantic import BaseModel, Field

from evals import braintrust_parameter_patch as patch_mod


class _SingleFieldParam(BaseModel):
    value: str = Field(default="hello", description="param description")


def test_patched_parameters_to_json_schema_unwraps_single_field_model():
    schema = patch_mod.patched_parameters_to_json_schema({"x": _SingleFieldParam})
    assert schema["x"]["type"] == "data"
    assert schema["x"]["schema"]["type"] == "string"
    assert "properties" not in schema["x"]["schema"]
    assert schema["x"]["default"] == "hello"
    assert schema["x"]["description"] == "param description"


def test_apply_parameter_patch_skips_when_native_is_compatible(monkeypatch):
    import braintrust.parameters as params_module

    original = params_module.parameters_to_json_schema
    monkeypatch.setattr(
        patch_mod,
        "_is_native_parameter_schema_compatible",
        lambda: True,
    )
    assert patch_mod.apply_parameter_patch(verbose=False) is True
    assert params_module.parameters_to_json_schema is original


def test_apply_parameter_patch_overrides_when_native_is_incompatible(monkeypatch):
    import braintrust.parameters as params_module

    original = params_module.parameters_to_json_schema
    monkeypatch.setattr(
        patch_mod,
        "_is_native_parameter_schema_compatible",
        lambda: False,
    )
    assert patch_mod.apply_parameter_patch(verbose=False) is True
    assert params_module.parameters_to_json_schema is patch_mod.patched_parameters_to_json_schema

    # Restore function for test isolation.
    monkeypatch.setattr(params_module, "parameters_to_json_schema", original)
