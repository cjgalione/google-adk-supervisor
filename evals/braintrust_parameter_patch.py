"""Compatibility shim for Braintrust parameter JSON schema serialization."""

from __future__ import annotations

import sys
from typing import Any

from pydantic import BaseModel, Field


def _pydantic_to_json_schema(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    if hasattr(model, "schema"):
        return model.schema()
    raise ValueError(f"Cannot convert {model} to JSON schema - not a pydantic model")


def _get_pydantic_field_info(model_class: Any, field_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {}

    if hasattr(model_class, "model_fields"):
        field_info = model_class.model_fields.get(field_name)
        if field_info:
            if hasattr(field_info, "default") and field_info.default is not None:
                result["default"] = field_info.default
            elif hasattr(field_info, "default_factory") and field_info.default_factory:
                try:
                    result["default"] = field_info.default_factory()
                except Exception:
                    pass
            if hasattr(field_info, "description") and field_info.description:
                result["description"] = field_info.description
        return result

    if hasattr(model_class, "__fields__"):
        field_info = model_class.__fields__.get(field_name)
        if field_info:
            if hasattr(field_info, "default") and field_info.default is not None:
                result["default"] = field_info.default
            elif hasattr(field_info, "default_factory") and field_info.default_factory:
                try:
                    result["default"] = field_info.default_factory()
                except Exception:
                    pass
            if hasattr(field_info, "field_info") and hasattr(
                field_info.field_info, "description"
            ):
                if field_info.field_info.description:
                    result["description"] = field_info.field_info.description
    return result


def patched_parameters_to_json_schema(parameters: dict[str, Any]) -> dict[str, Any]:
    """Convert EvalParameters to JSON schema with single-field model unwrapping."""
    result: dict[str, Any] = {}

    for name, schema in parameters.items():
        if isinstance(schema, dict) and schema.get("type") == "prompt":
            result[name] = {
                "type": "prompt",
                "default": schema.get("default"),
                "description": schema.get("description"),
            }
            continue

        try:
            fields = getattr(schema, "__fields__", None) or getattr(
                schema, "model_fields", {}
            )
            if len(fields) == 1:
                field_name = list(fields.keys())[0]
                full_schema = _pydantic_to_json_schema(schema)
                if (
                    "properties" in full_schema
                    and field_name in full_schema["properties"]
                ):
                    field_schema = full_schema["properties"][field_name]
                    field_info = _get_pydantic_field_info(schema, field_name)
                    result[name] = {"type": "data", "schema": field_schema}
                    if "default" in field_info:
                        result[name]["default"] = field_info["default"]
                    if "description" in field_info:
                        result[name]["description"] = field_info["description"]
                    continue

            result[name] = {
                "type": "data",
                "schema": _pydantic_to_json_schema(schema),
            }
        except (ValueError, AttributeError):
            pass

    return result


def _is_native_parameter_schema_compatible() -> bool:
    """Check whether the installed Braintrust SDK already unwraps single-field models."""

    class _ProbeParameter(BaseModel):
        value: str = Field(default="probe-default", description="probe-description")

    try:
        import braintrust.parameters as params_module

        schema = params_module.parameters_to_json_schema({"probe": _ProbeParameter})
    except Exception:
        return False

    probe = schema.get("probe")
    if not isinstance(probe, dict):
        return False

    if probe.get("type") != "data":
        return False

    probe_schema = probe.get("schema")
    if not isinstance(probe_schema, dict):
        return False

    if probe_schema.get("type") != "string":
        return False

    if "properties" in probe_schema:
        return False

    return probe.get("default") == "probe-default" and probe.get(
        "description"
    ) == "probe-description"


def apply_parameter_patch(verbose: bool = True) -> bool:
    """Patch Braintrust only when native SDK behavior is not compatible."""
    try:
        import braintrust.parameters as params_module
    except ImportError as exc:
        if verbose:
            print(f"Failed to import Braintrust parameters module: {exc}")
        return False

    if _is_native_parameter_schema_compatible():
        if verbose:
            print("Braintrust parameter schema is natively compatible; patch skipped")
        return True

    try:
        params_module.parameters_to_json_schema = patched_parameters_to_json_schema
        for module_name, module in list(sys.modules.items()):
            if "braintrust" in module_name and hasattr(
                module, "parameters_to_json_schema"
            ):
                module.parameters_to_json_schema = patched_parameters_to_json_schema  # type: ignore[attr-defined]
        if verbose:
            print("Applied Braintrust parameter compatibility patch")
        return True
    except Exception as exc:
        if verbose:
            print(f"Failed to apply Braintrust parameter patch: {exc}")
        return False
