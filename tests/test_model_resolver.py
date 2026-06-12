from src.config import AgentConfig
from src.model_resolver import make_wrapped_openai_client, resolve_adk_model


def test_resolve_adk_model_returns_string_when_gateway_disabled():
    config = AgentConfig(use_gateway=False)
    model = resolve_adk_model("gemini-2.0-flash-lite", config)
    assert model == "gemini-2.0-flash-lite"


def test_resolve_adk_model_returns_gemini_with_gateway_url_when_enabled():
    config = AgentConfig(
        use_gateway=True,
        gateway_url="https://gateway.braintrust.dev/v1",
        gateway_api_key="test-key",
    )
    model = resolve_adk_model("gemini-2.0-flash-lite", config)
    assert model.__class__.__name__ == "LiteLlm"
    assert getattr(model, "model") == "openai/gemini-2.0-flash-lite"
    assert getattr(model, "api_base") == "https://gateway.braintrust.dev/v1"


def test_resolve_adk_model_raises_if_gateway_key_missing():
    config = AgentConfig(use_gateway=True, gateway_api_key=None)
    try:
        resolve_adk_model("gemini-2.0-flash-lite", config)
    except RuntimeError as exc:
        assert "BRAINTRUST_GATEWAY_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when gateway key is missing")


def test_make_wrapped_openai_client_uses_gateway(monkeypatch):
    monkeypatch.setenv("BRAINTRUST_USE_GATEWAY", "true")
    monkeypatch.setenv("BRAINTRUST_GATEWAY_API_KEY", "gateway-key")
    monkeypatch.setenv("BRAINTRUST_GATEWAY_URL", "https://gateway.braintrust.dev/v1")
    monkeypatch.setenv("BRAINTRUST_PROJECT_ID", "proj_123")
    client = make_wrapped_openai_client()
    assert str(client.base_url).startswith("https://gateway.braintrust.dev/v1")
    assert client.default_headers["x-bt-parent"] == "project_id:proj_123"
    assert client.default_headers["x-bt-project-id"] == "proj_123"


def test_resolve_adk_model_includes_gateway_logging_headers(monkeypatch):
    monkeypatch.setenv("BRAINTRUST_PROJECT", "google-adk-supervisor")
    config = AgentConfig(
        use_gateway=True,
        gateway_url="https://gateway.braintrust.dev/v1",
        gateway_api_key="test-key",
    )
    model = resolve_adk_model("gemini-2.0-flash-lite", config)
    headers = getattr(model, "extra_headers")
    assert headers["x-bt-parent"] == "project_name:google-adk-supervisor"
    assert headers["x-bt-project-name"] == "google-adk-supervisor"
