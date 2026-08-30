from src.config import AgentConfig
from src.model_resolver import make_wrapped_openai_client, resolve_adk_model


def test_resolve_adk_model_returns_string_when_gateway_disabled():
    config = AgentConfig(use_gateway=False)
    model = resolve_adk_model("gemini-2.0-flash-lite", config)
    assert model == "gemini-2.0-flash-lite"


def test_resolve_adk_model_returns_gemini_with_gateway_url_when_enabled(monkeypatch):
    captured = {}

    def fake_lite_llm(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("google.adk.models.lite_llm.LiteLlm", fake_lite_llm)
    config = AgentConfig(
        use_gateway=True,
        gateway_url="https://gateway.braintrust.dev",
        gateway_api_key="test-key",
    )
    resolve_adk_model("gemini-2.0-flash-lite", config)
    assert captured["model"] == "openai/gemini-2.0-flash-lite"
    assert captured["api_key"] == "test-key"
    assert captured["api_base"] == "https://gateway.braintrust.dev"


def test_resolve_adk_model_raises_if_gateway_key_missing(monkeypatch):
    monkeypatch.delenv("BRAINTRUST_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
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
    monkeypatch.setenv("BRAINTRUST_GATEWAY_URL", "https://gateway.braintrust.dev")
    monkeypatch.setenv("BRAINTRUST_PROJECT_ID", "proj_123")
    client = make_wrapped_openai_client()
    assert str(client.base_url).startswith("https://gateway.braintrust.dev")
    assert client.default_headers["x-bt-parent"] == "project_id:proj_123"
    assert client.default_headers["x-bt-project-id"] == "proj_123"


def test_resolve_adk_model_includes_gateway_logging_headers(monkeypatch):
    captured = {}

    def fake_lite_llm(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("google.adk.models.lite_llm.LiteLlm", fake_lite_llm)
    monkeypatch.setenv("BRAINTRUST_PROJECT", "google-adk-supervisor")
    config = AgentConfig(
        use_gateway=True,
        gateway_url="https://gateway.braintrust.dev",
        gateway_api_key="test-key",
    )
    resolve_adk_model("gemini-2.0-flash-lite", config)
    headers = captured["extra_headers"]
    assert headers["x-bt-parent"] == "project_name:google-adk-supervisor"
    assert headers["x-bt-project-name"] == "google-adk-supervisor"
