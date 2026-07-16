from scripts import run_queries


def test_preflight_classifies_auth_quota_and_transient_failures() -> None:
    assert run_queries._preflight_failure_category(RuntimeError("invalid API key")) == "authentication"
    assert run_queries._preflight_failure_category(RuntimeError("GenerateRequestsPerDay")) == "quota"
    assert run_queries._preflight_failure_category(RuntimeError("error code 429")) == "transient"


def test_preflight_redacts_provider_exception_details(monkeypatch) -> None:
    monkeypatch.setenv("BRAINTRUST_API_KEY", "test-braintrust")
    monkeypatch.setenv("EXA_API_KEY", "test-exa")
    monkeypatch.setattr(
        run_queries,
        "_generate_model_text",
        lambda _prompt: (_ for _ in ()).throw(RuntimeError("invalid API key sk-test-secret")),
    )

    try:
        run_queries._run_preflight()
    except RuntimeError as exc:
        assert str(exc) == "Provider preflight failed (authentication)."
        assert "sk-test-secret" not in str(exc)
    else:
        raise AssertionError("Expected preflight failure")
