"""Tests for recovering malformed supervisor math delegation calls."""

from __future__ import annotations

import pytest

from src.agents import deep_agent


class _NoopSpan:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def log(self, **kwargs):
        pass


class _FakeSupervisor:
    pass


@pytest.mark.asyncio
async def test_run_supervisor_recovers_missing_math_task(monkeypatch):
    async def failing_run_adk_agent(**kwargs):
        raise ValueError("Provide a non-empty math_task (or operation alias).")

    async def recover(query, exc):
        assert query == "What is the product of 15 and 18?"
        assert deep_agent._is_missing_math_task_error(exc)
        return {
            "final_output": "15 times 18 is 270.",
            "messages": [
                {"role": "user", "content": query},
                {"role": "system", "content": "handoff marker: handoff [MathAgent]"},
                {"role": "assistant", "content": "15 times 18 is 270."},
            ],
        }

    supervisor = _FakeSupervisor()
    supervisor._recover_math_delegation_error = recover

    monkeypatch.setattr(deep_agent, "start_span", lambda **kwargs: _NoopSpan())
    monkeypatch.setattr(deep_agent, "run_adk_agent", failing_run_adk_agent)

    result = await deep_agent.run_supervisor_with_critic(
        supervisor=supervisor,  # type: ignore[arg-type]
        query="What is the product of 15 and 18?",
        app_name="test-app",
    )

    assert result["final_output"] == "15 times 18 is 270."
    assert result["critic_decision"]["required_action"] == "accept"
    assert result["critic_corrected"] is False


@pytest.mark.asyncio
async def test_run_supervisor_does_not_recover_non_math_query(monkeypatch):
    async def failing_run_adk_agent(**kwargs):
        raise ValueError("Provide a non-empty math_task (or operation alias).")

    async def recover(query, exc):
        raise AssertionError("recovery should not be called")

    supervisor = _FakeSupervisor()
    supervisor._recover_math_delegation_error = recover

    monkeypatch.setattr(deep_agent, "start_span", lambda **kwargs: _NoopSpan())
    monkeypatch.setattr(deep_agent, "run_adk_agent", failing_run_adk_agent)

    with pytest.raises(ValueError, match="non-empty math_task"):
        await deep_agent.run_supervisor_with_critic(
            supervisor=supervisor,  # type: ignore[arg-type]
            query="Hello there",
            app_name="test-app",
        )
