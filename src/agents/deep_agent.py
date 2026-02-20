"""Google ADK supervisor with delegation tools for specialized subtasks."""

from __future__ import annotations

from google.adk import Agent

from src.agents.math_agent import add, divide, get_math_agent, multiply, subtract
from src.agents.research_agent import get_research_agent
from src.config import AgentConfig
from src.helpers import run_adk_agent


_MATH_OPS = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
}


def _run_math(operation: str, a: float, b: float) -> float:
    op = operation.strip().lower()
    if op not in _MATH_OPS:
        raise ValueError(f"Unsupported math operation: {operation}")
    return float(_MATH_OPS[op](a, b))


def get_deep_agent(config: AgentConfig | None = None) -> Agent:
    """Create the supervisor agent and wire delegation tools."""
    resolved_config = config or AgentConfig()
    supervisor_prompt = resolved_config.render_supervisor_prompt()

    research_agent: Agent | None = None
    math_agent: Agent | None = None

    async def request_research_subtask(query: str, max_results: int = 3) -> str:
        """Request research before completing a downstream math subtask."""
        del max_results
        if research_agent is None:
            raise RuntimeError("ResearchAgent is not initialized")
        result = await run_adk_agent(
            agent=research_agent,
            query=query,
            app_name="google-adk-supervisor-research-subtask",
        )
        return str(result.get("final_output", "")).strip()

    def request_math_subtask(operation: str, a: float, b: float) -> float:
        """Request a math subtask during compound research + calculation workflows."""
        return _run_math(operation=operation, a=a, b=b)

    research_agent = get_research_agent(
        system_prompt=resolved_config.research_agent_prompt,
        model=resolved_config.research_model,
        extra_tools=[request_math_subtask],
    )
    math_agent = get_math_agent(
        system_prompt=resolved_config.math_agent_prompt,
        model=resolved_config.math_model,
        extra_tools=[request_research_subtask],
    )

    async def delegate_to_research_agent(query: str, max_results: int = 3) -> str:
        """Delegate a factual lookup or web-research task to ResearchAgent."""
        del max_results
        result = await run_adk_agent(
            agent=research_agent,
            query=query,
            app_name="google-adk-supervisor-delegate-research",
        )
        return str(result.get("final_output", "")).strip()

    def delegate_to_math_agent(operation: str, a: float, b: float) -> float:
        """Delegate a numeric computation to MathAgent."""
        return _run_math(operation=operation, a=a, b=b)

    return Agent(
        name="SupervisorAgent",
        model=resolved_config.supervisor_model,
        instruction=supervisor_prompt,
        tools=[
            delegate_to_research_agent,
            request_research_subtask,
            delegate_to_math_agent,
            request_math_subtask,
        ],
    )


_cached_deep_agent: Agent | None = None


def get_supervisor(config: AgentConfig | None = None, force_rebuild: bool = False) -> Agent:
    """Get a cached or newly built supervisor agent."""
    global _cached_deep_agent

    if config is not None:
        return get_deep_agent(config)

    if force_rebuild or _cached_deep_agent is None:
        _cached_deep_agent = get_deep_agent()
    return _cached_deep_agent
