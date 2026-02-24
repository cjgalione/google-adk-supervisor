"""Google ADK supervisor with delegation tools for specialized subtasks."""

from __future__ import annotations

import re

from braintrust import SpanTypeAttribute, start_span
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


def _parse_number_token(token: str) -> float | None:
    cleaned = token.strip().lower().replace(",", "")
    if "^" in cleaned and "e" not in cleaned:
        base, exp = cleaned.split("^", 1)
        try:
            return float(base) ** float(exp)
        except ValueError:
            return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_conversion_operation(operation: str) -> tuple[float, str, str] | None:
    text = operation.strip()
    m = re.match(r"(?i)^convert\s+([^\s]+)\s+(.+?)\s+to\s+(.+)$", text)
    if not m:
        return None
    value_token, from_unit, to_unit = m.groups()
    value = _parse_number_token(value_token)
    if value is None:
        return None
    return value, from_unit.strip(), to_unit.strip()


def _is_basic_math_operation(operation: str) -> bool:
    return operation.strip().lower() in _MATH_OPS


def _classify_math_operation(operation: str) -> str:
    if _is_basic_math_operation(operation):
        return "arithmetic"
    if _parse_conversion_operation(operation) is not None:
        return "unit_conversion"
    return "other"


def _build_math_query(operation: str, a: float, b: float) -> str:
    if _is_basic_math_operation(operation):
        return (
            f"Use operation '{operation}' on the values a={a} and b={b}. "
            "Return the final numeric result."
        )

    conversion = _parse_conversion_operation(operation)
    if conversion is not None:
        value, from_unit, to_unit = conversion
        return (
            "Use unit conversion for this task and return only the final numeric result. "
            f"Convert value={value} from_unit='{from_unit}' to_unit='{to_unit}'."
        )

    return (
        "Solve the following quantitative task and return the final numeric result. "
        f"Task: {operation}. "
        f"Context values (if useful): a={a}, b={b}."
    )


def _run_math(operation: str, a: float, b: float) -> float:
    op = operation.strip().lower()
    if op not in _MATH_OPS:
        raise ValueError(f"Unsupported math operation: {operation}")
    return float(_MATH_OPS[op](a, b))


def _extract_float_from_text(text: str) -> float | None:
    sci_caret_matches = re.findall(r"(-?\d+(?:\.\d+)?)\s*[x×]\s*10\^(-?\d+)", text, flags=re.IGNORECASE)
    if sci_caret_matches:
        base_s, exp_s = sci_caret_matches[-1]
        try:
            return float(base_s) * (10 ** int(exp_s))
        except ValueError:
            pass

    matches = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _handoff_span_metadata(*, target: str, input_data: dict[str, object]) -> dict[str, object]:
    return {
        "target_agent": target,
        "handoff_input": input_data,
    }


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
        with start_span(
            name="handoff [ResearchAgent]",
            type=SpanTypeAttribute.TASK,
            input={"query": query},
            metadata=_handoff_span_metadata(
                target="ResearchAgent",
                input_data={"query": query, "mode": "subtask"},
            ),
        ) as handoff_span:
            result = await run_adk_agent(
                agent=research_agent,
                query=query,
                app_name="google-adk-supervisor-research-subtask",
            )
            final_output = str(result.get("final_output", "")).strip()
            handoff_span.log(output={"final_output": final_output, "messages": result.get("messages", [])})
            return final_output

    async def request_math_subtask(operation: str, a: float, b: float) -> float:
        """Request a math subtask during compound research + calculation workflows."""
        if math_agent is None:
            raise RuntimeError("MathAgent is not initialized")

        query = _build_math_query(operation=operation, a=a, b=b)
        with start_span(
            name="handoff [MathAgent]",
            type=SpanTypeAttribute.TASK,
            input={"operation": operation, "a": a, "b": b},
            metadata=_handoff_span_metadata(
                target="MathAgent",
                input_data={
                    "operation": operation,
                    "operation_type": _classify_math_operation(operation),
                    "a": a,
                    "b": b,
                    "mode": "subtask",
                },
            ),
        ) as handoff_span:
            result = await run_adk_agent(
                agent=math_agent,
                query=query,
                app_name="google-adk-supervisor-math-subtask",
            )
            final_text = str(result.get("final_output", "")).strip()
            parsed = _extract_float_from_text(final_text)
            if parsed is not None:
                parsed_result = parsed
            elif _is_basic_math_operation(operation):
                parsed_result = _run_math(operation=operation, a=a, b=b)
            else:
                raise ValueError(
                    f"MathAgent did not return a numeric result for operation '{operation}'. "
                    f"Model output: {final_text}"
                )
            handoff_span.log(
                output={
                    "final_output": final_text,
                    "parsed_result": parsed_result,
                    "messages": result.get("messages", []),
                }
            )
            return parsed_result

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
        with start_span(
            name="handoff [ResearchAgent]",
            type=SpanTypeAttribute.TASK,
            input={"query": query},
            metadata=_handoff_span_metadata(
                target="ResearchAgent",
                input_data={"query": query, "mode": "delegate"},
            ),
        ) as handoff_span:
            result = await run_adk_agent(
                agent=research_agent,
                query=query,
                app_name="google-adk-supervisor-delegate-research",
            )
            final_output = str(result.get("final_output", "")).strip()
            handoff_span.log(output={"final_output": final_output, "messages": result.get("messages", [])})
            return final_output

    async def delegate_to_math_agent(operation: str, a: float, b: float) -> float:
        """Delegate a numeric computation to MathAgent."""
        if math_agent is None:
            raise RuntimeError("MathAgent is not initialized")

        query = _build_math_query(operation=operation, a=a, b=b)
        with start_span(
            name="handoff [MathAgent]",
            type=SpanTypeAttribute.TASK,
            input={"operation": operation, "a": a, "b": b},
            metadata=_handoff_span_metadata(
                target="MathAgent",
                input_data={
                    "operation": operation,
                    "operation_type": _classify_math_operation(operation),
                    "a": a,
                    "b": b,
                    "mode": "delegate",
                },
            ),
        ) as handoff_span:
            result = await run_adk_agent(
                agent=math_agent,
                query=query,
                app_name="google-adk-supervisor-delegate-math",
            )
            final_text = str(result.get("final_output", "")).strip()
            parsed = _extract_float_from_text(final_text)
            if parsed is not None:
                parsed_result = parsed
            elif _is_basic_math_operation(operation):
                parsed_result = _run_math(operation=operation, a=a, b=b)
            else:
                raise ValueError(
                    f"MathAgent did not return a numeric result for operation '{operation}'. "
                    f"Model output: {final_text}"
                )
            handoff_span.log(
                output={
                    "final_output": final_text,
                    "parsed_result": parsed_result,
                    "messages": result.get("messages", []),
                }
            )
            return parsed_result

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
