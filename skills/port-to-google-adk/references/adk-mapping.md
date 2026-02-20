# ADK Mapping

Map source orchestration semantics to ADK primitives.

## Common concept mapping

1. Source supervisor/root agent -> ADK root agent.
2. Source specialist subagents -> ADK sub agents or separate tool-enabled agents.
3. Source runner/graph invoke -> ADK `Runner.run` or `Runner.run_async`.
4. Source stream events -> ADK event stream from `run_async`.

## Tracing setup

1. Initialize Braintrust + ADK wrapper before runtime use.
2. Keep project name explicit so traces land in predictable projects.
3. Confirm root invocation span and nested agent/llm spans exist.

## Serialization strategy

1. Add user query as first message when missing.
2. Extract assistant text from final response events.
3. Convert function/tool call events to assistant `tool_calls`.
4. Convert tool results to `tool` role messages.

## Routing observability strategy

Prefer these sources in order:
1. Trace span names/attributes for agent/tool calls.
2. Serialized tool call names in output messages.
3. Fallback metadata attached to hooks if available.

## Practical warning

ADK is event-loop oriented. Do not assume one event equals one final answer. Always iterate until final response and keep intermediate tool-call context.
