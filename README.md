# Google ADK Supervisor

A multi-agent supervisor system built with Google ADK that routes user tasks between:

- `ResearchAgent` (Tavily-backed web search)
- `MathAgent` (arithmetic tools)
- `Supervisor Agent` (routing + synthesis)

This repository is a near-structure translation of the `openai-agent-sdk-supervisor` project, preserving:

- eval task/scorer layout
- Braintrust remote eval server deployment pattern
- configurable prompts/models through eval parameters

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment:

```bash
cp .env.example .env
```

Required keys:

- `GOOGLE_API_KEY`
- `TAVILY_API_KEY`
- `BRAINTRUST_API_KEY` (if tracing/evals)
- `OPENAI_API_KEY` (used by judge scorers and question generation scripts)

3. Run local chat:

```bash
python -m src.local_runner
```

## Evals

Run full supervisor eval:

```bash
braintrust eval evals/eval_supervisor.py
```

Run focused eval suites:

```bash
braintrust eval evals/eval_math_agent.py
braintrust eval evals/eval_research_agent.py
```

## Remote Eval Server (Modal)

Deploy:

```bash
modal deploy src/eval_server.py
```

Local serve:

```bash
modal serve src/eval_server.py
```

Then connect the endpoint from Braintrust Playground remote eval UI.

## Project Layout

- `src/agents/` - supervisor + specialist agent construction
- `src/helpers.py` - ADK run loop + event serialization into eval message schema
- `evals/` - Braintrust eval tasks and scorers
- `src/eval_server.py` - Modal ASGI remote eval server
- `scorers.py` - reusable published scorers

## Notes

- The output contract remains `{"messages": [...]}` for scorer compatibility.
- Routing inference relies on span/tool-call names (`research`, `math`, `delegate_to_*`, `tavily_search`, arithmetic tool names).
