---
name: port-to-google-adk
description: Port an existing multi-agent or supervisor-style project to Google ADK while preserving Braintrust traces, scorer compatibility, and remote eval behavior. Use when execution semantics move from framework-specific runners/graphs to ADK event loops and you need per-agent/tool observability in Braintrust.
---

# Port To Google Adk

## Overview

Use this skill after applying `$agent-eval-porting-core`. Implement only ADK-specific mapping: agent topology, run loop handling, tracing setup, and event serialization.

## Load References First

1. Load `/Users/curtisjgalione/git/agent-eval-porting-core/references/output-contract.md`.
2. Load `/Users/curtisjgalione/git/google-adk-supervisor/skills/port-to-google-adk/references/adk-mapping.md`.
3. Use `/Users/curtisjgalione/git/google-adk-supervisor/skills/port-to-google-adk/references/adk-task-template.md` when writing the eval task body.

## Workflow

1. Create ADK agents mirroring source roles.
Map supervisor/specialists to ADK agent types (`LlmAgent`, `SequentialAgent`, `ParallelAgent`) while preserving role names used by scorers.

2. Initialize Braintrust ADK tracing at startup.
Use `setup_adk(...)` before creating/using runners so invocation, agent, and LLM spans are captured.

3. Port the eval task function.
Replace source runtime call with ADK `Runner.run_async(...)` event consumption. Build `{"messages": [...]}` payload from ADK events.

4. Keep scorer compatibility.
Preserve tool/handoff naming in serialized messages or spans so routing scorers can infer called specialists.

5. Keep remote eval server behavior stable.
Reuse existing Braintrust evaluator-loading/ASGI patterns unless user asks for a deployment change.

6. Validate parity.
Run local eval, then remote eval. Confirm the following before completion:
- Similar routing decisions to source framework.
- Non-empty per-agent or per-tool trace spans.
- Stable scorer outputs.

## Implementation Notes

1. Treat ADK events as stream data, not a single final object.
2. Always capture final response text plus tool-call events.
3. Preserve agent names (`Supervisor`, `MathAgent`, `ResearchAgent`, etc.) when possible to reduce scorer churn.

## Escalation Rule

If a source project uses custom routing semantics not represented cleanly by ADK abstractions, document the behavior gap explicitly and propose scorer updates instead of hiding the mismatch.
