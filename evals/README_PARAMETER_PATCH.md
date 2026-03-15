# Braintrust Parameter Compatibility

## Why this exists

Some Braintrust Python SDK versions serialize single-field Pydantic parameters
as object schemas, which can break Playground defaults and form validation for
simple scalar parameters.

This repository uses a compatibility shim in
`evals/braintrust_parameter_patch.py` that:

1. Probes the installed SDK behavior at runtime.
2. Skips patching when native behavior is already compatible.
3. Applies a local shim only when needed.

## Stable parameter contract

Parameter definitions live in `evals/parameters.py` and are shared across all
eval suites. Prompt-bearing fields use Braintrust's native `type="prompt"`
schema so the Playground shows a real prompt editor with embedded model
settings.

- `SUPERVISOR_EVAL_PARAMETERS`
- `RESEARCH_EVAL_PARAMETERS`
- `MATH_EVAL_PARAMETERS`

Parameter values are normalized by `evals/parameter_utils.py`.

## Where compatibility patching is applied

- `evals/eval_supervisor.py`
- `evals/eval_research_agent.py`
- `evals/eval_math_agent.py`
- `src/eval_server.py` (before loading evaluators)

## Verifying behavior

Run:

```bash
braintrust eval evals/eval_supervisor.py --no-send-logs
```

Expected startup logs:

- `✓ Braintrust parameter schema is natively compatible; patch skipped`
or
- `✓ Applied Braintrust parameter compatibility patch`

Either message indicates the parameter serialization path is configured.
