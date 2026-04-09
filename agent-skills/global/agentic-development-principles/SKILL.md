---
name: agentic-development-principles
description: Foundational principles for building reliable agentic software workflows. Use before implementing multi-step AI-agent systems, tool-calling loops, evaluator feedback cycles, and autonomous task execution.
---

# Agentic Development Principles

This local compatibility skill preserves the requested `agentic-development-principles` trigger name.

## Principles

1. Define clear task boundaries, success criteria, and stop conditions.
2. Keep tools minimal and explicit; grant least privilege.
3. Use deterministic checkpoints for state, artifacts, and logs.
4. Separate planning, execution, and evaluation into distinct steps.
5. Add verification gates before claiming completion.
6. Handle retries with bounded backoff and idempotent operations.
7. Preserve auditability with traceable prompts, inputs, and outputs.
8. Escalate to human review on ambiguity, safety, or high-impact changes.

## Execution Pattern

- Plan: generate a concise, testable plan.
- Act: execute one bounded step at a time.
- Evaluate: check output quality against explicit rubric.
- Refine: loop only when evaluation fails; otherwise finalize.
