---
name: azure-observability
description: Azure observability workflows for logs, metrics, traces, and incident triage. Use when diagnosing production issues, querying telemetry, configuring alerts, or improving monitoring in Azure workloads.
---

# Azure Observability

This local compatibility skill preserves the requested `azure-observability` trigger name.

Use this with installed skills:
- `azure-diagnostics`
- `appinsights-instrumentation`

## Core Workflow

1. Verify service health and recent incidents.
2. Collect logs, traces, and metrics from the affected workload.
3. Correlate failures across app, infra, and network layers.
4. Identify likely root cause and blast radius.
5. Propose short-term mitigation and long-term prevention.

## Best Practices

- Start with impact and timeline before deep diagnostics.
- Prefer reproducible queries and measurable hypotheses.
- Separate symptoms from causes in incident summaries.
- Record follow-up actions for alerting and instrumentation gaps.
