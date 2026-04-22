import { expect, Page, test } from "@playwright/test";

interface RouteHarness {
  cancelRequests: Array<{ url: string; body: unknown }>;
}

const runtimeTaskRunning = {
  task_id: "coord-trace-e2e-001",
  parent_task_id: null,
  state: "running",
  permission_scope: null,
  child_task_ids: [],
  created_at: "2026-04-05T10:00:00+00:00",
  updated_at: "2026-04-05T10:00:00+00:00",
  metadata: {
    trace_id: "trace-e2e-001",
  },
};

const runtimeTaskCompleted = {
  ...runtimeTaskRunning,
  state: "completed",
  updated_at: "2026-04-05T10:01:00+00:00",
};

const runtimeTaskRefs = {
  trace_id: "trace-e2e-001",
  coordinator_task_id: runtimeTaskRunning.task_id,
  subagent_task_ids: [],
  all_task_ids: [runtimeTaskRunning.task_id],
  api: {
    list: "/api/runtime/tasks",
    coordinator: `/api/runtime/tasks/${runtimeTaskRunning.task_id}`,
    cancel_coordinator: `/api/runtime/tasks/${runtimeTaskRunning.task_id}/cancel`,
    subagents: [],
  },
};

async function installRuntimeRouteHarness(page: Page): Promise<RouteHarness> {
  const cancelRequests: Array<{ url: string; body: unknown }> = [];

  await page.route("**/api/health", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "ok",
        model_loaded: true,
        model_info: {
          quantization: "Q4",
          device: "CPU",
          gemini_available: false,
          fine_tuned: false,
          training_stages_completed: 0,
        },
      }),
    });
  });

  await page.route("**/api/llm/provider", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        provider: "local",
        available: ["local"],
        gemini_configured: false,
        local_model_loaded: true,
      }),
    });
  });

  await page.route("**/api/runtime/safety/permissions", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        count: 0,
        pending: [],
        expired_count: 0,
      }),
    });
  });

  await page.route("**/api/runtime/safety/executor", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        enabled: true,
        running: true,
        summary: {
          approved_total: 0,
          pending_total: 0,
          running: 0,
          waiting_retry: 0,
          completed: 0,
          failed: 0,
          unsupported: 0,
          idle: 0,
        },
      }),
    });
  });

  await page.route("**/api/rag/traces**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        traces: [],
        analytics: {
          total_traces: 0,
          showing: 0,
          avg_duration_ms: 0,
          avg_confidence: 0,
          avg_evidence_count: 0,
          channel_usage: {},
          step_stats: {},
          crag_activation_rate: 0,
          selfrag_activation_rate: 0,
          flare_activation_rate: 0,
          cache_hit_rate: 0,
          stop_reason_distribution: {},
          runtime_loop: {
            avg_iterations: 0,
            avg_tool_calls: 0,
          },
        },
      }),
    });
  });

  await page.route("**/rag/observability/metrics", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        total_queries: 1,
        avg_pipeline_ms: 200,
        total_steps_executed: 6,
        compression_invocations: 0,
        cache: {
          total_hits: 0,
          total_queries: 1,
          hit_rate: 0,
        },
      }),
    });
  });

  await page.route("**/api/runtime/tasks", async (route) => {
    if (route.request().method() !== "GET") {
      await route.fallback();
      return;
    }

    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        count: 1,
        tasks: [runtimeTaskRunning],
      }),
    });
  });

  await page.route("**/runtime/tasks/events", async (route) => {
    const eventPayload = {
      event_id: "task-evt-001",
      sequence: 1,
      event_type: "task_transition",
      timestamp: "2026-04-05T10:01:00+00:00",
      task: runtimeTaskCompleted,
      previous_state: "running",
      state: "completed",
      note: "execution completed",
    };

    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: `data: ${JSON.stringify(eventPayload)}\n\n`,
    });
  });

  await page.route("**/rag/pipeline-events", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: ": keepalive\n\n",
    });
  });

  await page.route("**/rag/chat", async (route) => {
    const request = route.request();
    if (request.method() !== "POST") {
      await route.fallback();
      return;
    }

    const streamBody = [
      `data: ${JSON.stringify({ rag_meta: { runtime_tasks: runtimeTaskRefs } })}`,
      `data: ${JSON.stringify({ delta: "Working on runtime coordination." })}`,
      `data: ${JSON.stringify({ done: true })}`,
      "",
    ].join("\n\n");

    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: streamBody,
    });
  });

  await page.route("**/api/runtime/tasks/**/cancel", async (route) => {
    const bodyText = route.request().postData() || "{}";
    let parsedBody: unknown = {};
    try {
      parsedBody = JSON.parse(bodyText);
    } catch {
      parsedBody = bodyText;
    }

    cancelRequests.push({
      url: route.request().url(),
      body: parsedBody,
    });

    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        cancelled_task_ids: [runtimeTaskRunning.task_id],
      }),
    });
  });

  return { cancelRequests };
}

test.describe("Runtime task lifecycle UI", () => {
  test("shows SSE lifecycle transitions in observability", async ({ page }) => {
    await installRuntimeRouteHarness(page);

    await page.goto("/");

    await page.getByRole("button", { name: "Pipeline Observability" }).click();

    await expect(page.getByTestId("observability-runtime-task-panel")).toBeVisible();
    await expect(page.getByTestId("observability-task-stream-status")).toHaveText("LIVE");
    await expect(page.getByTestId("observability-task-notification").first()).toContainText("running -> completed");
  });

  test("shows runtime strip and sends cancel action from chat", async ({ page }) => {
    const harness = await installRuntimeRouteHarness(page);

    await page.goto("/");

    const input = page.getByPlaceholder("Ask Cortex Lab anything…");
    await input.fill("Please coordinate a multi-agent answer");
    await input.press("Enter");

    await expect(page.getByTestId("chat-runtime-task-strip")).toBeVisible();

    await page.getByTestId(`chat-runtime-task-cancel-${runtimeTaskRunning.task_id}`).click();

    await expect.poll(() => harness.cancelRequests.length).toBe(1);

    expect(harness.cancelRequests[0]?.url).toContain(`/api/runtime/tasks/${runtimeTaskRunning.task_id}/cancel`);
    expect(harness.cancelRequests[0]?.body).toEqual({
      reason: "Cancelled from main chat runtime strip",
      propagate: true,
    });
  });
});
