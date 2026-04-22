import { CloudAdapter } from "./adapters/cloudAdapter";
import { LocalAdapter } from "./adapters/localAdapter";
import type {
  InferenceRequest,
  InferenceResponse,
  RuntimeCapabilities,
  RuntimeSelection,
} from "./types";

export class InferenceRouter {
  constructor(
    private readonly cloudAdapter: CloudAdapter,
    private readonly localAdapter: LocalAdapter,
  ) {}

  resolve(selection: RuntimeSelection, caps: RuntimeCapabilities): RuntimeSelection {
    if (selection.mode === "local_offline") {
      return {
        ...selection,
        allowCloudFallback: false,
      };
    }

    if (selection.mode === "hybrid" && !caps.localLlmReady && selection.allowCloudFallback) {
      return {
        ...selection,
        mode: "cloud",
        llmProvider: "gemini",
      };
    }

    return selection;
  }

  async run(
    request: InferenceRequest,
    selection: RuntimeSelection,
    caps: RuntimeCapabilities,
  ): Promise<InferenceResponse> {
    const resolved = this.resolve(selection, caps);

    if (this.localAdapter.canHandle(resolved, caps) && resolved.mode !== "cloud") {
      return this.localAdapter.run(request, resolved);
    }

    if (resolved.mode === "local_offline") {
      throw new Error("Offline mode requested but browser local runtime is unavailable.");
    }

    if (!this.cloudAdapter.canHandle(resolved, caps)) {
      throw new Error("Cloud path unavailable and no local runtime fallback is ready.");
    }

    return this.cloudAdapter.run(request, resolved);
  }
}
