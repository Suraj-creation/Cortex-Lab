import { CloudAdapter } from "./adapters/cloudAdapter";
import { LocalOfflineAdapter } from "./adapters/localOfflineAdapter";
import type {
  InferenceCapabilities,
  InferenceRequest,
  InferenceResponse,
  RuntimeSelection,
} from "./types";

export class InferenceRouter {
  constructor(
    private readonly cloudAdapter: CloudAdapter,
    private readonly localAdapter: LocalOfflineAdapter,
  ) {}

  resolveSelection(
    selection: RuntimeSelection,
    caps: InferenceCapabilities,
  ): RuntimeSelection {
    if (selection.mode === "local_offline") {
      return {
        ...selection,
        allowCloudFallback: false,
      };
    }

    if (selection.mode === "hybrid" && caps.localLlmReady) {
      return selection;
    }

    if (selection.mode === "hybrid" && selection.allowCloudFallback) {
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
    caps: InferenceCapabilities,
  ): Promise<InferenceResponse> {
    const resolved = this.resolveSelection(selection, caps);

    if (this.localAdapter.canHandle(resolved, caps) && resolved.mode !== "cloud") {
      return this.localAdapter.run(request, resolved);
    }

    if (resolved.mode === "local_offline") {
      throw new Error("Offline mode requested but local inference runtime is unavailable.");
    }

    if (!this.cloudAdapter.canHandle(resolved, caps)) {
      throw new Error("Cloud inference is unavailable and no local fallback is ready.");
    }

    return this.cloudAdapter.run(request, resolved);
  }
}
