import type {
  InferenceAdapter,
  InferenceCapabilities,
  InferenceRequest,
  InferenceResponse,
  RuntimeSelection,
} from "../types";

export type LocalInferenceHandler = (
  request: InferenceRequest,
  selection: RuntimeSelection,
) => Promise<InferenceResponse>;

export class LocalOfflineAdapter implements InferenceAdapter {
  readonly id = "local_offline";
  private handler: LocalInferenceHandler | null = null;

  setHandler(handler: LocalInferenceHandler): void {
    this.handler = handler;
  }

  canHandle(selection: RuntimeSelection, caps: InferenceCapabilities): boolean {
    return selection.mode !== "cloud" && caps.localLlmReady;
  }

  async run(request: InferenceRequest, selection: RuntimeSelection): Promise<InferenceResponse> {
    if (!this.handler) {
      throw new Error(
        "Local offline adapter is not wired to the native Gemma runtime yet. Install model packs and runtime plugin.",
      );
    }
    return this.handler(request, selection);
  }
}
