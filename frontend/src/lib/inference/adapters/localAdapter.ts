import type {
  InferenceAdapter,
  InferenceRequest,
  InferenceResponse,
  RuntimeCapabilities,
  RuntimeSelection,
} from "../types";

export type BrowserLocalHandler = (
  request: InferenceRequest,
  selection: RuntimeSelection,
) => Promise<InferenceResponse>;

export class LocalAdapter implements InferenceAdapter {
  readonly id = "local";
  private handler: BrowserLocalHandler | null = null;

  registerHandler(handler: BrowserLocalHandler): void {
    this.handler = handler;
  }

  canHandle(selection: RuntimeSelection, caps: RuntimeCapabilities): boolean {
    return selection.mode !== "cloud" && caps.localLlmReady;
  }

  async run(request: InferenceRequest, selection: RuntimeSelection): Promise<InferenceResponse> {
    if (!this.handler) {
      throw new Error("Browser local inference runtime is not registered.");
    }
    return this.handler(request, selection);
  }
}
