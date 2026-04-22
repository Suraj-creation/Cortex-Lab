import { ragChat, sendMessage } from "../../api";
import type {
  InferenceAdapter,
  InferenceRequest,
  InferenceResponse,
  RuntimeCapabilities,
  RuntimeSelection,
} from "../types";

export class CloudAdapter implements InferenceAdapter {
  readonly id = "cloud";

  canHandle(_selection: RuntimeSelection, caps: RuntimeCapabilities): boolean {
    return caps.cloudAvailable;
  }

  async run(request: InferenceRequest, selection: RuntimeSelection): Promise<InferenceResponse> {
    const result = request.settings.useRAG
      ? await ragChat(request.messages, request.settings, request.sessionId || "")
      : await sendMessage(request.messages, request.settings);

    return {
      content: result.content,
      thinking: result.thinking,
      provider: selection.llmProvider,
      mode: selection.mode,
      traceId:
        request.settings.useRAG && "pipeline_trace" in result
          ? result.pipeline_trace?.trace_id
          : undefined,
    };
  }
}
