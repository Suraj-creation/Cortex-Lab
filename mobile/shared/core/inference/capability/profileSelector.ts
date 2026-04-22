import type { RuntimeSelection } from "../types";
import type { DeviceProfile } from "./deviceProfiler";

interface RuntimeAvailability {
  localLlmReady: boolean;
  localVoiceReady: boolean;
  cloudReachable: boolean;
}

export function selectRuntimeForDevice(
  profile: DeviceProfile,
  availability: RuntimeAvailability,
): RuntimeSelection {
  if (availability.localLlmReady && availability.localVoiceReady) {
    return {
      mode: "local_offline",
      llmProvider: "gemma_local",
      sttProvider: "local",
      ttsProvider: "local",
      allowCloudFallback: false,
    };
  }

  if (availability.localLlmReady && availability.cloudReachable) {
    return {
      mode: "hybrid",
      llmProvider: "gemma_local",
      sttProvider: "local",
      ttsProvider: "local",
      allowCloudFallback: true,
    };
  }

  return {
    mode: "cloud",
    llmProvider: "gemini",
    sttProvider: profile.platform === "web" ? "gemini" : "traditional",
    ttsProvider: profile.platform === "web" ? "gemini" : "traditional",
    allowCloudFallback: true,
  };
}
