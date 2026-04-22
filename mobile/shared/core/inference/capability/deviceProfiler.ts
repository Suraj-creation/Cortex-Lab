import { Platform } from "react-native";

export interface DeviceProfile {
  platform: "ios" | "android" | "web";
  cpuTier: "low" | "mid" | "high";
  memoryTier: "low" | "mid" | "high";
  batteryAware: boolean;
}

export function getDeviceProfile(): DeviceProfile {
  const platform = Platform.OS === "ios" || Platform.OS === "android" ? Platform.OS : "web";

  // Conservative defaults that can be overridden once native runtime telemetry lands.
  const profile: DeviceProfile = {
    platform,
    cpuTier: platform === "android" ? "mid" : "high",
    memoryTier: platform === "android" ? "mid" : "high",
    batteryAware: platform !== "web",
  };

  return profile;
}
