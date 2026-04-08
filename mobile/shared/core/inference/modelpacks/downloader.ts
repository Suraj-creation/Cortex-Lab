import { parseModelpackManifest, type InferenceModelpackManifest } from "./manifest";

interface DownloadJob {
  id: string;
  packId: string;
  status: "queued" | "downloading" | "completed" | "failed";
  progress: number;
  error?: string;
}

function normalizeBaseUrl(rawBaseUrl: string): string {
  return rawBaseUrl.endsWith("/") ? rawBaseUrl.slice(0, -1) : rawBaseUrl;
}

export class ModelpackDownloader {
  private readonly baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = normalizeBaseUrl(baseUrl);
  }

  async fetchManifest(): Promise<InferenceModelpackManifest> {
    const res = await fetch(`${this.baseUrl}/modelpacks/manifest`);
    if (!res.ok) {
      throw new Error(`Failed to fetch modelpack manifest (${res.status})`);
    }
    const data = await res.json();
    return parseModelpackManifest(data);
  }

  async queueDownload(packId: string): Promise<DownloadJob> {
    // Artifact transfer is deferred until native runtime plugin wiring is complete.
    return {
      id: `dl-${Date.now()}`,
      packId,
      status: "queued",
      progress: 0,
    };
  }
}
