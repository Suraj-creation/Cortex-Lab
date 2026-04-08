import type { InferenceModelpackDefinition, InferenceModelpackManifest } from "./manifest";

export type InstallState = "not_installed" | "downloading" | "installed" | "error";

export interface InstalledModelpack {
  id: string;
  version: string;
  state: InstallState;
  updatedAt: string;
  error?: string;
}

export class ModelpackRegistry {
  private manifest: InferenceModelpackManifest | null = null;
  private installed = new Map<string, InstalledModelpack>();

  setManifest(manifest: InferenceModelpackManifest): void {
    this.manifest = manifest;
  }

  getManifest(): InferenceModelpackManifest | null {
    return this.manifest;
  }

  listPacks(): InferenceModelpackDefinition[] {
    return this.manifest?.packs || [];
  }

  setInstallState(id: string, state: InstallState, error?: string): InstalledModelpack {
    const current = this.installed.get(id);
    const next: InstalledModelpack = {
      id,
      version: current?.version || this.findPackVersion(id),
      state,
      updatedAt: new Date().toISOString(),
      error,
    };
    this.installed.set(id, next);
    return next;
  }

  getInstallState(id: string): InstalledModelpack | null {
    return this.installed.get(id) || null;
  }

  listInstallState(): InstalledModelpack[] {
    return Array.from(this.installed.values()).sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  }

  private findPackVersion(id: string): string {
    const pack = this.manifest?.packs.find((item) => item.id === id);
    return pack?.version || "unknown";
  }
}
