export interface InferenceModelpackFileEntry {
  path: string;
  size_bytes: number;
  sha256: string;
}

export interface InferenceModelpackDefinition {
  id: string;
  display_name: string;
  version: string;
  target: string;
  requires: string[];
  files: InferenceModelpackFileEntry[];
}

export interface InferenceModelpackManifest {
  schema_version: string;
  generated_at: string;
  packs: InferenceModelpackDefinition[];
  signature_required: boolean;
  source?: string;
}

export function parseModelpackManifest(input: unknown): InferenceModelpackManifest {
  const data = (input || {}) as Partial<InferenceModelpackManifest>;
  return {
    schema_version: data.schema_version || "1.0",
    generated_at: data.generated_at || new Date().toISOString(),
    packs: Array.isArray(data.packs) ? data.packs : [],
    signature_required: data.signature_required !== false,
    source: data.source,
  };
}
