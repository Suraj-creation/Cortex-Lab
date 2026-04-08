export interface ModelpackVerifyResult {
  verified: boolean;
  algorithm: string;
  file_path: string;
  file_size_bytes: number;
  expected_sha256: string;
  actual_sha256: string;
}

function normalizeBaseUrl(rawBaseUrl: string): string {
  return rawBaseUrl.endsWith("/") ? rawBaseUrl.slice(0, -1) : rawBaseUrl;
}

export async function verifyModelpackWithBackend(
  baseUrl: string,
  filePath: string,
  expectedSha256: string,
): Promise<ModelpackVerifyResult> {
  const normalized = normalizeBaseUrl(baseUrl);
  const res = await fetch(`${normalized}/modelpacks/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      file_path: filePath,
      expected_sha256: expectedSha256,
    }),
  });

  if (!res.ok) {
    const detail = await res
      .json()
      .then((data) => data?.detail as string | undefined)
      .catch(() => undefined);
    throw new Error(detail || `Modelpack verification failed (${res.status})`);
  }

  return res.json();
}
