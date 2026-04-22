import type { LocalRAGDocument } from "./store";

export interface RetrievalResult {
  doc: LocalRAGDocument;
  score: number;
}

function tokenize(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((token) => token.length > 1),
  );
}

export function rankByTokenOverlap(
  query: string,
  docs: LocalRAGDocument[],
  topK = 5,
): RetrievalResult[] {
  const queryTokens = tokenize(query);

  const scored = docs
    .map((doc) => {
      const docTokens = tokenize(doc.text);
      let overlap = 0;
      queryTokens.forEach((token) => {
        if (docTokens.has(token)) overlap += 1;
      });
      const score = queryTokens.size > 0 ? overlap / queryTokens.size : 0;
      return { doc, score };
    })
    .filter((entry) => entry.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.max(topK, 1));

  return scored;
}
