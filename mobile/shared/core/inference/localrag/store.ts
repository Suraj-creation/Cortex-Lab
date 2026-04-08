export interface LocalRAGDocument {
  id: string;
  text: string;
  source: string;
  metadata?: Record<string, unknown>;
  createdAt: string;
}

export class InMemoryLocalRAGStore {
  private docs = new Map<string, LocalRAGDocument>();

  upsert(doc: LocalRAGDocument): LocalRAGDocument {
    this.docs.set(doc.id, doc);
    return doc;
  }

  get(id: string): LocalRAGDocument | null {
    return this.docs.get(id) || null;
  }

  list(limit = 100): LocalRAGDocument[] {
    return Array.from(this.docs.values())
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
      .slice(0, Math.max(limit, 1));
  }

  remove(id: string): boolean {
    return this.docs.delete(id);
  }

  size(): number {
    return this.docs.size;
  }
}
