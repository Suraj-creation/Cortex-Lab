import type { ChatMessage } from "../../types";
import type { InMemoryLocalRAGStore, LocalRAGDocument } from "./store";

export function ingestConversationTurn(
  store: InMemoryLocalRAGStore,
  sessionId: string,
  turn: ChatMessage,
): LocalRAGDocument {
  const doc: LocalRAGDocument = {
    id: `${sessionId}-${turn.id}`,
    text: turn.content,
    source: turn.role,
    metadata: {
      role: turn.role,
      timestamp: turn.timestamp,
      session_id: sessionId,
    },
    createdAt: new Date(turn.timestamp).toISOString(),
  };

  return store.upsert(doc);
}
