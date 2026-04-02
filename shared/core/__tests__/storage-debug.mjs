/**
 * Storage layer debug test
 * Traces through the archival logic
 */

const mockStorage = {};

const StorageBackend = {
  getItem: async (key) => mockStorage[key] || null,
  setItem: async (key, value) => {
    mockStorage[key] = value;
  },
  removeItem: async (key) => {
    delete mockStorage[key];
  },
  getAllKeys: async () => Object.keys(mockStorage),
};

const STORAGE_KEYS = {
  CONVERSATIONS: 'conversations',
};

async function saveConversation(messages, conversationId, title) {
  const id = conversationId || `conv_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
  const conversations = await getAllConversations();

  const conversation = {
    id,
    timestamp: Date.now(),
    messages,
    title: title || `Chat ${new Date(Date.now()).toLocaleDateString()}`,
  };

  const index = conversations.findIndex((c) => c.id === id);
  if (index >= 0) {
    conversations[index] = conversation;
  } else {
    conversations.push(conversation);
  }

  console.log(`  Before archival: ${conversations.length} conversations`);

  if (conversations.length > 50) {
    const toRemove = conversations.length - 50;
    conversations.splice(0, toRemove);
    console.log(`  Removed ${toRemove} old conversations to keep 50`);
  }

  console.log(`  After archival: ${conversations.length} conversations`);

  await StorageBackend.setItem(STORAGE_KEYS.CONVERSATIONS, JSON.stringify(conversations));
  return id;
}

async function getAllConversations() {
  try {
    const data = await StorageBackend.getItem(STORAGE_KEYS.CONVERSATIONS);
    return data ? JSON.parse(data) : [];
  } catch {
    return [];
  }
}

// Debug test
async function debugTest() {
  console.log('🔍 Storage Archival Debug Test\n');

  console.log('Pre-archival creation: Adding 60 conversations\n');

  for (let i = 1; i <= 60; i++) {
    const msgs = [{ id: `msg_${i}`, role: 'user', content: `Msg ${i}`, timestamp: Date.now() }];
    console.log(`  Iteration ${i}:`);
    await saveConversation(msgs, undefined, `Conv ${i}`);

    if (i % 10 === 0 || i === 60) {
      const all = await getAllConversations();
      console.log(`  → Checkpoint: ${all.length} conversations stored\n`);
    }
  }

  const final = await getAllConversations();
  console.log(`\n✅ Final state: ${final.length} conversations`);
  console.log(`Storage content size: ${JSON.stringify(mockStorage).length} bytes`);
}

debugTest().catch(console.error);
