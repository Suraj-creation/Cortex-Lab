/**
 * Storage layer validation test
 * Verifies persistence logic works correctly before mobile deployment
 * Run with: node shared/core/__tests__/storage.test.mjs
 */

// Mock AsyncStorage for Node environment
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

// Simulated storage functions (copy-pasted from storage.ts for testing)
const STORAGE_KEYS = {
  CONVERSATIONS: 'conversations',
  CURRENT_CONVERSATION_ID: 'current_conversation_id',
  CHAT_SETTINGS: 'chat_settings',
};

async function saveConversation(messages, conversationId, title) {
  const id = conversationId || `conv_${Date.now()}`;
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

  if (conversations.length > 50) {
    conversations.splice(0, conversations.length - 50);
  }

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

async function getConversation(conversationId) {
  const conversations = await getAllConversations();
  return conversations.find((c) => c.id === conversationId) || null;
}

async function loadChatSettings() {
  try {
    const data = await StorageBackend.getItem(STORAGE_KEYS.CHAT_SETTINGS);
    if (data) {
      return { ...{ temperature: 0.6, topP: 0.95 }, ...JSON.parse(data) };
    }
  } catch (e) {
    console.error('Failed to load chat settings:', e);
  }
  return { temperature: 0.6, topP: 0.95 };
}

async function saveChatSettings(settings) {
  await StorageBackend.setItem(STORAGE_KEYS.CHAT_SETTINGS, JSON.stringify(settings));
}

// Test suite
async function runTests() {
  console.log('🧪 Starting Storage Layer Tests\n');

  // Test 1: Save conversation
  console.log('Test 1: Save conversation');
  const messages1 = [
    { id: 'msg1', role: 'user', content: 'Hello', timestamp: Date.now() },
    { id: 'msg2', role: 'assistant', content: 'Hi there!', timestamp: Date.now() },
  ];
  const convId1 = await saveConversation(messages1, undefined, 'Test Chat 1');
  console.log(`✓ Saved conversation ${convId1}`);

  // Test 2: Retrieve conversation
  console.log('\nTest 2: Retrieve conversation');
  const retrieved = await getConversation(convId1);
  if (retrieved && retrieved.messages.length === 2) {
    console.log(`✓ Retrieved conversation with ${retrieved.messages.length} messages`);
  } else {
    console.log(`✗ Failed to retrieve conversation`);
  }

  // Test 3: Save settings
  console.log('\nTest 3: Save settings');
  const settings = { temperature: 0.8, topP: 0.9, stream: true, useRAG: true };
  await saveChatSettings(settings);
  console.log('✓ Saved chat settings');

  // Test 4: Load settings
  console.log('\nTest 4: Load settings');
  const loaded = await loadChatSettings();
  if (loaded.temperature === 0.8 && loaded.stream === true) {
    console.log(`✓ Loaded settings correctly: temp=${loaded.temperature}, stream=${loaded.stream}`);
  } else {
    console.log(`✗ Failed to load settings`);
  }

  // Test 5: Multiple conversations
  console.log('\nTest 5: Multiple conversations');
  const messages2 = [
    { id: 'msg3', role: 'user', content: 'Another question', timestamp: Date.now() },
  ];
  const convId2 = await saveConversation(messages2, undefined, 'Test Chat 2');
  const all = await getAllConversations();
  if (all.length === 2) {
    console.log(`✓ Stored ${all.length} conversations`);
  } else {
    console.log(`✗ Expected 2 conversations, got ${all.length}`);
  }

  // Test 6: Update conversation
  console.log('\nTest 6: Update conversation');
  const updatedMessages = [
    ...messages1,
    { id: 'msg4', role: 'user', content: 'Follow-up', timestamp: Date.now() },
  ];
  await saveConversation(updatedMessages, convId1, 'Updated Chat 1');
  const updated = await getConversation(convId1);
  if (updated && updated.messages.length === 3) {
    console.log(`✓ Updated conversation now has ${updated.messages.length} messages`);
  } else {
    console.log(`✗ Failed to update conversation`);
  }

  // Test 7: Conversation archival (> 50)
  console.log('\nTest 7: Conversation archival');
  for (let i = 0; i < 60; i++) {
    const msgs = [{ id: `msg_${i}`, role: 'user', content: `Msg ${i}`, timestamp: Date.now() }];
    await saveConversation(msgs, undefined, `Conv ${i}`);
  }
  const archived = await getAllConversations();
  if (archived.length === 50) {
    console.log(`✓ Storage capped at ${archived.length} conversations (archival working)`);
  } else {
    console.log(`✗ Expected 50 conversations, got ${archived.length}`);
  }

  console.log('\n✅ Storage layer validation complete!');
}

runTests().catch(console.error);
