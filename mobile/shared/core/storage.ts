/**
 * Shared cross-platform storage service
 * Uses AsyncStorage for conversations and user data
 * Platform-agnostic interface for persistence
 */

import { ChatMessage, ChatSettings, DEFAULT_SETTINGS, ModelpackInstallState } from './types';

// Storage keys
const STORAGE_KEYS = {
  CONVERSATIONS: 'conversations',
  CURRENT_CONVERSATION_ID: 'current_conversation_id',
  CHAT_SETTINGS: 'chat_settings',
  LAST_API_URL: 'last_api_url',
  ONBOARDING_COMPLETED: 'onboarding_completed',
  SYNC_METADATA: 'sync_metadata',
  MODELPACK_INSTALLS: 'modelpack_installs',
};

interface StoredConversation {
  id: string;
  timestamp: number;
  messages: ChatMessage[];
  title?: string;
}

interface SyncMetadata {
  lastSyncTime: number;
  conversationCount: number;
}

/**
 * AsyncStorage-based implementation for web/mobile
 * For secure data (API keys), use SecureStore separately
 */

// Lazy-loaded storage backend to prevent initialization errors
let _storageLazyBackend: any = null;

function getStorageBackend() {
  if (_storageLazyBackend !== null) {
    return _storageLazyBackend;
  }

  try {
    // Native React Native environment
    const AsyncStorage = require('@react-native-async-storage/async-storage').default;
    _storageLazyBackend = AsyncStorage;
  } catch (e) {
    console.warn('AsyncStorage not available, using memory fallback:', e);
    // Fallback for web or error case: use memory store
    const memoryStore: Record<string, string> = {};
    _storageLazyBackend = {
      getItem: async (key: string) => memoryStore[key] || null,
      setItem: async (key: string, value: string) => {
        memoryStore[key] = value;
      },
      removeItem: async (key: string) => {
        delete memoryStore[key];
      },
      getAllKeys: async () => Object.keys(memoryStore),
    };
  }

  return _storageLazyBackend;
}

export const StorageBackend = {
  getItem: (key: string) => getStorageBackend().getItem(key),
  setItem: (key: string, value: string) => getStorageBackend().setItem(key, value),
  removeItem: (key: string) => getStorageBackend().removeItem(key),
  getAllKeys: () => getStorageBackend().getAllKeys(),
};

/**
 * Save a conversation (new or update existing)
 */
export async function saveConversation(
  messages: ChatMessage[],
  conversationId?: string,
  title?: string
): Promise<string> {
  const id = conversationId || `conv_${Date.now()}`;
  const conversations = await getAllConversations();

  const conversation: StoredConversation = {
    id,
    timestamp: Date.now(),
    messages,
    title: title || `Chat ${new Date(Date.now()).toLocaleDateString()}`,
  };

  // Update or add conversation
  const index = conversations.findIndex((c) => c.id === id);
  if (index >= 0) {
    conversations[index] = conversation;
  } else {
    conversations.push(conversation);
  }

  // Keep only last 50 conversations to limit storage
  if (conversations.length > 50) {
    conversations.splice(0, conversations.length - 50);
  }

  await StorageBackend.setItem(STORAGE_KEYS.CONVERSATIONS, JSON.stringify(conversations));
  await updateSyncMetadata();

  return id;
}

/**
 * Load all conversations
 */
export async function getAllConversations(): Promise<StoredConversation[]> {
  try {
    const data = await StorageBackend.getItem(STORAGE_KEYS.CONVERSATIONS);
    return data ? JSON.parse(data) : [];
  } catch {
    console.error('Failed to load conversations');
    return [];
  }
}

/**
 * Get specific conversation by ID
 */
export async function getConversation(conversationId: string): Promise<StoredConversation | null> {
  const conversations = await getAllConversations();
  return conversations.find((c) => c.id === conversationId) || null;
}

/**
 * Delete conversation
 */
export async function deleteConversation(conversationId: string): Promise<void> {
  const conversations = await getAllConversations();
  const filtered = conversations.filter((c) => c.id !== conversationId);
  await StorageBackend.setItem(STORAGE_KEYS.CONVERSATIONS, JSON.stringify(filtered));
  await updateSyncMetadata();
}

/**
 * Clear all conversations
 */
export async function clearAllConversations(): Promise<void> {
  await StorageBackend.removeItem(STORAGE_KEYS.CONVERSATIONS);
  await StorageBackend.removeItem(STORAGE_KEYS.CURRENT_CONVERSATION_ID);
  await updateSyncMetadata();
}

/**
 * Save chat settings (temperature, provider, etc.)
 */
export async function saveChatSettings(settings: ChatSettings): Promise<void> {
  await StorageBackend.setItem(STORAGE_KEYS.CHAT_SETTINGS, JSON.stringify(settings));
}

/**
 * Load chat settings
 */
export async function loadChatSettings(): Promise<ChatSettings> {
  try {
    const data = await StorageBackend.getItem(STORAGE_KEYS.CHAT_SETTINGS);
    if (data) {
      return { ...DEFAULT_SETTINGS, ...JSON.parse(data) };
    }
  } catch (e) {
    console.error('Failed to load chat settings:', e);
  }
  return DEFAULT_SETTINGS;
}

/**
 * Save current conversation ID
 */
export async function saveCurrentConversationId(conversationId: string): Promise<void> {
  await StorageBackend.setItem(STORAGE_KEYS.CURRENT_CONVERSATION_ID, conversationId);
}

/**
 * Get current conversation ID
 */
export async function getCurrentConversationId(): Promise<string | null> {
  return await StorageBackend.getItem(STORAGE_KEYS.CURRENT_CONVERSATION_ID);
}

/**
 * Save last API URL (for multi-deployment scenario)
 */
export async function saveLastApiUrl(url: string): Promise<void> {
  await StorageBackend.setItem(STORAGE_KEYS.LAST_API_URL, url);
}

/**
 * Get last API URL
 */
export async function getLastApiUrl(): Promise<string | null> {
  return await StorageBackend.getItem(STORAGE_KEYS.LAST_API_URL);
}

/**
 * Persist onboarding completion flag
 */
export async function saveOnboardingCompleted(completed: boolean): Promise<void> {
  await StorageBackend.setItem(STORAGE_KEYS.ONBOARDING_COMPLETED, completed ? 'true' : 'false');
}

/**
 * Check whether onboarding has been completed
 */
export async function getOnboardingCompleted(): Promise<boolean> {
  try {
    return (await StorageBackend.getItem(STORAGE_KEYS.ONBOARDING_COMPLETED)) === 'true';
  } catch {
    return false;
  }
}

export async function saveModelpackInstalls(
  installs: Record<string, ModelpackInstallState>,
): Promise<void> {
  await StorageBackend.setItem(STORAGE_KEYS.MODELPACK_INSTALLS, JSON.stringify(installs));
}

export async function loadModelpackInstalls(): Promise<Record<string, ModelpackInstallState>> {
  try {
    const data = await StorageBackend.getItem(STORAGE_KEYS.MODELPACK_INSTALLS);
    const parsed = data ? JSON.parse(data) : {};
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch (error) {
    console.error('Failed to load modelpack installs:', error);
    return {};
  }
}

/**
 * Update sync metadata (for observability)
 */
async function updateSyncMetadata(): Promise<void> {
  const conversations = await getAllConversations();
  const metadata: SyncMetadata = {
    lastSyncTime: Date.now(),
    conversationCount: conversations.length,
  };
  await StorageBackend.setItem(STORAGE_KEYS.SYNC_METADATA, JSON.stringify(metadata));
}

/**
 * Get sync metadata
 */
export async function getSyncMetadata(): Promise<SyncMetadata | null> {
  try {
    const data = await StorageBackend.getItem(STORAGE_KEYS.SYNC_METADATA);
    return data ? JSON.parse(data) : null;
  } catch {
    return null;
  }
}

/**
 * Initialize storage on app launch
 * Cleans up old/corrupt data
 */
export async function initializeStorage(): Promise<void> {
  try {
    const conversations = await getAllConversations();
    // Remove conversations older than 90 days
    const cutoff = Date.now() - 90 * 24 * 60 * 60 * 1000;
    const filtered = conversations.filter((c) => c.timestamp > cutoff);

    if (filtered.length !== conversations.length) {
      await StorageBackend.setItem(STORAGE_KEYS.CONVERSATIONS, JSON.stringify(filtered));
    }

    await updateSyncMetadata();
  } catch (e) {
    console.error('Failed to initialize storage:', e);
  }
}
