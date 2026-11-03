import { del, get, set as idbSet } from "idb-keyval";
import { create } from "zustand";
import { createJSONStorage, persist, type StateStorage } from "zustand/middleware";
import type { Timings, Usage } from "@/lib/api";
import { uid } from "@/lib/utils";
import { useSettings } from "@/stores/settings";

export interface MessageStats extends Timings {
  completion_tokens?: number;
  finish_reason?: string | null;
  elapsed_ms?: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
  model?: string;
  stats?: MessageStats;
  usage?: Usage;
  error?: string;
  stopped?: boolean;
  createdAt: number;
}

export interface Conversation {
  id: string;
  title: string;
  systemPrompt: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
  /** Started in private mode: kept in memory only, never persisted. */
  ephemeral?: boolean;
}

/** IndexedDB storage with write coalescing: streaming updates hit the store per frame, disk at most every 800 ms. */
function idbStorage(): StateStorage {
  const pending = new Map<string, string>();
  let timer: ReturnType<typeof setTimeout> | null = null;
  const flush = () => {
    timer = null;
    for (const [k, v] of pending) void idbSet(k, v);
    pending.clear();
  };
  if (typeof window !== "undefined") {
    window.addEventListener("pagehide", flush);
    document.addEventListener("visibilitychange", () => document.visibilityState === "hidden" && flush());
  }
  return {
    getItem: async (name) => (pending.get(name) ?? (await get<string>(name))) ?? null,
    setItem: (name, value) => {
      pending.set(name, value);
      timer ??= setTimeout(flush, 800);
    },
    removeItem: async (name) => {
      pending.delete(name);
      await del(name);
    },
  };
}

interface ConversationState {
  conversations: Conversation[];
  activeId: string | null;
  hydrated: boolean;
  /** System prompt typed before the first message of a new chat. */
  draftSystem: string;
  setDraftSystem: (v: string) => void;
  create: () => string;
  select: (id: string | null) => void;
  rename: (id: string, title: string) => void;
  remove: (id: string) => void;
  setSystemPrompt: (id: string, prompt: string) => void;
  addMessage: (convId: string, msg: Omit<Message, "id" | "createdAt">) => string;
  patchMessage: (convId: string, msgId: string, patch: Partial<Message> | ((m: Message) => Partial<Message>)) => void;
  /** Drops the message with `msgId` and everything after it. */
  truncateFrom: (convId: string, msgId: string) => void;
  /** Deletes every saved (non-private) conversation, here and in browser storage. */
  removeSaved: () => void;
}

const touch = (c: Conversation, patch: Partial<Conversation>): Conversation => ({ ...c, ...patch, updatedAt: Date.now() });

export const useConversations = create<ConversationState>()(
  persist(
    (set, getState) => {
      const update = (id: string, fn: (c: Conversation) => Conversation) =>
        set((s) => ({ conversations: s.conversations.map((c) => (c.id === id ? fn(c) : c)) }));
      return {
        conversations: [],
        activeId: null,
        hydrated: false,
        draftSystem: "",
        setDraftSystem: (draftSystem) => set({ draftSystem }),
        create: () => {
          const now = Date.now();
          const ephemeral = useSettings.getState().privateMode || undefined;
          const conv: Conversation = { id: uid(), title: "New chat", systemPrompt: getState().draftSystem, messages: [], createdAt: now, updatedAt: now, ephemeral };
          set((s) => ({ conversations: [conv, ...s.conversations], activeId: conv.id, draftSystem: "" }));
          return conv.id;
        },
        select: (activeId) => set({ activeId }),
        rename: (id, title) => update(id, (c) => touch(c, { title: title.trim() || "Untitled" })),
        remove: (id) =>
          set((s) => ({
            conversations: s.conversations.filter((c) => c.id !== id),
            activeId: s.activeId === id ? null : s.activeId,
          })),
        setSystemPrompt: (id, systemPrompt) => update(id, (c) => touch(c, { systemPrompt })),
        addMessage: (convId, msg) => {
          const id = uid();
          update(convId, (c) => {
            const title =
              c.messages.length === 0 && msg.role === "user" && c.title === "New chat"
                ? msg.content.replace(/\s+/g, " ").trim().slice(0, 60) || c.title
                : c.title;
            return touch(c, { title, messages: [...c.messages, { ...msg, id, createdAt: Date.now() }] });
          });
          return id;
        },
        patchMessage: (convId, msgId, patch) =>
          update(convId, (c) =>
            touch(c, {
              messages: c.messages.map((m) => (m.id === msgId ? { ...m, ...(typeof patch === "function" ? patch(m) : patch) } : m)),
            }),
          ),
        truncateFrom: (convId, msgId) =>
          update(convId, (c) => {
            const i = c.messages.findIndex((m) => m.id === msgId);
            return i < 0 ? c : touch(c, { messages: c.messages.slice(0, i) });
          }),
        removeSaved: () =>
          set((s) => {
            const conversations = s.conversations.filter((c) => c.ephemeral);
            return { conversations, activeId: conversations.some((c) => c.id === s.activeId) ? s.activeId : null };
          }),
      };
    },
    {
      name: "lapid.conversations",
      version: 1,
      storage: createJSONStorage(idbStorage),
      // Private chats never reach storage; the rest is saved as before.
      partialize: (s) => {
        const conversations = s.conversations.filter((c) => !c.ephemeral);
        return { conversations, activeId: conversations.some((c) => c.id === s.activeId) ? s.activeId : null };
      },
      onRehydrateStorage: () => () => useConversations.setState({ hydrated: true }),
    },
  ),
);

/** Ephemeral (not persisted) streaming state. */
interface StreamState {
  convId: string | null;
  msgId: string | null;
  controller: AbortController | null;
  start: (convId: string, msgId: string, controller: AbortController) => void;
  finish: () => void;
  stop: () => void;
}

export const useStream = create<StreamState>()((set, getState) => ({
  convId: null,
  msgId: null,
  controller: null,
  start: (convId, msgId, controller) => set({ convId, msgId, controller }),
  finish: () => set({ convId: null, msgId: null, controller: null }),
  stop: () => getState().controller?.abort(),
}));
