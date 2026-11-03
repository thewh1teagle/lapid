import { toast } from "sonner";
import { type ChatRequest, errorMessage } from "@/lib/api";
import { streamChat } from "@/lib/sse";
import { type Message, type MessageStats, useConversations, useStream } from "@/stores/conversations";
import { type Sampling, type SteerChoice, useSettings } from "@/stores/settings";

function buildRequest(model: string, system: string, history: Message[], s: Sampling, steer?: Record<string, SteerChoice>): ChatRequest {
  const messages: ChatRequest["messages"] = [];
  if (system.trim()) messages.push({ role: "system", content: system.trim() });
  for (const m of history) {
    if (m.role === "assistant" && !m.content && m.error) continue;
    messages.push({ role: m.role, content: m.content });
  }
  const stop = s.stop
    .split("\n")
    .map((x) => x.replace(/\\n/g, "\n"))
    .filter((x) => x.length > 0);
  const seed = s.seed.trim() === "" ? undefined : Number(s.seed);
  return {
    model,
    messages,
    stream: true,
    stream_options: { include_usage: true },
    temperature: s.temperature,
    top_p: s.top_p,
    top_k: s.top_k,
    max_tokens: s.max_tokens,
    repetition_penalty: s.repetition_penalty,
    ...(stop.length ? { stop } : {}),
    ...(seed !== undefined && Number.isFinite(seed) ? { seed } : {}),
    chat_template_kwargs: { enable_thinking: s.enable_thinking },
    ...(steer ? { steer: Object.fromEntries(Object.entries(steer).filter(([, c]) => c.enabled).map(([n, c]) => [n, c.scale])) } : {}),
  };
}

/**
 * Streams an assistant reply into a fresh placeholder message appended to `convId`.
 * Deltas are batched per animation frame so long answers do not re-render per token.
 */
export async function generateReply(convId: string, model: string): Promise<void> {
  const store = useConversations.getState();
  const stream = useStream.getState();
  if (stream.controller) return;
  const conv = store.conversations.find((c) => c.id === convId);
  if (!conv) return;
  const history = conv.messages;
  const msgId = store.addMessage(convId, { role: "assistant", content: "", reasoning: "", model });
  const controller = new AbortController();
  stream.start(convId, msgId, controller);

  const settings = useSettings.getState();
  const req = buildRequest(model, conv.systemPrompt, history, settings.sampling, settings.steer[model]);
  const t0 = performance.now();
  let firstToken = 0;
  let content = "";
  let reasoning = "";
  let stats: MessageStats = {};
  let frame = 0;
  const flush = () => {
    frame = 0;
    useConversations.getState().patchMessage(convId, msgId, { content, reasoning, stats });
  };
  const schedule = () => {
    frame ||= requestAnimationFrame(flush);
  };

  try {
    await streamChat(
      req,
      (chunk) => {
        const choice = chunk.choices?.[0];
        const d = choice?.delta;
        if (d?.content || d?.reasoning_content) {
          if (!firstToken) firstToken = performance.now();
          if (d.content) content += d.content;
          if (d.reasoning_content) reasoning += d.reasoning_content;
        }
        if (choice?.finish_reason) stats = { ...stats, finish_reason: choice.finish_reason };
        if (chunk.timings) stats = { ...stats, ...chunk.timings };
        if (chunk.usage) stats = { ...stats, prompt_tokens: stats.prompt_tokens ?? chunk.usage.prompt_tokens, completion_tokens: chunk.usage.completion_tokens };
        schedule();
      },
      controller.signal,
    );
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      useConversations.getState().patchMessage(convId, msgId, { stopped: true });
    } else {
      const msg = errorMessage(err);
      toast.error(msg);
      useConversations.getState().patchMessage(convId, msgId, { error: msg });
    }
  } finally {
    if (frame) cancelAnimationFrame(frame);
    const elapsed = performance.now() - t0;
    stats = {
      ...stats,
      elapsed_ms: Math.round(elapsed),
      ttft_ms: stats.ttft_ms ?? (firstToken ? Math.round(firstToken - t0) : undefined),
    };
    flush();
    useStream.getState().finish();
  }
}

export async function sendMessage(text: string, model: string): Promise<void> {
  if (useStream.getState().controller) return;
  const store = useConversations.getState();
  let convId = store.activeId;
  if (!convId || !store.conversations.some((c) => c.id === convId)) convId = store.create();
  store.addMessage(convId, { role: "user", content: text });
  await generateReply(convId, model);
}

export async function regenerate(convId: string, model: string): Promise<void> {
  const conv = useConversations.getState().conversations.find((c) => c.id === convId);
  const last = conv?.messages.at(-1);
  if (!conv || !last) return;
  if (last.role === "assistant") useConversations.getState().truncateFrom(convId, last.id);
  await generateReply(convId, model);
}

/** Replaces a user message's text, drops everything after it and generates a new reply. */
export async function editAndResend(convId: string, msgId: string, text: string, model: string): Promise<void> {
  const s = useConversations.getState();
  const conv = s.conversations.find((c) => c.id === convId);
  const idx = conv?.messages.findIndex((m) => m.id === msgId) ?? -1;
  const next = conv?.messages[idx + 1];
  if (next) s.truncateFrom(convId, next.id);
  s.patchMessage(convId, msgId, { content: text });
  await generateReply(convId, model);
}
