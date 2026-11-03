// Typed client for the lapid server (same origin). See README for the contract.

export type Capability = "chat" | "completion" | "embedding" | "transcription" | "speech";

export interface ModelInfo {
  id: string;
  name?: string;
  architecture?: string;
  task?: string;
  dtype?: string;
  license?: string;
  source?: string;
  path?: string;
  file_size?: number;
  device?: string;
  capabilities: Capability[];
  load_ms?: number;
  queued?: number;
  busy?: boolean;
  voices?: string[];
  default_voice?: string;
  sample_rate?: number;
  embedding_dim?: number;
  prompts?: string[];
  /** Activation steering files loaded with the model (`lapid serve --steer`). */
  steering?: SteeringInfo[];
}

export interface SteeringInfo {
  name: string;
  description?: string;
  /** Multiplier applied when a request does not choose (0 = loaded but off). */
  default_scale: number;
  source?: string;
  license?: string;
  entries?: { hook: string; op: string; scale: number; layers: string }[];
}

export interface ServerInfo {
  name: string;
  version: string;
  uptime_s: number;
  device?: string;
  devices?: string[];
  models: ModelInfo[];
}

export interface Timings {
  prompt_tokens?: number;
  cached_tokens?: number;
  generated_tokens?: number;
  ttft_ms?: number;
  prefill_tps?: number;
  decode_tps?: number;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens?: number;
  total_tokens: number;
}

export interface ChatChunk {
  id?: string;
  model?: string;
  choices?: {
    index: number;
    delta?: { role?: string; content?: string | null; reasoning_content?: string | null };
    finish_reason?: string | null;
  }[];
  timings?: Timings;
  usage?: Usage | null;
}

export interface ChatRequest {
  model: string;
  messages: { role: "system" | "user" | "assistant"; content: string }[];
  stream: true;
  stream_options: { include_usage: true };
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_tokens?: number;
  stop?: string[];
  seed?: number;
  repetition_penalty?: number;
  chat_template_kwargs?: { enable_thinking: boolean };
  /** Steering files by name with multipliers ({} = none); omitted = the server's default. */
  steer?: Record<string, number>;
}

export interface TranscriptionSegment {
  id: number;
  start: number;
  end: number;
  text: string;
}

export interface Transcription {
  task?: string;
  language?: string;
  duration?: number;
  text: string;
  segments?: TranscriptionSegment[];
  words?: { word: string; start: number; end: number }[];
}

export interface EmbeddingResponse {
  data: { index: number; embedding: number[] }[];
  usage?: { prompt_tokens: number; total_tokens: number };
}

export class ApiError extends Error {
  status: number;
  type?: string;
  code?: string | number | null;
  constructor(message: string, status: number, type?: string, code?: string | number | null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.type = type;
    this.code = code;
  }
}

/** Builds an ApiError from a non-2xx response using the `{error:{message,type,code}}` envelope. */
export async function toApiError(res: Response): Promise<ApiError> {
  let message = `${res.status} ${res.statusText}`.trim();
  let type: string | undefined;
  let code: string | number | null | undefined;
  try {
    const text = await res.text();
    try {
      const body = JSON.parse(text) as { error?: { message?: string; type?: string; code?: string | number | null } | string };
      if (typeof body.error === "string") message = body.error;
      else if (body.error?.message) {
        message = body.error.message;
        type = body.error.type;
        code = body.error.code;
      }
    } catch {
      if (text.trim()) message = text.trim().slice(0, 300);
    }
  } catch {
    /* body unreadable */
  }
  return new ApiError(message, res.status, type, code);
}

export function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError("Cannot reach the lapid server", 0, "network_error");
  }
  if (!res.ok) throw await toApiError(res);
  return (await res.json()) as T;
}

function postJson(body: unknown, signal?: AbortSignal): RequestInit {
  return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal };
}

export const api = {
  info: (signal?: AbortSignal) => request<ServerInfo>("/api/info", { signal }),
  health: (signal?: AbortSignal) => request<{ status: string }>("/health", { signal }),

  embeddings: (body: { model: string; input: string[]; dimensions?: number }) =>
    request<EmbeddingResponse>("/v1/embeddings", postJson(body)),

  transcribe: (args: { model: string; wav: Blob; language?: string; words?: boolean; signal?: AbortSignal }) => {
    const fd = new FormData();
    fd.append("file", args.wav, "audio.wav");
    fd.append("model", args.model);
    if (args.language?.trim()) fd.append("language", args.language.trim());
    fd.append("response_format", "verbose_json");
    fd.append("timestamp_granularities[]", "segment");
    if (args.words) fd.append("timestamp_granularities[]", "word");
    return request<Transcription>("/v1/audio/transcriptions", { method: "POST", body: fd, signal: args.signal });
  },

  speech: async (body: { model: string; input: string; voice?: string; seed?: number; temperature?: number }, signal?: AbortSignal) => {
    let res: Response;
    try {
      res = await fetch("/v1/audio/speech", postJson({ ...body, response_format: "wav" }, signal));
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") throw e;
      throw new ApiError("Cannot reach the lapid server", 0, "network_error");
    }
    if (!res.ok) throw await toApiError(res);
    return new Blob([fixStreamedWav(await res.arrayBuffer())], { type: "audio/wav" });
  },
};

/**
 * Streamed WAV bodies carry placeholder sizes (0xFFFFFFFF) in the RIFF and data chunk headers.
 * Patch both to the real lengths so decodeAudioData, wavesurfer and downloads accept the file.
 */
export function fixStreamedWav(buf: ArrayBuffer): ArrayBuffer {
  if (buf.byteLength < 44) return buf;
  const v = new DataView(buf);
  const tag = (o: number) => String.fromCharCode(v.getUint8(o), v.getUint8(o + 1), v.getUint8(o + 2), v.getUint8(o + 3));
  if (tag(0) !== "RIFF" || tag(8) !== "WAVE") return buf;
  v.setUint32(4, buf.byteLength - 8, true);
  let off = 12;
  while (off + 8 <= buf.byteLength) {
    const size = v.getUint32(off + 4, true);
    if (tag(off) === "data") {
      v.setUint32(off + 4, buf.byteLength - off - 8, true);
      break;
    }
    off += 8 + size + (size & 1);
  }
  return buf;
}
