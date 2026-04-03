import { spawn } from "child_process";
import crypto from "crypto";
import os from "os";
import path from "path";

const DEFAULT_TIMEOUT_MS = 300_000;
const GEMINI_DEFAULT_MODEL = "gemini-2.5-flash";

type JsonRecord = Record<string, unknown>;

export const GEMINI_CLI_STATIC_MODELS = [
  { id: "gemini-2.5-pro", name: "Gemini 2.5 Pro" },
  { id: "gemini-2.5-flash", name: "Gemini 2.5 Flash" },
];

type GeminiCliRunOptions = {
  prompt: string;
  model?: string | null;
  homeDir: string;
  systemPromptPath: string;
  signal?: AbortSignal | null;
  timeoutMs?: number;
};

type GeminiCliRunResult = {
  ok: boolean;
  code: number | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
  error: string | null;
};

type GeminiCliFailure = {
  status: number;
  message: string;
  code: string;
};

// ---------------------------------------------------------------------------
// Provider-specific data normalization
// ---------------------------------------------------------------------------

function asRecord(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as JsonRecord) : {};
}

function getString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

export function normalizeGeminiCliProxyProviderData(
  providerSpecificData: JsonRecord = {}
): JsonRecord {
  return {
    ...providerSpecificData,
    homeDir: getString(providerSpecificData.homeDir),
    cliPath: getString(providerSpecificData.cliPath) || "gemini",
    systemPromptPath: getString(providerSpecificData.systemPromptPath),
    timeoutMs: Number(providerSpecificData.timeoutMs) || DEFAULT_TIMEOUT_MS,
    email: getString(providerSpecificData.email),
  };
}

// ---------------------------------------------------------------------------
// Prompt building
// ---------------------------------------------------------------------------

/**
 * Sanitize `<<role>>` delimiters in user content by inserting a zero-width
 * space between the `<` characters so the delimiter cannot appear in output.
 */
function sanitizeDelimiters(text: string): string {
  return text.replace(/<</g, "<\u200B<");
}

function flattenMessageContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";

  return content
    .map((item) => {
      if (typeof item === "string") return item;
      if (!item || typeof item !== "object") return "";

      const record = item as JsonRecord;
      const itemType = getString(record.type);
      if (itemType === "text" || itemType === "input_text") {
        return getString(record.text);
      }
      if (itemType === "image_url" || itemType === "input_image") {
        return "[Image omitted]";
      }
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

/**
 * Serialize an OpenAI messages array into flat text using `<<role>>` delimiters.
 *
 * Special case: if there is exactly one user message and no history, send it
 * raw without delimiters (matches `-p` behavior).
 */
export function buildGeminiPrompt(body: unknown): string {
  const requestBody = asRecord(body);
  const messages = Array.isArray(requestBody.messages)
    ? requestBody.messages
    : Array.isArray(requestBody.input)
      ? requestBody.input
      : [];

  if (messages.length === 0) return "";

  // Single user message with no history — send raw
  if (
    messages.length === 1 &&
    asRecord(messages[0]).role === "user" &&
    !asRecord(messages[0]).tool_calls
  ) {
    return flattenMessageContent(asRecord(messages[0]).content);
  }

  const lines: string[] = [];

  for (const message of messages) {
    const record = asRecord(message);
    const role = getString(record.role).trim().toUpperCase() || "UNKNOWN";
    const base = flattenMessageContent(record.content);

    if (role === "TOOL") {
      const toolName = getString(record.name).trim();
      lines.push(`<<tool${toolName ? `:${toolName}` : ""}>>\n${sanitizeDelimiters(base)}`);
      continue;
    }

    const toolCalls = Array.isArray(record.tool_calls) ? record.tool_calls : [];
    if (toolCalls.length > 0) {
      const toolLines = toolCalls
        .map((toolCall) => {
          const toolRecord = asRecord(toolCall);
          const functionRecord = asRecord(toolRecord.function);
          const toolName =
            getString(functionRecord.name).trim() || getString(toolRecord.name).trim() || "tool";
          const toolArgs =
            getString(functionRecord.arguments).trim() || getString(toolRecord.arguments).trim();
          return `<<tool-call:${toolName}>>\n${sanitizeDelimiters(toolArgs)}`;
        })
        .filter(Boolean)
        .join("\n\n");

      lines.push(`<<${role.toLowerCase()}>>\n${sanitizeDelimiters(base)}\n\n${toolLines}`);
      continue;
    }

    lines.push(`<<${role.toLowerCase()}>>\n${sanitizeDelimiters(base)}`);
  }

  return lines.join("\n\n");
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

/**
 * Multi-strategy JSON parser for Gemini CLI's unstable output format.
 *
 * Strategy 1: Direct JSON parse — try top-level `response` field
 * Strategy 2: Regex extraction of JSON object from mixed output
 * Strategy 3: Fallback to raw text
 */
export function extractTextFromGeminiOutput(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) return "";

  // Strategy 1: Direct JSON parse
  try {
    const parsed = JSON.parse(trimmed);
    if (typeof parsed === "string") return parsed;
    if (typeof parsed !== "object" || !parsed) return trimmed;

    // Try known field names in priority order
    const record = parsed as JsonRecord;
    for (const key of ["response", "result", "text", "content"]) {
      const val = record[key];
      if (typeof val === "string" && val.trim()) return val.trim();
    }

    // Nested content array (like OpenAI format)
    const choices = Array.isArray(record.choices) ? record.choices : null;
    if (choices && choices.length > 0) {
      const message = asRecord(asRecord(choices[0]).message);
      const content = message.content;
      if (typeof content === "string" && content.trim()) return content.trim();
    }
  } catch {
    // Not valid JSON — try strategy 2
  }

  // Strategy 2: Extract JSON from mixed output
  const jsonMatch = trimmed.match(/\{[\s\S]*"response"\s*:\s*"([^"]*?)"/);
  if (jsonMatch && jsonMatch[1]) return jsonMatch[1];

  // Strategy 3: Raw text fallback
  return trimmed;
}

/**
 * Extract token usage from Gemini CLI JSON output stats block.
 */
export function extractUsageFromGeminiOutput(raw: string): {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
} {
  try {
    const parsed = JSON.parse(raw.trim());
    if (typeof parsed !== "object" || !parsed) return { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };

    const stats = asRecord(asRecord(parsed).stats);
    const models = asRecord(stats.models);

    // Sum across all model entries (utility_router + main, etc.)
    let promptTokens = 0;
    let completionTokens = 0;

    for (const modelEntry of Object.values(models)) {
      const model = asRecord(modelEntry);
      const tokens = asRecord(model.tokens);
      promptTokens += Number(tokens.input) || 0;
      completionTokens += Number(tokens.candidates) || 0;
    }

    return {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
    };
  } catch {
    return { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
  }
}

// ---------------------------------------------------------------------------
// Completion payload construction
// ---------------------------------------------------------------------------

export function buildGeminiCompletionPayload({
  model,
  text,
  rawOutput,
}: {
  model?: string | null;
  text: string;
  rawOutput?: string;
}) {
  const usage = rawOutput ? extractUsageFromGeminiOutput(rawOutput) : { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
  const created = Math.floor(Date.now() / 1000);
  return {
    id: `chatcmpl-${crypto.randomUUID()}`,
    object: "chat.completion",
    created,
    model: model || GEMINI_DEFAULT_MODEL,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: text,
        },
        finish_reason: "stop",
      },
    ],
    usage,
  };
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

export function parseGeminiCliFailure(stderrText: string, stdoutText = ""): GeminiCliFailure {
  const stderr = String(stderrText || "").trim();
  const stdout = String(stdoutText || "").trim();
  const rawCombined = `${stderr}\n${stdout}`.trim() || "Gemini CLI request failed";
  const combined = sanitizeErrorOutput(rawCombined);
  const normalized = rawCombined.toLowerCase();

  // Auth / ToS errors
  if (
    normalized.includes("terms of service") ||
    normalized.includes("tos_violation") ||
    normalized.includes("accountverificationrequired") ||
    normalized.includes("service has been disabled")
  ) {
    return { status: 403, message: combined, code: "upstream_tos_error" };
  }

  if (
    normalized.includes("invalid api key") ||
    normalized.includes("unauthorized") ||
    normalized.includes("authentication")
  ) {
    return { status: 401, message: combined, code: "upstream_auth_error" };
  }

  // Rate limits / quota
  if (
    normalized.includes("429") ||
    normalized.includes("rate_limit") ||
    normalized.includes("resource_exhausted") ||
    normalized.includes("quota") ||
    normalized.includes("exhausted your daily quota")
  ) {
    return { status: 429, message: combined, code: "rate_limited" };
  }

  // CLI not found
  if (
    normalized.includes("command not found") ||
    normalized.includes("enoent") ||
    normalized.includes("not installed")
  ) {
    return { status: 503, message: combined, code: "runtime_error" };
  }

  // Timeout
  if (normalized.includes("timed out") || normalized.includes("timeout")) {
    return { status: 504, message: combined, code: "timeout" };
  }

  return { status: 502, message: combined, code: "upstream_error" };
}

export function createGeminiCliErrorResponse(failure: GeminiCliFailure): Response {
  return new Response(
    JSON.stringify({
      error: {
        message: failure.message,
        type: failure.status === 401 ? "authentication_error" : "provider_error",
        code: failure.code,
      },
    }),
    {
      status: failure.status,
      headers: {
        "Content-Type": "application/json",
      },
    }
  );
}

// ---------------------------------------------------------------------------
// Error sanitization
// ---------------------------------------------------------------------------

/**
 * Strip internal paths, account emails, and temp file names from CLI error
 * output before returning to API clients.
 */
function sanitizeErrorOutput(text: string): string {
  return text
    .replace(/\/home\/[^/\s]+\/\.gemini-cli[^/\s]*/gi, "~/.gemini-cli/...")
    .replace(/\/home\/[^/\s]+\/\.gemini-sessions\/[^/\s]+/gi, "~/gemini-sessions/[account]")
    .replace(/\/tmp\/omniroute-gemini-[a-f0-9-]+/gi, "/tmp/omniroute-gemini-[id]")
    .replace(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g, "[email]");
}

// ---------------------------------------------------------------------------
// Process spawner
// ---------------------------------------------------------------------------

export async function runGeminiCliCommand({
  prompt,
  model,
  homeDir,
  systemPromptPath,
  signal,
  timeoutMs = DEFAULT_TIMEOUT_MS,
  cliPath = "gemini",
}: GeminiCliRunOptions & { cliPath?: string }): Promise<GeminiCliRunResult> {
  const resolvedCliPath = cliPath;
  const args = ["--model", model || GEMINI_DEFAULT_MODEL, "-o", "json", "--yolo", "--sandbox=false"];

  const env: Record<string, string> = {
    ...process.env,
    HOME: homeDir,
    GEMINI_SYSTEM_MD: systemPromptPath,
    GEMINI_PROMPT_PREAMBLE: "false",
    GEMINI_PROMPT_COREMANDATES: "false",
    GEMINI_PROMPT_AGENTCONTEXTS: "false",
    GEMINI_PROMPT_PRIMARYWORKFLOWS: "false",
    GEMINI_PROMPT_OPERATIONALGUIDELINES: "false",
    GEMINI_PROMPT_SANDBOX: "false",
    GEMINI_PROMPT_AGENTSKILLS: "false",
    GEMINI_PROMPT_HOOKCONTEXT: "false",
    NODE_OPTIONS: "--max-old-space-size=128",
  } as Record<string, string>;

  // Remove OmniRoute-specific env vars that could confuse the CLI
  delete env.OMNIROUTE_PORT;
  delete env.PORT;
  delete env.NEXT_PUBLIC_BASE_URL;

  return new Promise((resolve) => {
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    let settled = false;

    const child = spawn(resolvedCliPath, args, {
      env,
      stdio: ["pipe", "pipe", "pipe"],
      ...(process.platform === "win32" ? { shell: true } : {}),
    });

    const cleanup = (result: GeminiCliRunResult) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener?.("abort", abortHandler);
      resolve(result);
    };

    const abortChild = () => {
      try {
        child.kill("SIGTERM");
      } catch {
        // Already dead
      }
      setTimeout(() => {
        try {
          child.kill("SIGKILL");
        } catch {
          // Already dead
        }
      }, 3000).unref?.();
    };

    const abortHandler = () => {
      abortChild();
      cleanup({
        ok: false,
        code: null,
        stdout: stdout.trim(),
        stderr: stderr.trim(),
        timedOut: false,
        error: "aborted",
      });
    };

    if (signal?.aborted) {
      abortHandler();
      return;
    }

    signal?.addEventListener?.("abort", abortHandler, { once: true });

    const timer = setTimeout(() => {
      timedOut = true;
      abortChild();
    }, timeoutMs);

    // Write prompt to stdin
    child.stdin.on("error", () => {
      // EPIPE or similar — child exited before reading stdin. Not fatal.
    });
    try {
      child.stdin.write(prompt);
      child.stdin.end();
    } catch {
      // stdin write failed — process may have already exited
    }

    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });

    child.on("error", (error: Error) => {
      cleanup({
        ok: false,
        code: null,
        stdout: stdout.trim(),
        stderr: stderr.trim(),
        timedOut,
        error: error?.message || "spawn_error",
      });
    });

    child.on("close", (code) => {
      cleanup({
        ok: !timedOut && code === 0,
        code,
        stdout: stdout.trim(),
        stderr: stderr.trim(),
        timedOut,
        error: timedOut ? "timeout" : null,
      });
    });
  });
}
