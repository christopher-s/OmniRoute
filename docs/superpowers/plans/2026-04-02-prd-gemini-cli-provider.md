# PRD: Gemini CLI Subprocess Provider for OmniRoute

**Version:** 1.1
**Date:** 2026-04-02
**Author:** Winston (with Chris Staley)
**Status:** Draft

> **Validated against:** Gemini CLI v0.36.0 source code, OmniRoute `@omniroute/open-sse` executor architecture, star-cliproxy (TypeScript), KiCk (Dart), code-proxy (Go). Uncertain items marked `[UNVERIFIED]`.
>
> **v1.1 changes:** Corrected reference implementation status (Qoder is merged but not production-validated). Added exact registration steps. Fixed configuration model to match OmniRoute's database-backed `providerSpecificData`. Clarified that no translator is needed. Added interface requirements from `BaseExecutor`. Resolved open questions where possible.

---

## 1. Problem Statement

OmniRoute's existing `gemini-cli` executor makes direct HTTP calls to `cloudcode-pa.googleapis.com` using extracted OAuth tokens. Google flagged this as "third-party software using CLI OAuth" and blocked the account.

We need a new executor (`gemini-cli-proxy`) that routes requests through the actual Gemini CLI binary, so Google sees legitimate CLI traffic.

---

## 2. Design Decision: Subprocess Over Alternatives

| Approach | Verdict | Reason |
|----------|---------|--------|
| **Subprocess spawning** | ✅ Chosen | Google sees real Gemini CLI traffic. Handles OAuth, telemetry, headers natively. |
| **ACP protocol** | ❌ Rejected | Requires one persistent Node.js process per account (~100MB each). Over-engineered for stateless proxying. |
| **A2A protocol** | ❌ Rejected | Inter-agent communication framework, not a model proxy. Adds unnecessary complexity. |
| **Direct API with spoofing** | ❌ Rejected | Same thing that got the account blocked, just with better camouflage. |
| **Library import (`@google/gemini-cli-core`)** | ❌ Rejected | Wraps direct HTTP calls to `cloudcode-pa.googleapis.com`. Still missing telemetry, not real CLI traffic. |

The subprocess approach is the only one where Google sees traffic from an actual Gemini CLI binary. The CLI handles OAuth refresh, Clearcut telemetry, User-Agent fingerprinting, and surface detection exactly as Google expects.

---

## 3. Phase 1: MVP

### 3.0 Reference Implementation

OmniRoute's Qoder executor (`open-sse/executors/qoder.ts` + `open-sse/services/qoderCli.ts`) implements the subprocess executor pattern for Qoder AI's `qodercli` binary. It is merged into the codebase but **not production-validated** — it has never been tested under real load with the actual `qodercli` binary. Treat it as a structural template, not a proven pattern.

**Key files to use as templates:**
- `open-sse/executors/qoder.ts` → Template for `gemini-cli-proxy.ts`
- `open-sse/services/qoderCli.ts` → Template for `geminiCli.ts`

**Structural patterns from Qoder (merged but unvalidated in production):**
- `BaseExecutor` subclass with overridden `execute()` that returns synthetic `Response` objects
- `new Response(body, {status, headers})` for non-streaming responses
- `ReadableStream` wrapping `child_process.spawn` stdout for SSE streaming
- `AbortSignal` propagation to `child.kill('SIGTERM')`
- `setTimeout` → SIGTERM → SIGKILL timeout chain
- Error classification from CLI stderr
- OpenAI chunk construction (`buildQoderChunk`)

**Important caveat:** These patterns are architecturally sound (they match how `BaseExecutor` works — the base `execute()` does `fetch()` and returns `{ response, url, headers, transformedBody }`, and the Qoder executor overrides `execute()` entirely to return the same shape with synthetic `Response` objects). But edge cases (memory leaks under load, zombie processes, signal handling) have not been stress-tested. The Gemini CLI implementation must include its own robust testing.

**Differences from Qoder implementation:**

| Aspect | Qoder PR #934 | Gemini CLI (ours) | Why different |
|--------|---------------|-------------------|-------------|
| Prompt delivery | `-p` flag | stdin pipe | ARG_MAX limits on large conversation histories |
| Non-streaming capture | Pipe | File redirect | Gemini CLI's 8KB stdout buffer truncation bug |
| Auth | `QODER_PAT` env var | `HOME` override | Gemini uses OAuth tokens stored in `~/.gemini-cli/` per HOME dir |
| System prompt | Prepended text in prompt | Empty file via `GEMINI_SYSTEM_MD` | We want zero system prompt, not injected instructions |
| Tool exclusion | Prompt says "don't use tools" | `excludeTools` in settings.json | We want tools fully removed from API declarations |
| Multi-account | Single account | Phase 2 | Qoder uses PAT auth, no account pooling needed |
| Message serialization | `buildQoderPrompt()` with role headers | `<<role>>` delimiter format | Same concept, different delimiter style |

### 3.1 What MVP Does

Spawn the `gemini` binary as a subprocess. Pipe prompt via stdin. Capture stdout to a temp file. Parse the JSON response. Return as OpenAI-compatible format.

**Scope:**
- Single account
- Non-streaming only
- Text only (no images)
- Zero system prompt (no coding agent behavior)
- All tools excluded
- Serialized execution (one request at a time)

### 3.2 OmniRoute Integration

OmniRoute uses an executor pattern in `@omniroute/open-sse`. The new executor spawns a subprocess instead of making HTTP calls. It creates a synthetic `Response` from CLI stdout, since `handleChatCore` expects `{ response, url, headers, transformedBody }` from `executor.execute()`. No translator is needed — the executor constructs OpenAI-compatible responses directly (same as Qoder).

**Files to create:**
- `open-sse/executors/gemini-cli-proxy.ts` — New executor class
- `open-sse/services/geminiCli.ts` — CLI service module (spawn, parse, serialize)

**Files to modify (registration):**
- `open-sse/executors/index.ts` — Add `"gemini-cli-proxy": new GeminiCliProxyExecutor()` to the `executors` object (line 16-36)
- `open-sse/config/providerRegistry.ts` — Add `gemini-cli-proxy` registry entry (following `gemini-cli` pattern at line 213)
- `src/shared/constants/providers.ts` — Add to `FREE_PROVIDERS` (alongside existing `gemini-cli` entry at line 7)

**Registration details:**

1. **Executor mapping** (`open-sse/executors/index.ts`):
   ```typescript
   import { GeminiCliProxyExecutor } from "./gemini-cli-proxy.ts";
   // Add to executors object:
   "gemini-cli-proxy": new GeminiCliProxyExecutor(),
   ```

2. **Provider registry** (`open-sse/config/providerRegistry.ts`):
   ```typescript
   "gemini-cli-proxy": {
     id: "gemini-cli-proxy",
     alias: "gcp",   // short alias for CLI usage
     format: "openai",  // Returns OpenAI-compatible responses directly (no translator)
     executor: "gemini-cli-proxy",
     baseUrl: "gemini-cli-proxy://local",  // Placeholder — no HTTP calls
     authType: "oauth",  // Uses OAuth tokens from HOME dir via Gemini CLI
     authHeader: "bearer",
     defaultContextLength: 1000000,
     models: [
       { id: "gemini-2.5-pro", name: "Gemini 2.5 Pro" },
       { id: "gemini-2.5-flash", name: "Gemini 2.5 Flash" },
     ],
   }
   ```

3. **Provider constants** (`src/shared/constants/providers.ts`):
   ```typescript
   "gemini-cli-proxy": {
     id: "gemini-cli-proxy",
     alias: "gcp",
     name: "Gemini CLI Proxy",
     icon: "terminal",
     color: "#4285F4",
   }
   ```

**Configuration model:** OmniRoute stores provider config in the database via `providerSpecificData` on the credentials object (`ProviderCredentials.providerSpecificData`). The YAML in section 3.9 is illustrative only. Actual config fields go into `providerSpecificData`:
```typescript
providerSpecificData: {
  homeDir: "/home/overmind/.gemini-sessions/acct-001",
  cliPath: "gemini",
  systemPromptPath: "/opt/gemini-sessions/config/empty-prompt.md",
  timeoutMs: 300000,
}
```

A `normalizeGeminiCliProxyProviderData()` function in `geminiCli.ts` (following `normalizeQoderPatProviderData` pattern) should set defaults and validate these fields.

**BaseExecutor interface requirements:**

The executor must extend `BaseExecutor` and override `execute()` completely (like Qoder). The base `execute()` does HTTP fetch with retry logic — not applicable for subprocess executors.

Required overrides:
- `execute(input: ExecuteInput)` — Main entry point. Must return `{ response: Response, url: string, headers: Record<string, string>, transformedBody: unknown }`
- `buildHeaders(credentials, stream)` — Return basic headers (Content-Type, etc.)
- `refreshCredentials()` — Return `null` (Gemini CLI handles OAuth refresh internally via stored tokens)

Can inherit without override:
- `buildUrl()` — Not used since we override `execute()`, but keep for interface compatibility
- `transformRequest()` — Not used since we override `execute()`
- `needsRefresh()` — Default behavior is fine (checks `expiresAt`)

### 3.3 Process Spawner

**Spawn command:**
```bash
NODE_OPTIONS="--max-old-space-size=128" \
HOME={home_dir} \
GEMINI_SYSTEM_MD={empty_prompt_path} \
GEMINI_PROMPT_PREAMBLE=false \
GEMINI_PROMPT_COREMANDATES=false \
GEMINI_PROMPT_AGENTCONTEXTS=false \
GEMINI_PROMPT_PRIMARYWORKFLOWS=false \
GEMINI_PROMPT_OPERATIONALGUIDELINES=false \
GEMINI_PROMPT_SANDBOX=false \
GEMINI_PROMPT_AGENTSKILLS=false \
GEMINI_PROMPT_HOOKCONTEXT=false \
gemini --model {model} -o json --yolo --sandbox=false \
  > {temp_file}
```

Prompt delivered via stdin pipe. Output captured via file redirection (avoids 8KB stdout buffer truncation bug in Node.js pipes).

**Process lifecycle:**
1. Spawn process with account's HOME dir, env vars above
2. Write serialized prompt to stdin, close stdin
3. Wait for process exit (max 300s)
4. On timeout: SIGTERM → 3s → SIGKILL
5. Read temp file, parse JSON response
6. Clean up temp file (always, even on error)
7. Return response or error

**Startup cleanup:** Kill any orphaned `gemini` processes from previous OmniRoute instances. Clean orphaned temp files.

### 3.4 Zero System Prompt

Gemini CLI's `GEMINI_SYSTEM_MD` env var controls the system prompt:

- `GEMINI_SYSTEM_MD=false` or `"0"` → loads the FULL default coding agent prompt (do NOT use this)
- `GEMINI_SYSTEM_MD=/path/to/empty-file.md` → uses empty file as base prompt → effectively zero system prompt

**`[UNVERIFIED]`** This is inferred from source code (`promptProvider.ts`). Validation needed. **Fallback:** If empty file is rejected, use a minimal file containing a single space character or a no-op instruction like "You are a helpful assistant."

The client's `messages[0].role == "system"` content is serialized into the prompt text. The Gemini CLI itself sends no system prompt.

### 3.5 Tool Exclusion

All 25 built-in tools excluded via `~/.gemini/settings.json` in each account's HOME dir:

```json
{
  "excludeTools": [
    "read_file", "write_file", "read_many_files", "list_directory",
    "glob", "grep", "run_shell_command", "web_fetch", "web_search",
    "edit", "write_todos", "ask_user", "enter_plan_mode",
    "exit_plan_mode", "get_internal_docs", "activate_skill",
    "update_topic", "complete_task", "memory",
    "tracker_create_task", "tracker_update_task", "tracker_get_task",
    "tracker_list_tasks", "tracker_add_dependency", "tracker_visualize"
  ]
}
```

Tool names sourced from `ALL_BUILTIN_TOOL_NAMES` in Gemini CLI `tool-names.ts` (v0.36.0).

**`[UNVERIFIED]`** Whether excluding all tools causes Gemini CLI to error. Test required. **Fallback:** If excluding all tools fails, keep `web_search` as the least dangerous tool and instruct the prompt to not use it.

### 3.6 Message Serialization

OpenAI messages array serialized to flat text for stdin:

```
<<system>> {system_message}

<<user>> {user_message_1}

<<assistant>> {assistant_message_1}

<<user>> {user_message_2}
```

Role delimiters sanitized in user content (zero-width space insertion) to prevent injection. Single-user-message with no history sent raw without delimiters.

**Note on format choice:** The `<<role>>` delimiter format comes from star-cliproxy. Qoder uses a different format (`ROLE:\ntext`). The `<<role>>` format is chosen because it's more visually distinct from natural text and harder to accidentally produce. **Validation test #9** compares both formats — if quality is equivalent, either works.

**`[UNVERIFIED]`** Whether Gemini CLI's non-interactive mode handles stdin the same as `-p`. Whether the `<<role>>` delimiter format produces quality equivalent to `-p`.

### 3.7 Response Parser

Gemini CLI's JSON output format is **unstable**. The parser must try multiple extraction strategies:

1. Direct JSON parse → try fields: `response`, `result`, `text`, `content`
2. Regex extraction of JSON object from mixed output
3. Fallback: raw text with estimated token count (`length / 4`)

**`[UNVERIFIED]`** The exact JSON output format. Run `gemini -o json -p "hello" --yolo --sandbox=false` and capture before implementing.

Field mapping to OpenAI format:

| Gemini CLI field | OpenAI response field |
|------------------|-----------------------|
| Extracted text | `choices[0].message.content` |
| `usageMetadata.promptTokenCount` | `usage.prompt_tokens` |
| `usageMetadata.candidatesTokenCount` | `usage.completion_tokens` |

### 3.8 Error Handling

Three categories, following Qoder's `parseQoderCliFailure()` + `createQoderErrorResponse()` pattern:

| Error | Detection | Response |
|-------|-----------|----------|
| Process failure | Non-zero exit code, timeout, crash | Return 502 with sanitized error |
| Rate limit (429) | Parsed from CLI stderr/output | Return 429 with `Retry-After` |
| Auth failure (401/403) | CLI outputs auth error | Return 502, alert admin |

All errors sanitized before returning to client (strip internal paths, account emails, token fragments).

**Implementation:** Create `parseGeminiCliFailure(stderr, stdout)` and `createGeminiCliErrorResponse(failure)` in `geminiCli.ts`, mirroring `qoderCli.ts` functions of the same pattern. The error response must be a standard `Response` object (JSON body with `error.message`, `error.type`, `error.code`), returned inside the `{ response, url, headers, transformedBody }` shape from `execute()`.

### 3.9 Configuration (MVP)

Configuration is stored in OmniRoute's database as `providerSpecificData` on the provider credentials record (not a YAML file). Fields:

| Field | Default | Description |
|-------|---------|-------------|
| `homeDir` | *(required)* | Absolute path to account's isolated HOME directory |
| `cliPath` | `"gemini"` | Path to `gemini` binary |
| `systemPromptPath` | *(required)* | Path to empty `.md` file for zero system prompt |
| `timeoutMs` | `300000` | Process timeout in milliseconds |
| `email` | *(optional)* | Account email for display/logging only |

Example `providerSpecificData` record:
```json
{
  "homeDir": "/home/overmind/.gemini-sessions/acct-001",
  "cliPath": "gemini",
  "systemPromptPath": "/opt/gemini-sessions/config/empty-prompt.md",
  "timeoutMs": 300000,
  "email": "account@gmail.com"
}
```

A `normalizeGeminiCliProxyProviderData(data)` function validates and sets defaults.

### 3.10 Account Setup (Manual)

1. Create isolated HOME: `mkdir -p /opt/gemini-sessions/acct-001`
2. Copy settings.json with excludedTools into `acct-001/.gemini/settings.json`
3. Authenticate: `HOME=/opt/gemini-sessions/acct-001 gemini` → complete browser OAuth
4. Verify: `HOME=/opt/gemini-sessions/acct-001 gemini -p "hi" --yolo`
5. Configure OmniRoute with `home_dir`

### 3.11 Logging

Use OmniRoute's `ExecutorLog` interface (passed via `execute()` input as `log` parameter) with `log.debug()`, `log.info()`, `log.warn()`, `log.error()` methods. Do not create custom logging.

Example log lines:
```
log.info("GEMINI-CLI-PROXY", "request_id=xxx model=gemini-2.5-pro duration_ms=1234 exit_code=0")
log.warn("GEMINI-CLI-PROXY", "request_id=xxx timeout after 300000ms")
log.error("GEMINI-CLI-PROXY", "request_id=xxx auth failure: <sanitized error>")
```

### 3.12 Security (MVP)

Three things:
1. **stdin delivery** — Prompts via pipe, never CLI arguments (avoids shell injection and ARG_MAX)
2. **Account dir permissions** — `chmod 700` on each account HOME
3. **Secure temp files** — `mkstemps` or equivalent (unpredictable names, exclusive creation)

### 3.13 Registration in LiteLLM

```yaml
model_list:
  - model_name: "gemini-2.5-pro"
    litellm_params:
      model: "openai/gemini-2.5-pro"
      api_base: "http://localhost:20128/v1"
      api_key: "omniroute-api-key"
```

---

## 4. Phase 2: Multi-Account + Streaming

Phase 2 adds production-grade features on top of the working MVP.

### 4.1 Multi-Account Pool

- Multiple pre-authenticated Google accounts, each with isolated HOME directory
- Account pool stored in OmniRoute's DB with metadata:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier |
| `home_dir` | string | Absolute path to isolated HOME |
| `email` | string | Google account email (display only) |
| `status` | enum | `active`, `auth_expired`, `rate_limited`, `tos_violated`, `disabled` |
| `rpm_used` | integer | Requests in current 60s window |
| `rpd_used` | integer | Requests in current 24h window |
| `cooldown_until` | timestamp | Account in cooldown until this time |
| `cooldown_reason` | enum | `quota`, `auth`, `capacity`, `tos` |
| `tos_status` | enum | `unknown`, `healthy`, `flagged` |
| `unsupported_models` | string[] | Models that returned 404 for this account |

**Account selection:** Pick account with most remaining quota. Selection + quota increment must be atomic (mutex).

**Rate limits (per account):**
- RPM: 60 requests per 60-second fixed window
- RPD: 1,000 requests per 24-hour fixed window (reset at midnight UTC)

**Cooldown durations (from KiCk's real-world data):**

| Error Kind | Cooldown |
|------------|----------|
| Quota (429) | 45 minutes |
| Auth (401/403) | 30 days |
| Capacity (429/503 no quota) | 3 minutes |
| ToS violation | Indefinite (manual review) |

**Pool size:** `max(2, min(accounts / 5, 10))` concurrent processes. RAM scales linearly (~40MB/process with `--max-old-space-size=128`).

### 4.2 SSE Streaming

Use `--output-format stream-json` for streaming requests. NDJSON output parsed from pipe (no 8KB buffer issue with streaming).

**`[UNVERIFIED]`** The exact stream-json format. Run `gemini -o stream-json -p "hello" --yolo` and capture before implementing.

Streaming is only added after non-streaming is proven stable.

### 4.3 Admin API

```
GET    /admin/gemini-cli-proxy/accounts          List accounts
POST   /admin/gemini-cli-proxy/accounts          Register account
DELETE /admin/gemini-cli-proxy/accounts/:id      Remove account
POST   /admin/gemini-cli-proxy/accounts/:id/test Test account auth
GET    /admin/gemini-cli-proxy/stats              Pool stats
```

Requires admin API key. Rate limited to 10 req/min.

### 4.4 Error Classification

Sophisticated error handling with per-category retry logic:

| Error Kind | HTTP | Detection | Retry |
|------------|------|-----------|-------|
| Auth | 401, 403 | `TOS_VIOLATION`, `VALIDATION_REQUIRED` | Skip account, alert admin |
| Quota | 429 | `RESOURCE_EXHAUSTED`, `RATE_LIMIT_EXCEEDED` | Try next account |
| Capacity | 429, 503 | "no capacity", "overloaded" | Retry same account after 3min |
| Unsupported model | 404 | "model not found" | Add to account's unsupported list |
| Service error | 500, 503 | Generic | Exponential backoff |

Specific detection strings from KiCk:
- `accountVerificationRequired`
- `termsOfServiceViolation` (what happened to Chris)
- `projectIdMissing`
- `reasoningConfigUnsupported`

Max 10 retries across all accounts before returning 429 to client.

### 4.5 Health Monitoring

- **Every 6 hours:** Spawn lightweight prompt on each account to verify OAuth + API access
- **On registration:** Probe `loadCodeAssist` for ToS violation detection
- **On failure:** Mark account status, alert admin via OmniRoute dashboard

### 4.6 Model Mapping

Register multiple Gemini models:

| Alias | Model |
|-------|-------|
| `gemini-2.5-pro` | `gemini-2.5-pro` |
| `gemini-2.5-flash` | `gemini-2.5-flash` |
| `gemini-2.5-flash-lite` | `gemini-2.5-flash-lite` |

Model selected via `--model` flag or `GEMINI_MODEL` env var.

### 4.7 Observability

- Request ID propagation through queue → account selection → process → response
- Structured JSON logging: `request_id`, `account_id`, `model`, `duration_ms`, `error_kind`, `exit_code`
- Queue metrics: depth, avg wait time, timeout count
- Process metrics: exit codes, stderr snippets, RSS memory, wall time

### 4.8 Security Enhancements

- API key auth on all endpoints
- Admin API rate limiting
- CLI version pinning with startup validation
- Environment variable cleanup (strip OmniRoute-specific vars before spawning)

---

## 5. Phase 3: Multimodal + Optimization

### 5.1 Image Support

Detect `image_url` content in OpenAI requests. Decode base64, write to secure temp file, reference in prompt text.

**`[UNVERIFIED]`** Whether Gemini CLI supports image references in non-interactive stdin mode.

### 5.2 Process Pooling

Maintain a small pool of warm `gemini` processes in interactive mode. Feed prompts via stdin, read responses from stdout. Reduces per-request latency from ~500ms to ~50ms. Requires reverse-engineering Gemini CLI's interactive protocol.

### 5.3 Continuation Handling

When model hits `MAX_TOKENS`, auto-send continuation prompts ("Please continue from where you left off"). Up to 12 passes with overlap detection.

### 5.4 Quota Analytics

Dashboard showing per-account quota usage, error rates, health status. Historical trends. Alerting on anomalous patterns.

---

## 6. Pre-Implementation Validation

**These tests must pass before writing any code.**

| # | Test | Command | Why |
|---|------|---------|-----|
| 1 | Zero system prompt | `GEMINI_SYSTEM_MD=/tmp/empty.md gemini -p "hello" --yolo` | Confirms empty file approach |
| 2 | All tools excluded | Add 25-name excludeTools to settings.json, run test 1 | Confirms full exclusion works |
| 3 | JSON output format | `gemini -o json -p "hello" --yolo --sandbox=false > /tmp/out.json` | Parser design |
| 4 | Stream-json format | `gemini -o stream-json -p "hello" --yolo --sandbox=false` | Streaming parser design |
| 5 | **stdin as prompt** | `echo "hello" \| gemini --yolo --sandbox=false` (no `-p`) | **CRITICAL: Entire approach depends on this** |
| 6 | HOME override | `HOME=/tmp/test-home gemini --version` | Account isolation |
| 7 | OAuth refresh after idle | Use account idle 3+ days | Token longevity |
| 8 | Node heap cap | `NODE_OPTIONS="--max-old-space-size=128" gemini ...` | RAM budget |
| 9 | Message serialization | Send `<<user>>\nhello` via stdin vs `hello`, compare quality | Delimiter validation |
| 10 | Tool exclusion edge case | Exclude all 25 tools, check for errors | No silent failures |
| 11 | **8KB buffer truncation** | Spawn via Node `child_process.spawn` with pipe, generate >8KB output | **Determines whether file redirect is needed** |

---

## 7. Research References

### 7.1 Key Findings from Existing Projects

**star-cliproxy (TypeScript):**
- 8KB stdout buffer truncation bug → use file redirection for non-streaming
- stdin preferred over `-p` flag (ARG_MAX limits on macOS/Linux)
- Multi-strategy JSON parser (unstable output format)
- `<<role>>` message serialization with delimiter sanitization
- Client disconnect → kill subprocess immediately
- Clean parent env vars before spawning

**KiCk (Dart):**
- OAuth client ID/secret: `681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com` / `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`
- Cooldown durations from real-world data (45min quota, 30d auth, 3min capacity)
- ToS violation probing via `loadCodeAssist`
- Warmup requests (loadCodeAssist, listExperiments, fetchAdminControls) sent automatically by CLI
- Continuation handling for MAX_TOKENS (up to 12 passes)
- Detailed error classification with specific detection strings

**code-proxy (Go):**
- Only project that actually spawns the `gemini` binary as a subprocess
- Very simple: `exec.CommandContext(ctx, "gemini", "--model", model, "--sandbox", "--yes")`
- Pipes prompt via stdin, reads stdout line by line
- No streaming, no JSON parsing, no rate limiting (MVP-level)

**CLIProxyAPI (Go):**
- Despite the name, makes direct HTTP calls to `cloudcode-pa.googleapis.com`
- Has sophisticated Go-based OAuth flow, header spoofing, model fallback chains
- NOT subprocess-based

**Brioch/gemini-openai-proxy (Node.js):**
- Imports `@google/gemini-cli-core` as a library
- Library wraps direct HTTP calls to `cloudcode-pa.googleapis.com`
- NOT subprocess-based

### 7.2 Gemini CLI Internals (Source Code)

**System prompt flow (`packages/core/src/prompts/promptProvider.ts`):**
- `GEMINI_SYSTEM_MD` set to falsy → Standard Composition (full coding agent prompt)
- `GEMINI_SYSTEM_MD` set to existing file path → Template File Override (uses file contents)
- `GEMINI_PROMPT_*` env vars disable individual prompt sections

**Tool names (`packages/core/src/tools/tool-names.ts`):**
- `ALL_BUILTIN_TOOL_NAMES` array with 25 entries
- `excludeTools` in `~/.gemini/settings.json` removes tools

**OAuth credentials (`~/.gemini-cli/`):**
- Stored in account's HOME directory
- Auto-refreshed on process startup (long-lived refresh tokens)

**Clearcut telemetry:**
- Sent to `play.googleapis.com/log?format=json&hasfast=true`
- Includes session data, model usage, tool calls, interactive mode flag
- Real Gemini CLI sends this automatically; subprocess approach gets this for free

---

## 8. Open Questions / Risks

1. **stdin prompt delivery** — The most critical unverified assumption. If Gemini CLI doesn't accept stdin as prompt input without `-p`, the entire approach fails. **Fallback:** Use `-p` flag and accept the ARG_MAX risk (truncate prompts over ~128KB, which covers most real requests).
2. **Google's next move** — They called out CLI OAuth third-party usage. Spawning the CLI is legitimate, but they may add non-interactive pattern detection.
3. **CLI output format stability** — JSON and stream-json formats are undocumented public APIs that may change between versions.
4. **Qoder reference not production-validated** — The Qoder subprocess executor is merged but has never run under real load. Edge cases (memory leaks, zombie processes, signal handling) may surface during Gemini CLI testing. Budget time for debugging.
5. **Coexistence with existing executor** — The existing `gemini-cli` executor (direct HTTP) remains untouched. The new `gemini-cli-proxy` executor is a separate provider entry with a different ID.
6. **Gemini CLI auto-updates** — npm auto-updates may change output formats. Consider pinning version.
7. **8KB stdout buffer truncation** — This claim comes from star-cliproxy's experience. It may be specific to their Node.js version or pipe configuration. **Validation needed:** Test whether `child_process.spawn` with `stdio: ['pipe', 'pipe', 'pipe']` actually truncates at 8KB, or if file redirect is truly necessary. If pipe works fine, it simplifies the implementation significantly.
