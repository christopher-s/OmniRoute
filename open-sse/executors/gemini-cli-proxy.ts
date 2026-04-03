import {
  BaseExecutor,
  mergeUpstreamExtraHeaders,
  type ExecuteInput,
  type ProviderCredentials,
} from "./base.ts";
import { PROVIDERS } from "../config/constants.ts";
import {
  buildGeminiCompletionPayload,
  buildGeminiPrompt,
  createGeminiCliErrorResponse,
  extractTextFromGeminiOutput,
  normalizeGeminiCliProxyProviderData,
  parseGeminiCliFailure,
  runGeminiCliCommand,
} from "../services/geminiCli.ts";

export class GeminiCliProxyExecutor extends BaseExecutor {
  constructor() {
    super("gemini-cli-proxy", PROVIDERS["gemini-cli-proxy"]);
  }

  buildHeaders(_credentials: ProviderCredentials, stream = true): Record<string, string> {
    return {
      "Content-Type": "application/json",
      ...(stream ? { Accept: "text/event-stream" } : {}),
    };
  }

  async refreshCredentials() {
    // Gemini CLI handles OAuth refresh internally via stored tokens in ~/.gemini-cli/
    return null;
  }

  async execute({
    model,
    body,
    stream: _stream,
    credentials,
    signal,
    upstreamExtraHeaders,
  }: ExecuteInput) {
    const headers = this.buildHeaders(credentials, false);
    mergeUpstreamExtraHeaders(headers, upstreamExtraHeaders);

    const psd = normalizeGeminiCliProxyProviderData(
      credentials.providerSpecificData || {}
    );

    const homeDir = String(psd.homeDir || "").trim();
    if (!homeDir) {
      return {
        response: createGeminiCliErrorResponse({
          status: 400,
          message:
            "gemini-cli-proxy requires homeDir in provider configuration. Set the homeDir field in provider-specific data.",
          code: "config_error",
        }),
        url: "gemini-cli-proxy://local",
        headers,
        transformedBody: body,
      };
    }

    const systemPromptPath = String(psd.systemPromptPath || "").trim();
    if (!systemPromptPath) {
      return {
        response: createGeminiCliErrorResponse({
          status: 400,
          message:
            "gemini-cli-proxy requires systemPromptPath in provider configuration. Create an empty .md file and set the path.",
          code: "config_error",
        }),
        url: "gemini-cli-proxy://local",
        headers,
        transformedBody: body,
      };
    }

    const prompt = buildGeminiPrompt(body);
    const timeoutMs = Number(psd.timeoutMs) || 300_000;

    const result = await runGeminiCliCommand({
      prompt,
      model,
      homeDir,
      systemPromptPath,
      signal,
      timeoutMs,
    });

    if (!result.ok) {
      const failure = parseGeminiCliFailure(result.stderr || result.error || "", result.stdout);
      return {
        response: createGeminiCliErrorResponse(failure),
        url: "gemini-cli-proxy://local",
        headers,
        transformedBody: body,
      };
    }

    const assistantText = extractTextFromGeminiOutput(result.stdout);
    const payload = buildGeminiCompletionPayload({
      model,
      text: assistantText,
      rawOutput: result.stdout,
    });

    return {
      response: new Response(JSON.stringify(payload), {
        status: 200,
        headers: {
          "Content-Type": "application/json",
        },
      }),
      url: "gemini-cli-proxy://local",
      headers,
      transformedBody: body,
    };
  }
}

export default GeminiCliProxyExecutor;
