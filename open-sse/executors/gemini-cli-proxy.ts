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
  validateHomeDir,
  validateCliPath,
  validateSystemPromptPath,
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

    // Validate homeDir
    const homeDirError = validateHomeDir(homeDir);
    if (homeDirError) {
      return {
        response: createGeminiCliErrorResponse({
          status: 400,
          message: `gemini-cli-proxy config error: ${homeDirError}`,
          code: "config_error",
        }),
        url: "gemini-cli-proxy://local",
        headers,
        transformedBody: body,
      };
    }

    // Validate systemPromptPath
    const systemPromptPath = String(psd.systemPromptPath || "").trim();
    const promptPathError = validateSystemPromptPath(systemPromptPath);
    if (promptPathError) {
      return {
        response: createGeminiCliErrorResponse({
          status: 400,
          message: `gemini-cli-proxy config error: ${promptPathError}`,
          code: "config_error",
        }),
        url: "gemini-cli-proxy://local",
        headers,
        transformedBody: body,
      };
    }

    // Validate cliPath if provided
    const cliPath = String(psd.cliPath || "gemini");
    const cliPathError = validateCliPath(cliPath);
    if (cliPathError) {
      return {
        response: createGeminiCliErrorResponse({
          status: 400,
          message: `gemini-cli-proxy config error: ${cliPathError}`,
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
      cliPath,
    });

    if (!result.ok) {
      if (result.error === "output_overflow") {
        return {
          response: createGeminiCliErrorResponse({
            status: 502,
            message: "Gemini CLI output exceeded maximum size (10 MB). Response was too large.",
            code: "output_overflow",
          }),
          url: "gemini-cli-proxy://local",
          headers,
          transformedBody: body,
        };
      }
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
