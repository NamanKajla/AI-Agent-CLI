import { GoogleGenerativeAI } from "@google/generative-ai";

/**
 * @param {import("openai").OpenAI.ChatCompletionMessageParam[]} messages
 */
export function getSystemInstruction(messages) {
  return messages
    .filter((m) => m.role === "system")
    .map((m) => (typeof m.content === "string" ? m.content : JSON.stringify(m.content)))
    .filter(Boolean)
    .join("\n\n");
}

/**
 * Maps OpenAI-style chat messages to Gemini `contents` (strict user/model alternation).
 * @param {import("openai").OpenAI.ChatCompletionMessageParam[]} messages
 */
export function messagesToGeminiContents(messages) {
  const rest = messages.filter((m) => m.role !== "system");
  /** @type {{ role: "user" | "model"; parts: { text: string }[] }[]} */
  const contents = [];

  for (const m of rest) {
    const raw = typeof m.content === "string" ? m.content : JSON.stringify(m.content);
    const text = m.role === "developer" ? `Tool observation:\n${raw}` : raw;
    const role = m.role === "assistant" ? "model" : "user";
    const last = contents[contents.length - 1];
    if (last && last.role === role) {
      last.parts[0].text += `\n\n${text}`;
    } else {
      contents.push({ role, parts: [{ text }] });
    }
  }

  return contents;
}

/**
 * @param {string | undefined} reason
 */
function geminiFinishToOpenAi(reason) {
  if (!reason) {
    return "stop";
  }
  const u = String(reason).toUpperCase();
  if (u === "MAX_TOKENS" || u === "LENGTH") {
    return "length";
  }
  return "stop";
}

/**
 * @param {unknown} err
 */
export function isGeminiRetriableError(err) {
  const msg = err instanceof Error ? err.message : String(err);
  const status = /** @type {{ status?: number }} */ (err).status;
  return (
    status === 429 ||
    status === 503 ||
    /429|RESOURCE_EXHAUSTED|quota|rate|503|UNAVAILABLE|overloaded/i.test(msg)
  );
}

/**
 * Parse Google RPC duration like "42s", "1m" to ms.
 * @param {string} d
 * @returns {number | null}
 */
function parseRpcDurationToMs(d) {
  const s = String(d).trim();
  const m = /^(\d+(?:\.\d+)?)\s*([sm])?$/i.exec(s);
  if (!m) return null;
  const n = Number.parseFloat(m[1]);
  if (!Number.isFinite(n)) return null;
  const unit = (m[2] || "s").toLowerCase();
  return unit === "m" ? Math.round(n * 60_000) : Math.round(n * 1000);
}

/**
 * Uses RetryInfo in error details or "retry in Ns" in the message when present.
 * @param {unknown} err
 * @returns {number | null} wait time in ms
 */
export function getSuggestedRetryMsFromGeminiError(err) {
  const msg = err instanceof Error ? err.message : String(err);
  const retryIn = /retry in ([\d.]+)\s*s(?:ec(?:onds?)?)?/i.exec(msg);
  if (retryIn) {
    const sec = Number.parseFloat(retryIn[1]);
    if (Number.isFinite(sec)) {
      return Math.min(Math.ceil(sec * 1000) + 500, 300_000);
    }
  }
  const details = /** @type {{ errorDetails?: unknown[] }} */ (err).errorDetails;
  if (Array.isArray(details)) {
    for (const d of details) {
      if (!d || typeof d !== "object") continue;
      const rec = /** @type {Record<string, unknown>} */ (d);
      const delay = rec.retryDelay;
      if (typeof delay === "string") {
        const ms = parseRpcDurationToMs(delay);
        if (ms != null) return Math.min(ms + 500, 300_000);
      }
    }
  }
  return null;
}

/**
 * One-line summary for logs (no stack).
 * @param {unknown} err
 */
export function formatGeminiErrorBrief(err) {
  if (err instanceof Error) {
    const st = /** @type {{ status?: number }} */ (err).status;
    const line = err.message.split("\n")[0]?.trim() || err.message;
    const prefix = st != null ? `HTTP ${st} — ` : "";
    if (line.length > 320) {
      return `${prefix}${line.slice(0, 317)}…`;
    }
    return `${prefix}${line}`;
  }
  return String(err).slice(0, 320);
}

/**
 * @param {string} apiKey
 * @param {string} modelName
 * @param {import("openai").OpenAI.ChatCompletionMessageParam[]} messages
 * @param {number} maxOutputTokens
 */
export async function geminiGenerateContent(apiKey, modelName, messages, maxOutputTokens) {
  const systemInstruction = getSystemInstruction(messages);
  const contents = messagesToGeminiContents(messages);

  if (contents.length === 0) {
    throw new Error("gemini: no messages after system prompt");
  }

  const genAI = new GoogleGenerativeAI(apiKey);
  const genModel = genAI.getGenerativeModel({
    model: modelName,
    ...(systemInstruction
      ? {
          systemInstruction: {
            role: "system",
            parts: [{ text: systemInstruction }],
          },
        }
      : {}),
  });

  const result = await genModel.generateContent({
    contents,
    generationConfig: {
      maxOutputTokens,
      temperature: 0.35,
    },
  });

  const response = result.response;
  const candidate = response.candidates?.[0];
  const finishReason = candidate?.finishReason;
  let rawContent = "";
  try {
    rawContent = response.text();
  } catch {
    rawContent = "";
  }

  return {
    rawContent,
    finishReason: geminiFinishToOpenAi(finishReason),
    safetyRatings: candidate?.safetyRatings,
  };
}

/**
 * OpenAI-shaped response for shared handling in agent loop.
 */
export function wrapGeminiAsChatCompletion(payload) {
  return {
    choices: [
      {
        finish_reason: payload.finishReason,
        message: { role: "assistant", content: payload.rawContent },
      },
    ],
  };
}

/**
 * @param {string} apiKey
 * @param {string} modelName
 * @param {import("openai").OpenAI.ChatCompletionMessageParam[]} messages
 * @param {number} maxOutputTokens
 * @param {number} retryMax
 * @param {number} retryBaseMs
 */
export async function geminiChatWithRetry(
  apiKey,
  modelName,
  messages,
  maxOutputTokens,
  retryMax,
  retryBaseMs
) {
  let attempt = 0;
  while (true) {
    try {
      const payload = await geminiGenerateContent(apiKey, modelName, messages, maxOutputTokens);
      return wrapGeminiAsChatCompletion(payload);
    } catch (err) {
      const canRetry = isGeminiRetriableError(err) && attempt < retryMax;
      if (!canRetry) {
        throw err;
      }
      if (attempt === 0) {
        console.error(`\n[Gemini] ${formatGeminiErrorBrief(err)}`);
      }
      attempt += 1;
      const suggested = getSuggestedRetryMsFromGeminiError(err);
      const backoff = Math.min(retryBaseMs * 2 ** (attempt - 1), 120_000);
      const exp = suggested ?? backoff;
      const jitter = Math.random() * 1000;
      const totalSec = Math.ceil((exp + jitter) / 1000);
      const src = suggested ? "server hint" : "backoff";
      console.error(
        `[Gemini] Rate limited / overloaded. Waiting ~${totalSec}s (${src}; retry ${attempt}/${retryMax})...`
      );
      await new Promise((r) => setTimeout(r, exp + jitter));
    }
  }
}
