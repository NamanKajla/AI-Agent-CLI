import "dotenv/config";
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import OpenAI, {
  APIConnectionError,
  RateLimitError,
  InternalServerError,
  BadRequestError,
} from "openai";
import {
  ensureDir,
  writeTextFile,
  readTextFile,
  listDir,
  openPath,
  runShellCommand,
  fetchUrlContent,
  fetchUrlToFile,
  getProjectRoot,
} from "./tools.js";
import { parseCliArgs, printCliHelp, resolveInitialPrompt } from "./cli.js";
import { geminiChatWithRetry, isGeminiRetriableError, formatGeminiErrorBrief } from "./gemini.js";

function getMaxInnerIterations() {
  const defaultMax = 90;
  const raw = process.env.AGENT_MAX_ITERATIONS?.trim();
  if (!raw) {
    return defaultMax;
  }
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n) || n < 10) {
    return defaultMax;
  }
  return Math.min(n, 500);
}

const MAX_INNER_ITERATIONS = getMaxInnerIterations();
const GROQ_BASE_URL = process.env.GROQ_BASE_URL?.trim() || "https://api.groq.com/openai/v1";
const DEFAULT_GROQ_MODEL = "mixtral-8x7b-32768";
const groqModel = process.env.GROQ_MODEL?.trim() || DEFAULT_GROQ_MODEL;
const DEFAULT_GEMINI_MODEL = "gemini-2.0-flash";
const geminiModelName = process.env.GEMINI_MODEL?.trim() || DEFAULT_GEMINI_MODEL;
const OPENROUTER_BASE_URL = process.env.OPENROUTER_BASE_URL?.trim() || "https://openrouter.ai/api/v1";
const DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini";
const openrouterModel = process.env.OPENROUTER_MODEL?.trim() || DEFAULT_OPENROUTER_MODEL;

/** @type {"groq"|"gemini"|"openrouter"} */
function resolveLlmProvider() {
  const raw = (process.env.LLM_PROVIDER || "groq").trim().toLowerCase();
  if (raw === "gemini") return "gemini";
  if (raw === "openrouter") return "openrouter";
  if (raw === "groq" || raw === "") return "groq";
  console.warn(`[WARN] Unknown LLM_PROVIDER="${raw}"; using groq.`);
  return "groq";
}

const LLM_PROVIDER = resolveLlmProvider();

function getGroqMaxTokens() {
  const d = 32768;
  const raw = process.env.GROQ_MAX_TOKENS?.trim();
  if (!raw) return d;
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n)) return d;
  return Math.min(Math.max(n, 2048), 131072);
}

const GROQ_MAX_TOKENS = getGroqMaxTokens();

function getOpenRouterMaxTokens() {
  const d = 8192;
  const raw = process.env.OPENROUTER_MAX_TOKENS?.trim();
  if (!raw) return d;
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n)) return d;
  return Math.min(Math.max(n, 512), 200_000);
}

const OPENROUTER_MAX_TOKENS = getOpenRouterMaxTokens();

function getGeminiMaxTokens() {
  const d = 8192;
  const raw = process.env.GEMINI_MAX_TOKENS?.trim();
  if (!raw) return d;
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n)) return d;
  return Math.min(Math.max(n, 1024), 32_768);
}

const GEMINI_MAX_TOKENS = getGeminiMaxTokens();

function getAgentContextCharBudget(llm) {
  const raw = process.env.AGENT_CONTEXT_CHAR_BUDGET?.trim();
  if (raw) {
    const n = Number.parseInt(raw, 10);
    if (Number.isFinite(n) && n >= 50_000) {
      return Math.min(n, 1_500_000);
    }
  }
  const outTok =
    llm.provider === "openrouter"
      ? OPENROUTER_MAX_TOKENS
      : llm.provider === "gemini"
        ? GEMINI_MAX_TOKENS
        : GROQ_MAX_TOKENS;

  if (llm.provider === "openrouter") {
    const inputTokTarget = 108_000;
    const effectiveIn = Math.max(inputTokTarget - outTok - 2500, 28_000);
    return Math.floor(effectiveIn * 3.2);
  }

  if (llm.provider === "gemini") {
    const inputTokTarget = 900_000;
    const effectiveIn = Math.max(inputTokTarget - outTok - 4000, 60_000);
    return Math.floor(Math.min(effectiveIn * 3.2, 1_200_000));
  }

  const inputTokTarget = 30_000;
  const effectiveIn = Math.max(inputTokTarget - outTok - 800, 10_000);
  return Math.floor(effectiveIn * 3.2);
}

/**
 * @param {import("openai").OpenAI.ChatCompletionMessageParam} m
 */
function messageContentCharLength(m) {
  const c = m.content;
  if (typeof c === "string") {
    return c.length;
  }
  if (Array.isArray(c)) {
    return c.reduce((acc, part) => {
      if (part && typeof part === "object" && "text" in part && typeof part.text === "string") {
        return acc + part.text.length;
      }
      return acc;
    }, 0);
  }
  return 0;
}

/**
 * @param {import("openai").OpenAI.ChatCompletionMessageParam[]} messages
 */
function totalConversationChars(messages) {
  return messages.reduce((acc, m) => acc + messageContentCharLength(m) + 8, 0);
}

const OBSERVE_TRUNC_NOTE =
  "\n\n...[CLI truncated this observation to fit the model context window — use readTextFile on files under output/, or fetchUrlContent with a smaller maxChars (e.g. 80000), or fetch one asset at a time.]";

/**
 * @param {string} outer
 * @param {number} maxOuterChars
 */
function shrinkObserveDeveloperMessage(outer, maxOuterChars) {
  if (typeof outer !== "string" || outer.length <= maxOuterChars) {
    return outer;
  }
  const headLimit = Math.max(0, maxOuterChars - OBSERVE_TRUNC_NOTE.length);
  try {
    const o = JSON.parse(outer);
    if (o?.step !== "OBSERVE" || typeof o.content !== "string") {
      return outer.slice(0, headLimit) + OBSERVE_TRUNC_NOTE;
    }
    const MIN_INNER = 4_000;
    let inner = o.content;
    for (let guard = 0; guard < 40; guard++) {
      o.content = inner + OBSERVE_TRUNC_NOTE;
      const serialized = JSON.stringify(o);
      if (serialized.length <= maxOuterChars) {
        return serialized;
      }
      if (inner.length <= MIN_INNER) {
        break;
      }
      inner = inner.slice(0, Math.floor(inner.length * 0.58));
    }
    o.content = inner.slice(0, MIN_INNER) + OBSERVE_TRUNC_NOTE;
    const out2 = JSON.stringify(o);
    if (out2.length <= maxOuterChars) {
      return out2;
    }
  } catch {
    /* fall through */
  }
  return outer.slice(0, headLimit) + OBSERVE_TRUNC_NOTE;
}

/**
 * Mutates `messages` in place: keeps system[0]; shortens oldest/largest entries until under budget.
 * @param {import("openai").OpenAI.ChatCompletionMessageParam[]} messages
 * @param {number} charBudget
 */
function pruneMessagesForContext(messages, charBudget) {
  let guard = 0;
  while (totalConversationChars(messages) > charBudget && guard < 250) {
    guard += 1;
    let bestIdx = -1;
    let bestLen = 0;
    for (let i = 1; i < messages.length; i++) {
      const L = messageContentCharLength(messages[i]);
      if (L > bestLen) {
        bestLen = L;
        bestIdx = i;
      }
    }
    if (bestIdx < 0 || bestLen < 5_000) {
      break;
    }
    const m = messages[bestIdx];
    if (typeof m.content !== "string") {
      break;
    }
    const overshoot = Math.max(0, totalConversationChars(messages) - charBudget);
    const shrinkBy = Math.max(25_000, Math.floor(Math.min(bestLen * 0.4, overshoot + 48_000)));
    const targetOuter = Math.max(28_000, bestLen - shrinkBy);

    m.content = shrinkObserveDeveloperMessage(m.content, targetOuter);
  }
}

/**
 * OpenAI / OpenRouter: context overflow often appears as nested JSON in `error.metadata.raw`.
 * @param {unknown} err
 */
function isContextLengthExceeded(err) {
  const status = /** @type {{ status?: number }} */ (err).status;
  if (status !== 400) {
    return false;
  }
  /** @type {string} */
  let rawMeta = "";
  const top = /** @type {{ error?: unknown }} */ (err).error;
  if (top && typeof top === "object" && "metadata" in top && top.metadata && typeof top.metadata === "object") {
    const raw = /** @type {{ metadata?: { raw?: string }}} */ (top).metadata?.raw;
    if (typeof raw === "string") {
      rawMeta = raw;
    }
  }
  if (/context_length_exceeded/i.test(rawMeta)) {
    return true;
  }
  if (/maximum context length/i.test(rawMeta) && /tokens/i.test(rawMeta)) {
    return true;
  }
  const msg = err instanceof Error ? err.message : String(err);
  return /context.?length|maximum context length|too many tokens/i.test(msg);
}

function getLlmRetryMax() {
  const d = 10;
  const raw = process.env.AGENT_LLM_RETRY_MAX?.trim();
  if (!raw) return d;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n >= 0 ? Math.min(n, 50) : d;
}

function getLlmRetryBaseMs() {
  const d = 2500;
  const raw = process.env.AGENT_LLM_RETRY_BASE_MS?.trim();
  if (!raw) return d;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n >= 200 ? Math.min(n, 120_000) : d;
}

function getLlmCooldownMs() {
  const raw = process.env.AGENT_LLM_COOLDOWN_MS?.trim();
  if (raw) {
    const n = Number.parseInt(raw, 10);
    if (Number.isFinite(n) && n >= 0) {
      return Math.min(n, 30_000);
    }
  }
  return LLM_PROVIDER === "gemini" ? 2500 : LLM_PROVIDER === "openrouter" ? 1200 : 400;
}

function getGeminiRetryBaseMs() {
  const raw = process.env.GEMINI_RETRY_BASE_MS?.trim();
  if (raw) {
    const n = Number.parseInt(raw, 10);
    if (Number.isFinite(n) && n >= 500) {
      return Math.min(n, 120_000);
    }
  }
  return Math.max(LLM_RETRY_BASE_MS, 5000);
}

const LLM_RETRY_MAX = getLlmRetryMax();
const LLM_RETRY_BASE_MS = getLlmRetryBaseMs();
const LLM_COOLDOWN_MS = getLlmCooldownMs();
const GEMINI_RETRY_BASE_MS = getGeminiRetryBaseMs();

/**
 * @param {number} ms
 */
function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

/**
 * @param {unknown} headers
 */
function getRetryAfterMs(headers) {
  if (!headers || typeof headers !== "object") {
    return null;
  }
  /** @type {string | null | undefined} */
  let raw;
  if ("get" in headers && typeof headers.get === "function") {
    raw = headers.get("retry-after") ?? headers.get("Retry-After");
  } else {
    raw = /** @type {Record<string, string | undefined>} */ (headers)["retry-after"] ?? /** @type {Record<string, string | undefined>} */ (headers)["Retry-After"];
  }
  if (raw == null || raw === "") {
    return null;
  }
  const sec = Number.parseFloat(String(raw), 10);
  if (!Number.isFinite(sec) || sec < 0) {
    return null;
  }
  return Math.min(sec * 1000, 120_000);
}

/**
 * @param {unknown} err
 * @param {boolean} underCap
 */
function isRetriableApiError(err, underCap) {
  if (!underCap || !(err instanceof Error)) {
    return false;
  }
  if (err instanceof RateLimitError || err instanceof InternalServerError) {
    return true;
  }
  const status = /** @type {{ status?: number }} */ (err).status;
  return status === 429 || (status !== undefined && status >= 500 && status < 600);
}

/**
 * @param {import("openai").OpenAI} client
 * @param {import("openai").OpenAI.ChatCompletionCreateParams} params
 * @param {string} [logLabel]
 */
async function chatCompletionsWithRetry(client, params, logLabel = "Groq") {
  let attempt = 0;
  while (true) {
    try {
      return await client.chat.completions.create(params);
    } catch (err) {
      const canRetry = isRetriableApiError(err, attempt < LLM_RETRY_MAX);
      if (!canRetry) {
        throw err;
      }
      attempt += 1;
      const headers = /** @type {{ headers?: unknown }} */ (err).headers;
      const fromHeader = getRetryAfterMs(headers);
      const exp = Math.min(LLM_RETRY_BASE_MS * 2 ** (attempt - 1), 90_000);
      const wait = fromHeader ?? exp;
      const jitter = Math.random() * 800;
      const totalSec = Math.ceil((wait + jitter) / 1000);
      console.error(
        `\n[${logLabel}] Rate limit or temporary server error. Waiting ~${totalSec}s (retry ${attempt}/${LLM_RETRY_MAX})...`
      );
      await sleep(wait + jitter);
    }
  }
}

/**
 * OpenRouter: HTTP 402 → lower max_tokens; HTTP 429 / 5xx → same backoff as Groq.
 * @param {import("openai").OpenAI} client
 * @param {Omit<import("openai").OpenAI.ChatCompletionCreateParams, "max_tokens"> & { max_tokens?: number }} params
 * @param {string} [logLabel]
 */
async function chatCompletionsOpenRouterAdaptive(client, params, logLabel = "OpenRouter") {
  let maxTokens = typeof params.max_tokens === "number" && params.max_tokens > 0 ? params.max_tokens : 4096;
  const max402Adjusts = 6;
  let count402 = 0;
  let rateAttempt = 0;

  while (true) {
    try {
      return await client.chat.completions.create({
        ...params,
        max_tokens: maxTokens,
      });
    } catch (err) {
      const status = /** @type {{ status?: number }} */ (err).status;

      if (status === 402 && count402 < max402Adjusts) {
        count402 += 1;
        const msg = err instanceof Error ? err.message : String(err);
        const afford = /can only afford (\d+)/i.exec(msg);
        let nextCap = 0;
        if (afford) {
          const n = Number.parseInt(afford[1], 10);
          if (Number.isFinite(n) && n >= 256) {
            nextCap = Math.max(256, Math.floor(n * 0.92));
          }
        }
        if (nextCap <= 0) {
          nextCap = Math.max(512, Math.floor(maxTokens / 2));
        }
        if (nextCap >= maxTokens) {
          nextCap = Math.max(256, maxTokens - Math.max(400, Math.floor(maxTokens * 0.15)));
        }
        if (nextCap >= maxTokens && count402 >= max402Adjusts) {
          throw err;
        }
        console.error(
          `[${logLabel}] HTTP 402 — not enough credits for max_tokens=${maxTokens}. Retrying with max_tokens=${nextCap}. (Set OPENROUTER_MAX_TOKENS lower or add credits.)`
        );
        maxTokens = nextCap;
        continue;
      }

      if (status === 402) {
        throw err;
      }

      if (isRetriableApiError(err, rateAttempt < LLM_RETRY_MAX)) {
        rateAttempt += 1;
        const headers = /** @type {{ headers?: unknown }} */ (err).headers;
        const fromHeader = getRetryAfterMs(headers);
        const exp = Math.min(LLM_RETRY_BASE_MS * 2 ** (rateAttempt - 1), 90_000);
        const wait = fromHeader ?? exp;
        const jitter = Math.random() * 800;
        const totalSec = Math.ceil((wait + jitter) / 1000);
        console.error(
          `\n[${logLabel}] Rate limit or temporary server error. Waiting ~${totalSec}s (retry ${rateAttempt}/${LLM_RETRY_MAX})...`
        );
        await sleep(wait + jitter);
        continue;
      }

      throw err;
    }
  }
}

const tool_map = {
  ensureDir: async (tool_args) => {
    const p = String(tool_args ?? "")
      .trim()
      .replace(/^["']|["']$/g, "");
    return ensureDir(p);
  },
  listDir: async (tool_args) => {
    const p = String(tool_args ?? "")
      .trim()
      .replace(/^["']|["']$/g, "");
    return listDir(p);
  },
  writeTextFile: async (tool_args) => {
    // Handle both object and string formats
    if (typeof tool_args === "object" && tool_args !== null) {
      const parsed = tool_args;
      if (!parsed.path || typeof parsed.content !== "string") {
        throw new Error('tool_args must include "path" and "content" strings');
      }
      return writeTextFile(parsed.path, parsed.content);
    }
    const raw = String(tool_args ?? "").trim();
    const parsed = JSON.parse(raw);
    if (!parsed.path || typeof parsed.content !== "string") {
      throw new Error('tool_args must be JSON: {"path":"relative/or/absolute/path","content":"file contents"}');
    }
    return writeTextFile(parsed.path, parsed.content);
  },
  openPath: async (tool_args) => {
    const p = String(tool_args ?? "")
      .trim()
      .replace(/^["']|["']$/g, "");
    return openPath(p);
  },
  runShellCommand: async (tool_args) => runShellCommand(String(tool_args ?? "")),
  readTextFile: async (tool_args) => {
    // Handle both object and string formats
    if (typeof tool_args === "object" && tool_args !== null) {
      const o = tool_args;
      if (!o.path || typeof o.path !== "string") {
        throw new Error('readTextFile tool_args must include "path" string');
      }
      return readTextFile(o.path, typeof o.maxChars === "number" ? o.maxChars : undefined);
    }
    const raw = String(tool_args ?? "").trim();
    if (raw.startsWith("{")) {
      const o = JSON.parse(raw);
      if (!o.path || typeof o.path !== "string") {
        throw new Error('readTextFile tool_args JSON must include "path" string');
      }
      return readTextFile(o.path, typeof o.maxChars === "number" ? o.maxChars : undefined);
    }
    const p = raw.replace(/^["']|["']$/g, "");
    return readTextFile(p);
  },
  fetchUrlContent: async (tool_args) => {
    // Handle both object and string formats
    if (typeof tool_args === "object" && tool_args !== null) {
      const o = tool_args;
      if (!o.url || typeof o.url !== "string") {
        throw new Error('fetchUrlContent tool_args must include "url" string');
      }
      return fetchUrlContent(o.url, { maxChars: typeof o.maxChars === "number" ? o.maxChars : undefined });
    }
    const raw = String(tool_args ?? "").trim();
    if (raw.startsWith("{")) {
      const o = JSON.parse(raw);
      if (!o.url || typeof o.url !== "string") {
        throw new Error('fetchUrlContent tool_args JSON must include "url" string');
      }
      return fetchUrlContent(o.url, { maxChars: typeof o.maxChars === "number" ? o.maxChars : undefined });
    }
    const url = raw.replace(/^["']|["']$/g, "");
    return fetchUrlContent(url);
  },
  fetchUrlToFile: async (tool_args) => {
    // Handle both object and string formats
    if (typeof tool_args === "object" && tool_args !== null) {
      const o = tool_args;
      if (!o.url || typeof o.url !== "string" || !o.path || typeof o.path !== "string") {
        throw new Error('fetchUrlToFile tool_args must include "url" and "path" strings, optional "maxChars"');
      }
      return fetchUrlToFile(o.url, o.path, {
        maxChars: typeof o.maxChars === "number" ? o.maxChars : undefined,
      });
    }
    const raw = String(tool_args ?? "").trim();
    const o = JSON.parse(raw);
    if (!o.url || typeof o.url !== "string" || !o.path || typeof o.path !== "string") {
      throw new Error(
        'fetchUrlToFile tool_args must be JSON: {"url":"https://…","path":"output/site/styles.css"} with optional "maxChars"'
      );
    }
    return fetchUrlToFile(o.url, o.path, {
      maxChars: typeof o.maxChars === "number" ? o.maxChars : undefined,
    });
  },
};

const SYSTEM_PROMPT = `
You are a terminal AI agent that follows an explicit step protocol. You help the user with tasks by planning, using tools, and summarizing results.

You MUST output exactly one JSON object per message — no markdown fences, no extra text before or after the JSON.

Protocol steps (value of "step"):
- START — acknowledge the user goal in "content" (optional "tool_name", "tool_args" empty or omit).
- THINK — reasoning in "content" only.
- TOOL — set "tool_name" and "tool_args" (see tools below). "content" may briefly state what the tool will do.
- OUTPUT — final user-facing answer in "content". End the task here.

After every TOOL step, the system injects an OBSERVE message with tool results. You never emit OBSERVE yourself.

Rules:
1. One step per message. Wait for OBSERVE after each TOOL before continuing.
2. Use several THINK and TOOL steps — do not try to finish in a single assistant message.
3. URL IS THE SOURCE PAGE: If the user's message contains one or more http(s) URLs, you MUST clone from THAT URL only. The first fetchUrlContent (or the one matching the page they named) MUST use the exact same URL the user pasted (trailing slash optional). Do NOT fetch a different domain, do NOT invent a "similar" site, and do NOT fall back to a generic template when a URL was provided — only if fetchUrlContent clearly fails (explain in OUTPUT). If they gave multiple URLs, clone the one they said is the main page (default: first https URL in their message). If their message has NO http(s) URL, then you may build from the site name they describe.
4. CRITICAL CLONING ACCURACY REQUIREMENTS:
   - You MUST create a PIXEL-PERFECT or NEAR-PIXEL-PERFECT clone of the source website
   - Extract and replicate EXACT colors (hex codes), font families, font sizes, spacing, margins, padding
   - Preserve the EXACT layout structure: grid systems, flexbox arrangements, positioning
   - Copy EXACT text content from headings, paragraphs, buttons, links (paraphrase only for copyright if needed)
   - Match EXACT visual hierarchy: which elements are larger, bolder, more prominent
   - Replicate animations, transitions, hover effects if present in CSS
   - Your goal is that a user comparing the original and clone side-by-side should see minimal differences
5. HIGH-FIDELITY CLONE WORKFLOW (when a URL is provided):
   a) fetchUrlContent the page HTML with a high maxChars (e.g. {"url":"…","maxChars":400000}) so you see structure and inline styles.
   CONTEXT BUDGET: Large bodies in OBSERVE bloat history — prefer short previews via fetchUrlContent for HTML structure only when possible.
   b) DETAILED ANALYSIS PHASE - After fetching HTML, you MUST analyze and document in THINK steps:
      - Color scheme: Extract ALL hex/rgb colors used (backgrounds, text, buttons, borders)
      - Typography: List ALL font families, sizes, weights, line heights
      - Layout structure: Describe grid/flexbox systems, container widths, breakpoints
      - Spacing system: Note padding/margin patterns (e.g., 8px, 16px, 24px, 32px scale)
      - Component inventory: List all UI components (nav, hero, cards, footer sections)
      - Visual hierarchy: Note which elements are emphasized (size, color, position)
   c) From the HTML, find <link rel="stylesheet" href="…"> and @import URLs. Resolve relative hrefs to absolute https URLs. For each important stylesheet (1–3 bundles: main layout, typography, colors), call **fetchUrlToFile** with {"url":"…","path":"output/…/styles.css"} (or extra2.css). This saves the **full** CSS to disk without you pasting it into writeTextFile — writeTextFile MUST NOT be used for multi-hundred-KB CSS (models omit content and produce tiny broken styles.css). Optional: small maxChars on fetchUrlToFile only if a file is huge and you must cap size.
   d) Read inline <style> blocks in the HTML too. Use readTextFile on the saved CSS to extract key styles.
   e) Rebuild the page as static index.html + styles.css + app.js: match **section order**, **nav labels**, **heading hierarchy**, **button/link text**, **EXACT colors** (hex/rgb from CSS), **EXACT font families** (use Google Fonts or system fallbacks that match), **EXACT spacing/alignment** (hero center vs left, header sticky, etc.). Do NOT substitute an unrelated "startup landing" layout. Do NOT change colors, fonts, or spacing arbitrarily.
   f) NEVER paste broken asset URLs for local viewing: paths like /storyblok-assets/, /_next/static/, root-relative /assets/… will NOT work when the user opens output/.../index.html from disk. Replace logos with inline SVG or text wordmark; use CSS gradients/patterns instead of hotlinked hero images unless you use full https:// URLs that still load offline; paraphrase copy per copyright.
5. After fetching: study the returned HTML — section order, headings, nav items, hero, CTAs, footers, classes. Build a static replica in output/ as single-page index.html + styles.css + app.js (rewrite markup/CSS/JS yourself). JS-heavy sites may return a thin shell; reproduce the visible layout from what you got, not a random redesign.
6. NON-NEGOTIABLE page quality: index.html MUST contain a <body> with visible content: <header> (nav/links), a hero/main block with heading + text or CTAs, and a clear bottom region: <footer> OR a full-width contact/CTA bar (e.g. class contact-bar). Every tag must be CLOSED. Never output an empty <body></body>. Never use "..." or <!-- ... --> to skip markup. Put most styling in styles.css so index.html stays structured but not gigantic in one string.
7. If OBSERVE says VALIDATION failed, you MUST fix with another writeTextFile (complete file) before OUTPUT.
8. THREE FILES REQUIRED: index.html AND styles.css AND app.js in the SAME output folder. Typical order: ensureDir → **fetchUrlToFile** (main CSS) OR writeTextFile only for short hand-written CSS → writeTextFile app.js (or fetchUrlToFile for a small third-party script if needed) → writeTextFile index.html. After fetchUrlToFile, check OBSERVE bytesWritten: styles.css should be large (often 10k+ bytes) for real sites; if under ~2KB, you used a placeholder — fix with fetchUrlToFile. Do not stop after only CSS/JS — you MUST write index.html. Never claim the site is ready in OUTPUT unless writeTextFile for index.html has succeeded.
9. If fetch OBSERVE shows truncated:true or the HTML ends mid-tag, fetchUrlContent again with a higher maxChars or fetch the missing CSS file URL separately — do not guess the rest of the layout.
10. Before openPath: call listDir on that output folder and confirm index.html appears in entries. If missing, write index.html (do not OUTPUT excuses).
11. Optional: readTextFile on index.html to verify content before openPath.
12. Keep CSS in styles.css and interactivity in app.js. Respect copyright: paraphrase copy; do not impersonate the original site.
13. ANTI-PATTERNS TO AVOID: single full-screen gradient hero as a substitute for a white/minimal retail site; default blue underlined nav links only; one centered button in an empty page; placeholder Logo img src="/…" paths that are not in the project; writeTextFile for styles.css with "// Full CSS…", ellipses, or under ~2KB when you already fetched a real stylesheet URL — use fetchUrlToFile instead.

JSON shape (always include "step" and "content"; include "tool_name" and "tool_args" only for TOOL):
{ "step": "START | THINK | TOOL | OUTPUT", "content": "string", "tool_name": "string", "tool_args": "string" }

Tools (names must match exactly):
1. fetchUrlContent — tool_args: URL string, OR JSON {"url":"https://…","maxChars":N}. For HTML and inspection. Returns body in OBSERVE (can be huge).
2. fetchUrlToFile — tool_args: JSON only {"url":"https://…","path":"output/site/styles.css"} optional "maxChars":N. Fetches the URL and writes the **full** decoded text to path (no body echoed). **Use for large CSS/JS** from cloned sites so styling actually works.
3. ensureDir(path: string) — create directory; tool_args is the path string, e.g. "output/scaler-clone"
4. listDir(path: string) — list files in a folder under the project (e.g. "output/my-clone"). Use to verify index.html exists before openPath.
5. writeTextFile — tool_args is a JSON string: {"path":"...","content":"..."}. Content must be FULL file source (no ellipses). Best for index.html, small app.js, short CSS; not for pasting fetched 100KB+ stylesheets.
6. readTextFile — tool_args: file path string, or JSON {"path":"output/x/index.html","maxChars":50000}. Returns file text so you can verify what was written.
7. openPath(path: string) — opens file or URL in the default browser (Windows uses PowerShell). tool_args: project-relative path like "output/scaler-clone/index.html", or an "https://..." URL. The file must already exist on disk.
8. runShellCommand(cmd: string) — optional; prefer ensureDir, fetchUrlToFile, and writeTextFile for file tasks

Example pattern for a file write (tool_args is one JSON string):
{ "step": "TOOL", "content": "Write index.html", "tool_name": "writeTextFile", "tool_args": "{\\"path\\":\\"output/demo/index.html\\",\\"content\\":\\"<!DOCTYPE html><html><head><title>Demo</title></head><body><h1>Hi</h1></body></html>\\"}" }
`.trim();

/**
 * @param {string} text
 */
function extractJsonObject(text) {
  if (!text || typeof text !== "string") {
    throw new Error("Empty model response");
  }
  let s = text.trim();
  const fenceMatch = /^```(?:json)?\s*([\s\S]*?)```$/m.exec(s);
  if (fenceMatch) {
    s = fenceMatch[1].trim();
  }
  const start = s.indexOf("{");
  const end = s.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) {
    throw new Error("No JSON object found in response");
  }
  return JSON.parse(s.slice(start, end + 1));
}

/**
 * @param {unknown} data
 */
function formatObservePayload(data) {
  if (data === undefined || data === null) {
    return String(data);
  }
  if (typeof data === "string") {
    return data;
  }
  try {
    return JSON.stringify(data);
  } catch {
    return String(data);
  }
}

/**
 * @param {string} html
 * @returns {string} warning text, or "" if OK
 */
function validateWrittenCssPlaceholder(filePath, css) {
  if (typeof css !== "string") {
    return "VALIDATION: CSS content is not a string.";
  }
  const t = css.trim();
  if (t.length < 800 && /(\/\/\s*Full\s+CSS|\/\*\s*Additional content omitted|\.{3}\s*\/\/)/i.test(t)) {
    return "VALIDATION: styles.css looks like a placeholder. Do not paste partial CSS — use fetchUrlToFile with the stylesheet https URL to save the real bundle.";
  }
  return "";
}

/**
 * @param {string} html
 * @returns {string} warning text, or "" if OK
 */
function validateWrittenLandingHtml(html) {
  if (typeof html !== "string") {
    return "VALIDATION: HTML content is not a string.";
  }
  const t = html.trim();
  if (t.length < 350) {
    return `VALIDATION: index.html is only ${t.length} characters. Provide at least ~350+ chars with real <header>, main/hero, <footer> and text inside <body>.`;
  }
  if (/\.\.\.[\s<]*$|>\s*\.\.\.\s*</.test(t) || /<\/content>/.test(t)) {
    return "VALIDATION: HTML looks truncated or contains invalid fragments (e.g. ... or stray closing tags). Rewrite the FULL file with complete markup only.";
  }
  // Check for placeholder comments
  if (/\[COMPLETE.*BASED ON|TODO|PLACEHOLDER|INSERT.*HERE|ADD.*CONTENT/i.test(t)) {
    return "VALIDATION: HTML contains placeholder comments like [COMPLETE...] or TODO. Write the ACTUAL HTML content - no placeholders, no comments asking to fill in later. Extract real content from the fetched HTML.";
  }
  // Check for ellipsis placeholders
  if (/…\s*\[|…\s*<|>\s*…/.test(t)) {
    return "VALIDATION: HTML contains ellipsis placeholders (…). Write complete HTML markup with real content - no ellipsis, no shortcuts.";
  }
  const body = /<body[^>]*>([\s\S]*?)<\/body>/i.exec(html);
  if (!body) {
    return "VALIDATION: Missing <body>...</body> with real content.";
  }
  const inner = body[1].replace(/<script[\s\S]*?<\/script>/gi, " ").replace(/\s+/g, " ").trim();
  if (inner.length < 100) {
    return "VALIDATION: <body> has almost no visible content. Add nav, headings, paragraphs, hero, footer links.";
  }
  const lower = inner.toLowerCase();
  if (!lower.includes("<header") && !lower.includes('role="banner"') && !lower.includes("<nav")) {
    return "VALIDATION: Add a <header> or <nav> with links so the page is not blank above the fold.";
  }
  if (!lower.includes("<footer") && !lower.includes('role="contentinfo"') && !lower.includes("contact-bar")) {
    return "VALIDATION: Add a <footer> or bottom bar (e.g. class contact-bar / role=\"contentinfo\") with links or contact info.";
  }
  if (/\/storyblok-assets\/|\/_next\/static\/|\/wp-content\/uploads\//i.test(t)) {
    return "VALIDATION: Remove site-specific root paths (e.g. /storyblok-assets/, /_next/static/) that break when opening index.html from disk. Use self-contained markup/CSS, inline SVG, or full https:// asset URLs.";
  }
  return "";
}

/**
 * @param {unknown} err
 */
function printGroqConnectivityHint(err) {
  const cause = err instanceof APIConnectionError ? err.cause : null;
  const reason = cause instanceof Error ? cause.message : err instanceof Error ? err.message : String(err);
  console.error("\n[Network] Could not reach Groq at", GROQ_BASE_URL);
  console.error("Reason:", reason);
  console.error("");
  console.error("This is a connectivity/DNS issue (not your API key). Try:");
  console.error("  · Confirm you are online; toggle Wi‑Fi or plug in Ethernet.");
  console.error("  · Run: nslookup api.groq.com   (or: ping api.groq.com)");
  console.error("  · Pause VPN / switch network / flush DNS: ipconfig /flushdns");
  console.error("  · On restricted networks, set HTTPS_PROXY in .env if your org uses a proxy.");
  console.error("");
}

function printOpenRouterConnectivityHint(err) {
  const cause = err instanceof APIConnectionError ? err.cause : null;
  const reason = cause instanceof Error ? cause.message : err instanceof Error ? err.message : String(err);
  console.error("\n[Network] Could not reach OpenRouter at", OPENROUTER_BASE_URL);
  console.error("Reason:", reason);
  console.error("");
  console.error("This is a connectivity/DNS issue. Try another network or VPN off, or set OPENROUTER_BASE_URL if you use a proxy.");
  console.error("");
}

function printGroqRateLimitHint() {
  console.error("\n[Groq] Rate limit still hit after automatic retries.");
  console.error("  · Wait 2–5 minutes or try again later (free tier is strict).");
  console.error("  · In .env: raise AGENT_LLM_COOLDOWN_MS (e.g. 1000–2000) to space calls, or lower AGENT_MAX_ITERATIONS.");
  console.error("  · Or switch provider: LLM_PROVIDER=openrouter (OPENROUTER_API_KEY) or LLM_PROVIDER=gemini.");
  console.error("  · Docs: https://console.groq.com/docs/rate-limits");
  console.error("");
}

function printOpenRouterRateLimitHint() {
  console.error("\n[OpenRouter] Rate limit (429) or overload still hit after automatic retries.");
  console.error("  · Wait a few minutes — OpenRouter and upstream providers cap requests per minute.");
  console.error("  · In .env: raise AGENT_LLM_COOLDOWN_MS (e.g. 1500–3000) between inner steps; lower AGENT_MAX_ITERATIONS if you only hit limits near the end.");
  console.error("  · Try a cheaper or different OPENROUTER_MODEL; check status: https://openrouter.ai/docs/faq");
  console.error("  · Confirm credits if you see payment-related errors: https://openrouter.ai/settings/credits");
  console.error("");
}

function printOpenRouterPaymentHint(err) {
  const msg = err instanceof Error ? err.message : String(err);
  console.error("\n[OpenRouter] Payment / credits (HTTP 402)");
  if (msg && !msg.startsWith("OpenRouter: exhausted")) {
    console.error(msg);
  }
  console.error("  · Your balance may not cover the requested max_tokens for this model. The CLI will auto-lower max_tokens when the API reports what you can afford; you can set OPENROUTER_MAX_TOKENS in .env (e.g. 4096 or 3072) to start lower.");
  console.error("  · Add credits: https://openrouter.ai/settings/credits");
  console.error("");
}

function printGeminiRateLimitHint(err) {
  console.error("\n[Gemini] Quota or rate limit after automatic retries.");
  if (err != null) {
    console.error("Last error:", formatGeminiErrorBrief(err));
  }
  console.error("  · This CLI calls Gemini once per inner step (THINK/TOOL loop), so free-tier RPM/RPD caps are easy to hit.");
  console.error("  · In .env: AGENT_LLM_COOLDOWN_MS=5000 (or higher), AGENT_MAX_ITERATIONS=40, GEMINI_RETRY_BASE_MS=8000.");
  console.error("  · Try a different GEMINI_MODEL id for your key, or wait until the next minute / daily window resets.");
  console.error("  · If the API says limit 0 or billing required, enable billing for the Google Cloud project or use another key.");
  console.error("  · Docs: https://ai.google.dev/gemini-api/docs/rate-limits");
  console.error("");
}

function printGeminiFailureHint(err) {
  console.error("\n[Gemini] Request failed:", formatGeminiErrorBrief(err));
  console.error("  · Confirm GEMINI_API_KEY or GOOGLE_API_KEY and model id (GEMINI_MODEL).");
  console.error("  · Free tier still has per-minute limits; raise AGENT_LLM_COOLDOWN_MS or retry later.");
  console.error("");
}

/**
 * @typedef {object} LlmHandle
 * @property {'groq'|'gemini'|'openrouter'} provider
 * @property {import('openai').OpenAI | null} openai
 * @property {string | undefined} geminiApiKey
 * @property {string} groqModel
 * @property {string} geminiModel
 * @property {number} groqMaxTokens
 * @property {number} geminiMaxTokens
 */

/**
 * @param {LlmHandle} llm
 * @param {import("openai").OpenAI.ChatCompletionMessageParam[]} messages
 * @param {number} [maxInnerIterations]
 */
async function runAgentTurn(llm, messages, maxInnerIterations = MAX_INNER_ITERATIONS) {
  let iterations = 0;

  while (iterations < maxInnerIterations) {
    iterations += 1;

    if (iterations > 1 && LLM_COOLDOWN_MS > 0) {
      await sleep(LLM_COOLDOWN_MS);
    }

    pruneMessagesForContext(messages, getAgentContextCharBudget(llm));

    let response;
    try {
      if (llm.provider === "gemini") {
        const key = llm.geminiApiKey;
        if (!key) {
          throw new Error("GEMINI_API_KEY or GOOGLE_API_KEY is required when LLM_PROVIDER=gemini");
        }
        response = await geminiChatWithRetry(
          key,
          llm.geminiModel,
          messages,
          llm.geminiMaxTokens,
          LLM_RETRY_MAX,
          GEMINI_RETRY_BASE_MS
        );
      } else {
        if (!llm.openai) {
          throw new Error("LLM client not initialized; set GROQ_API_KEY, OPENROUTER_API_KEY, or Gemini keys per LLM_PROVIDER.");
        }
        const apiLabel = llm.provider === "openrouter" ? "OpenRouter" : "Groq";
        if (llm.provider === "openrouter") {
          response = await chatCompletionsOpenRouterAdaptive(
            llm.openai,
            {
              model: llm.groqModel,
              messages,
              max_tokens: llm.groqMaxTokens,
            },
            apiLabel
          );
        } else {
          response = await chatCompletionsWithRetry(
            llm.openai,
            {
              model: llm.groqModel,
              messages,
              max_tokens: llm.groqMaxTokens,
            },
            apiLabel
          );
        }
      }
    } catch (err) {
      if (llm.provider === "groq" && err instanceof APIConnectionError) {
        printGroqConnectivityHint(err);
        return;
      }
      if (llm.provider === "openrouter" && err instanceof APIConnectionError) {
        printOpenRouterConnectivityHint(err);
        return;
      }
      if (llm.provider === "groq" && err instanceof RateLimitError) {
        printGroqRateLimitHint();
        return;
      }
      if (
        llm.provider === "openrouter" &&
        (err instanceof RateLimitError || /** @type {{ status?: number }} */ (err).status === 429)
      ) {
        printOpenRouterRateLimitHint();
        return;
      }
      if (llm.provider === "openrouter" && /** @type {{ status?: number }} */ (err).status === 402) {
        printOpenRouterPaymentHint(err);
        return;
      }
      if (
        (llm.provider === "openrouter" || llm.provider === "groq") &&
        err instanceof BadRequestError &&
        isContextLengthExceeded(err)
      ) {
        console.warn(
          "\n[WARN] Conversation exceeded the model context window — trimming older tool results and retrying this step. For large sites prefer one HTML fetch plus one reduced maxChars stylesheet (see .env AGENT_CONTEXT_CHAR_BUDGET).\n"
        );
        pruneMessagesForContext(messages, Math.floor(getAgentContextCharBudget(llm) * 0.62));
        iterations -= 1;
        continue;
      }
      if (llm.provider === "gemini") {
        if (isGeminiRetriableError(err)) {
          printGeminiRateLimitHint(err);
          return;
        }
        printGeminiFailureHint(err);
        return;
      }
      throw err;
    }

    const choice = response.choices[0];
    const finishReason = choice?.finish_reason;
    const rawContent = choice?.message?.content ?? "";

    if (finishReason === "length") {
      const tokenHint =
        llm.provider === "gemini"
          ? "Raise GEMINI_MAX_TOKENS in .env"
          : llm.provider === "openrouter"
            ? "Raise OPENROUTER_MAX_TOKENS in .env"
            : "Raise GROQ_MAX_TOKENS in .env";
      console.warn(`\n[WARN] Model reply hit output token limit (truncated). ${tokenHint} or split into smaller TOOL steps.\n`);
      messages.push({ role: "assistant", content: rawContent });
      messages.push({
        role: "developer",
        content: JSON.stringify({
          step: "OBSERVE",
          content:
            `Output was truncated by the token limit. Send the next message as a single small THINK then TOOL: either writeTextFile with a COMPLETE smaller index.html (structure + short copy, heavy styling in styles.css next), or raise max output tokens in .env (${llm.provider === "gemini" ? "GEMINI_MAX_TOKENS" : llm.provider === "openrouter" ? "OPENROUTER_MAX_TOKENS" : "GROQ_MAX_TOKENS"}). Never put ... inside HTML.`,
        }),
      });
      continue;
    }

    /** @type {Record<string, unknown>} */
    let parsed;

    try {
      parsed = extractJsonObject(rawContent);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      messages.push({ role: "assistant", content: rawContent });
      messages.push({
        role: "developer",
        content: JSON.stringify({
          step: "OBSERVE",
          content: `Failed to parse your JSON: ${msg}. Output exactly one JSON object with keys step, content, and for TOOL also tool_name and tool_args. No markdown.`,
        }),
      });
      continue;
    }

    messages.push({ role: "assistant", content: JSON.stringify(parsed) });

    const step = parsed.step;
    const stepStr = typeof step === "string" ? step : "";

    if (stepStr === "START") {
      console.log("\n[START]", parsed.content ?? "");
    } else if (stepStr === "THINK") {
      console.log("\n[THINK]", parsed.content ?? "");
    } else if (stepStr === "TOOL") {
      const toolName = typeof parsed.tool_name === "string" ? parsed.tool_name : "";
      const toolArgs = parsed.tool_args;
      console.log("\n[TOOL]", toolName, toolArgs ?? "");

      const handler = tool_map[toolName];
      if (!handler) {
        messages.push({
          role: "developer",
          content: JSON.stringify({
            step: "OBSERVE",
            content:
              "This tool is not available. Use: fetchUrlContent, fetchUrlToFile, ensureDir, listDir, writeTextFile, readTextFile, openPath, runShellCommand.",
          }),
        });
      } else {
        try {
          const data = await handler(toolArgs);
          if (
            (toolName === "writeTextFile" || toolName === "fetchUrlToFile") &&
            data &&
            typeof data === "object" &&
            "path" in data
          ) {
            const rec = /** @type {{ path: string; bytesWritten?: number }} */ (data);
            console.log(`\n[Saved] ${rec.path} (${rec.bytesWritten ?? "?"} bytes)`);
          }
          if (toolName === "listDir" && data && typeof data === "object" && "entries" in data) {
            const rec = /** @type {{ path: string; entries: { name: string }[] }} */ (data);
            const names = rec.entries.map((x) => x.name).join(", ");
            console.log(`\n[Files in ${rec.path}] ${names || "(empty)"}`);
          }
          let observeText = formatObservePayload(data);
          if (toolName === "writeTextFile") {
            try {
              const o = JSON.parse(String(toolArgs ?? "").trim());
              if (typeof o.path === "string" && typeof o.content === "string" && /\.html$/i.test(o.path)) {
                const v = validateWrittenLandingHtml(o.content);
                if (v) {
                  observeText += `\n\n${v}`;
                }
              }
              if (typeof o.path === "string" && typeof o.content === "string" && /\.css$/i.test(o.path)) {
                const v = validateWrittenCssPlaceholder(o.path, o.content);
                if (v) {
                  observeText += `\n\n${v}`;
                }
              }
            } catch {
              /* invalid writeTextFile tool_args shape */
            }
          }
          if (toolName === "fetchUrlToFile" && data && typeof data === "object" && "path" in data && "bytesWritten" in data) {
            const rec = /** @type {{ path: string; bytesWritten: number }} */ (data);
            if (typeof rec.path === "string" && /\.css$/i.test(rec.path) && rec.bytesWritten < 2048) {
              observeText += `\n\nVALIDATION: ${rec.path} is only ${rec.bytesWritten} bytes — likely truncated or wrong URL. Retry fetchUrlToFile with the full stylesheet https URL, or omit maxChars.`;
            }
          }
          messages.push({
            role: "developer",
            content: JSON.stringify({
              step: "OBSERVE",
              content: observeText,
            }),
          });
        } catch (toolErr) {
          const errText = toolErr instanceof Error ? toolErr.message : String(toolErr);
          messages.push({
            role: "developer",
            content: JSON.stringify({
              step: "OBSERVE",
              content: `Tool error: ${errText}`,
            }),
          });
        }
      }
    } else if (stepStr === "OUTPUT") {
      console.log("\n[OUTPUT]", parsed.content ?? "");
      return;
    } else {
      messages.push({
        role: "developer",
        content: JSON.stringify({
          step: "OBSERVE",
          content: `Unknown step "${stepStr}". Use START, THINK, TOOL, or OUTPUT only.`,
        }),
      });
    }
  }

  console.error(`\n[STOP] Reached max iterations (${maxInnerIterations}).`);
}

async function main() {
  /** @type {{ help: boolean; once: boolean; stdin: boolean; message: string | null; file: string | null }} */
  let cli;
  try {
    cli = parseCliArgs(process.argv);
  } catch (err) {
    console.error(err instanceof Error ? err.message : err);
    process.exitCode = 1;
    return;
  }

  if (cli.help) {
    printCliHelp();
    return;
  }

  let initialContent = null;
  try {
    initialContent = await resolveInitialPrompt(cli);
  } catch (err) {
    console.error(err instanceof Error ? err.message : err);
    process.exitCode = 1;
    return;
  }

  const groqKey = process.env.GROQ_API_KEY?.trim();
  const openrouterKey = process.env.OPENROUTER_API_KEY?.trim();
  const geminiKey = process.env.GEMINI_API_KEY?.trim() || process.env.GOOGLE_API_KEY?.trim();

  if (LLM_PROVIDER === "gemini") {
    if (!geminiKey) {
      console.error(
        "Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env when LLM_PROVIDER=gemini. Get a key at https://aistudio.google.com/apikey (see .env.example)."
      );
      process.exitCode = 1;
      return;
    }
  } else if (LLM_PROVIDER === "openrouter") {
    if (!openrouterKey) {
      console.error(
        "Set OPENROUTER_API_KEY in .env when LLM_PROVIDER=openrouter. Create a key at https://openrouter.ai/keys (see .env.example)."
      );
      process.exitCode = 1;
      return;
    }
  } else if (!groqKey) {
    console.error(
      "Set GROQ_API_KEY in .env (default provider is groq), or set LLM_PROVIDER=openrouter with OPENROUTER_API_KEY, or LLM_PROVIDER=gemini with GEMINI_API_KEY. See .env.example."
    );
    process.exitCode = 1;
    return;
  }

  const referer = process.env.OPENROUTER_HTTP_REFERER?.trim();
  const openRouterHeaders = {
    "X-Title": process.env.OPENROUTER_APP_TITLE?.trim() || "AI Agent CLI",
    ...(referer ? { "HTTP-Referer": referer } : {}),
  };

  /** @type {LlmHandle} */
  const llm = {
    provider: LLM_PROVIDER,
    openai:
      LLM_PROVIDER === "groq" && groqKey
        ? new OpenAI({
            apiKey: groqKey,
            baseURL: GROQ_BASE_URL,
          })
        : LLM_PROVIDER === "openrouter" && openrouterKey
          ? new OpenAI({
              apiKey: openrouterKey,
              baseURL: OPENROUTER_BASE_URL,
              defaultHeaders: openRouterHeaders,
            })
          : null,
    geminiApiKey: geminiKey,
    groqModel: LLM_PROVIDER === "openrouter" ? openrouterModel : groqModel,
    geminiModel: geminiModelName,
    groqMaxTokens: LLM_PROVIDER === "openrouter" ? OPENROUTER_MAX_TOKENS : GROQ_MAX_TOKENS,
    geminiMaxTokens: GEMINI_MAX_TOKENS,
  };

  const maxIter = getMaxInnerIterations();

  console.log("AI Agent CLI — conversational agent with THINK / TOOL / OBSERVE steps.");
  console.log("Paste: Ctrl+Shift+V or right‑click in this terminal; long prompts: npm start -- --file prompt.txt");
  console.log("Stdin: Get-Content prompt.txt | npm start -- --stdin");
  console.log("Commands: type your request, or exit / quit. Run: npm start -- --help\n");
  console.log(`Project root: ${getProjectRoot()}  (output/... files go here, not necessarily your shell cwd)\n`);
  if (LLM_PROVIDER === "gemini") {
    console.log(
      `Using Gemini: ${geminiModelName} (max output: ${GEMINI_MAX_TOKENS}; max inner steps: ${maxIter}; LLM cooldown: ${LLM_COOLDOWN_MS}ms; retries: up to ${LLM_RETRY_MAX})\n`
    );
  } else if (LLM_PROVIDER === "openrouter") {
    console.log(
      `Using OpenRouter: ${openrouterModel} (max_tokens: ${OPENROUTER_MAX_TOKENS}; max inner steps: ${maxIter}; LLM cooldown: ${LLM_COOLDOWN_MS}ms; retries: up to ${LLM_RETRY_MAX})\n`
    );
  } else {
    console.log(
      `Using Groq: ${groqModel} (max_tokens: ${GROQ_MAX_TOKENS}; max inner steps: ${maxIter}; LLM cooldown: ${LLM_COOLDOWN_MS}ms; 429 retries: up to ${LLM_RETRY_MAX})\n`
    );
  }

  /** @type {import("openai").OpenAI.ChatCompletionMessageParam[]} */
  let conversation = [{ role: "system", content: SYSTEM_PROMPT }];

  if (initialContent) {
    const messages = [...conversation, { role: "user", content: initialContent }];
    await runAgentTurn(llm, messages, maxIter);
    conversation = messages;
    if (cli.once || cli.stdin) {
      return;
    }
  }

  const rl = readline.createInterface({ input, output, terminal: true });

  try {
    while (true) {
      const line = await rl.question("You> ");
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      if (/^(exit|quit)$/i.test(trimmed)) {
        break;
      }

      const messages = [...conversation, { role: "user", content: trimmed }];
      await runAgentTurn(llm, messages, maxIter);
      conversation = messages;
    }
  } finally {
    rl.close();
  }
}

main();
