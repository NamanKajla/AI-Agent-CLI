# AI Agent CLI

A conversational terminal agent that uses an explicit **START → THINK → TOOL → OBSERVE → OUTPUT** loop. It can create real files on disk (for example a Scaler-inspired landing page with header, hero, and footer) and open the result in your default browser.

## Prerequisites

- Node.js 18+
- An API key for at least one provider:
  - [Groq](https://console.groq.com/) (OpenAI-compatible chat API; default provider), **or**
  - [OpenRouter](https://openrouter.ai/) (set `LLM_PROVIDER=openrouter`; many models behind one key — usage is still subject to [OpenRouter’s limits and your credits](https://openrouter.ai/docs)), **or**
  

## Setup

1. Clone this repository.
2. Copy `.env.example` to `.env` and set your key:

   ```bash
   copy .env.example .env
   ```

   Edit `.env`:

   - **`LLM_PROVIDER`** — `groq` (default), `openrouter`, or `gemini`
   - **`GROQ_API_KEY`** — required when using Groq ([Groq Console](https://console.groq.com/keys))
   - **`OPENROUTER_API_KEY`** — required when `LLM_PROVIDER=openrouter` ([OpenRouter keys](https://openrouter.ai/keys)); **`OPENROUTER_MODEL`** — optional (defaults to `openai/gpt-4o-mini`; pick any id from [Models](https://openrouter.ai/models)); **`OPENROUTER_MAX_TOKENS`** — optional (default **8192**; try **4096** or lower if you get **HTTP 402**); **`OPENROUTER_BASE_URL`** — optional (default `https://openrouter.ai/api/v1`); optional **`OPENROUTER_HTTP_REFERER`** / **`OPENROUTER_APP_TITLE`** for OpenRouter attribution headers
   - **`GEMINI_API_KEY`** (or **`GOOGLE_API_KEY`**) — required when `LLM_PROVIDER=gemini` ([Google AI Studio](https://aistudio.google.com/apikey))
   - **`GEMINI_MODEL`** — optional (defaults to `gemini-2.0-flash`)
   - **`GEMINI_MAX_TOKENS`** — optional (default `8192`; raise if replies truncate; free tier models may cap lower)
   - **`GEMINI_RETRY_BASE_MS`** — optional (first backoff for Gemini-only 429 retries; default `max(AGENT_LLM_RETRY_BASE_MS, 5000)`)
   - `GROQ_MODEL` — optional (defaults to `mixtral-8x7b-32768`; see [Groq models](https://console.groq.com/docs/models)). Using a **different** model can hit a separate rate-limit bucket than Llama. Alternatives: `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `gemma2-9b-it`.
   - `GROQ_BASE_URL` — optional (defaults to `https://api.groq.com/openai/v1`)  
   - `GROQ_MAX_TOKENS` — optional (default `32768`; **raising** reduces truncated answers and empty `index.html` when the model tries to output large HTML in one step; Groq may cap per model)  
   - `AGENT_MAX_ITERATIONS` — optional (default `90`, max `500`; more tool steps for heavy clone tasks)  
   - **`AGENT_LLM_COOLDOWN_MS`** — optional (default **`400`** for Groq, **`1200`** for OpenRouter, **`2500`** for Gemini; space out inner-loop calls to reduce 429s)    
   - `AGENT_LLM_RETRY_MAX` / `AGENT_LLM_RETRY_BASE_MS` — optional; retries with backoff on 429 / 5xx for Groq and OpenRouter

3. Install dependencies:

   ```bash
   npm install
   ```

## Run

```bash
npm start
```

Type your instruction at the `You>` prompt. Type `exit` or `quit` to leave.

### Tuning long clone jobs

In `.env`, set **`AGENT_MAX_ITERATIONS`** (integer, default **90**, max **500**) if the agent hits `[STOP] Reached max iterations` before `OUTPUT`. The startup banner prints the active limit.

For **large pages**, raise **`AGENT_FETCH_MAX_CHARS`** (default **450000**, max **900000**) so `fetchUrlContent` returns more HTML/CSS per request before truncation. Optional: **`AGENT_FETCH_MAX_BYTES`**, **`AGENT_FETCH_TIMEOUT_MS`** (see `.env.example`).

### Cloning a specific URL (not random / generic)

The agent only uses **`fetchUrlContent`** on URLs you actually put in your message. The system prompt tells it: **if you pasted a link, that exact page is the only source** (no swapping in another site).

**Use one clear `https://...` line** and say that you want *that* page cloned:

```text
Clone ONLY this page as a static site. Source URL: https://www.example.com/
Put files in output/my-clone and open index.html in the browser when done.
```

Avoid mixing “build something like Scaler” with a URL to another site unless that is what you want. **No URL in the message** → the model improvises from the name you give (e.g. “Scaler-style”). **URL in the message** → it should fetch that URL first.

### Example prompts

- **With URL:** *Clone ONLY https://www.scaler.com/ — fetch this URL first, then build a static replica under `output/scaler-clone` (header, hero, footer), then open `index.html`.*
- **Without URL:** *Build a generic tech-education landing (no specific site). Save under `output/demo` and open it.*

Generated sites are written under `output/` (gitignored). Tools include **`fetchUrlContent`** (download a URL for reference), **`ensureDir`**, **`writeTextFile`**, **`openPath`**, and optionally **`runShellCommand`**.

## How it works

1. **LLM:** Chat completions go to **Groq** via the `openai` npm package (`baseURL` `https://api.groq.com/openai/v1`). Same code path as OpenAI’s API, different host and key.
2. **Outer loop:** `readline` collects your messages and keeps conversation history (plus the system prompt).
3. **Inner loop:** The model returns a single JSON object per turn with a `step`. On `TOOL`, the app runs a local function and injects a **developer** message `{ "step": "OBSERVE", "content": ... }`. The loop ends on `OUTPUT` or after a maximum number of iterations.
4. **Parsing:** Responses are parsed as JSON; optional markdown code fences are stripped. Malformed JSON triggers an OBSERVE hint so the model can recover.

## Project layout

| Path | Purpose |
|------|---------|
| `src/agent.js` | REPL, system prompt, agent loop, tool dispatch |
| `src/tools.js` | `fetchUrlContent`, `ensureDir`, `listDir`, `writeTextFile`, `readTextFile`, `openPath`, `runShellCommand`; files resolve from **project root** |
| `src/cli.js` | `--message`, `--file`, `--once`, `--help` argument parsing |
| `.env.example` | Environment variable template |

## Course submission checklist

Per assignment requirements, submit **both** on the course portal:

| Deliverable | Notes |
|-------------|--------|
| **Public GitHub repository** | Push this project; ensure `.env` is **not** committed (use `.env.example` only). |
| **YouTube video (2–3 minutes)** | Show the CLI running live (visible THINK / TOOL / OBSERVE steps), then open the generated page in a browser. Unlisted is fine if allowed by your instructor. |

### Suggested recording script

1. Briefly show the repo or README.
2. Run `npm start` and paste a prompt asking for a Scaler-style page with files saved and browser opened.
3. Scroll the terminal so grader can see multiple reasoning and tool steps.
4. Show the browser with the final HTML.

## Notes

- **Where files are saved:** `output/...` is always resolved from the **project root** (the folder that contains `package.json`), not from whatever directory your terminal is in. On startup the CLI prints `Project root: ...`. After each `writeTextFile` you should see `[Saved] C:\\...\\full\\path`.
- **No `index.html`?** The LLM sometimes writes only `styles.css` / `app.js` and lies in OUTPUT. Open the folder in Explorer and check: if `index.html` is missing, run the agent again or add `index.html` yourself. The prompt now requires `listDir` before `openPath` and forbids claiming success without an `index.html` write.
- **How close can a “clone” get?** Many sites (including heavy React/Next apps) ship almost empty HTML and render in the browser. The agent only sees what the server returns; it cannot run the site’s JavaScript. It will still try to match layout from that HTML and from linked CSS if it fetches stylesheet URLs.
- For coursework, rewrite copy and respect copyright; use cloning as a **layout exercise**.
- On **Windows**, local pages are opened via **PowerShell** `Start-Process` (works with spaces in paths like `AI Agent CLI`).

## Troubleshooting

**`APIConnectionError` / `getaddrinfo EAI_AGAIN api.groq.com`** — your machine could not resolve or reach Groq’s servers. Check internet access, try another network, turn VPN off temporarily, run `nslookup api.groq.com` or `ipconfig /flushdns` (Windows). On locked-down networks you may need `HTTPS_PROXY` in `.env`. The CLI now prints a short hint instead of only a stack trace for connection failures.

**Rate limit / `429` / “too many requests”** — each THINK/TOOL step is an API call, so long clones hit per-minute caps quickly. On **Groq**: **try another `GROQ_MODEL`** (limits are often per-model); defaults use **Mixtral**. This project **automatically retries** with exponential backoff (uses `retry-after` when the response includes it) and pauses between steps via **`AGENT_LLM_COOLDOWN_MS`** (default **`400`** on Groq, **`1200`** on OpenRouter, **`2500`** on Gemini). If you still overflow: wait a few minutes, raise **`AGENT_LLM_COOLDOWN_MS`**, lower **`AGENT_MAX_ITERATIONS`**, or switch provider (**OpenRouter**, **Gemini**, **Groq**) so quotas differ ([Groq](https://console.groq.com/docs/rate-limits), [OpenRouter FAQ](https://openrouter.ai/docs/faq), [Gemini](https://ai.google.dev/gemini-api/docs/rate-limits)). Gemini responses also log the **last API error line** so you can see daily vs per-minute quota messages.

**OpenRouter `402` / “requires more credits, or fewer max_tokens”** — your balance may not cover the **completion** size you asked for on that model. This CLI **lowers `max_tokens` automatically** when the API says what you can afford, and defaults **`OPENROUTER_MAX_TOKENS`** to **8192**. Set **`OPENROUTER_MAX_TOKENS=4096`** (or lower) in `.env`, or add credits in [OpenRouter settings](https://openrouter.ai/settings/credits).

**Blank or broken `index.html` (empty page, `...` in the file, stray `</content>`)** — HTML is sent inside a JSON `writeTextFile` message. If the model hits the **output token limit**, the reply is cut off and the file can be invalid or tiny. This app sets **`GROQ_MAX_TOKENS`** (default **32768**), **`OPENROUTER_MAX_TOKENS`** (default **8192**), or **`GEMINI_MAX_TOKENS`** (default **8192**), warns when output is truncated (**length**), and appends **VALIDATION** hints after HTML writes so the model is pushed to rewrite. Prefer **structure + copy in `index.html`**, **most styling in `styles.css`**, and raise the provider’s max output tokens if you still see truncation warnings.

## License

MIT (adjust for your course policy if needed).

## Getting Exact/Accurate Website Clones

**If your clones don't look like the original website**, this is usually due to:
1. Using a less capable model (mixtral-8x7b, gpt-4o-mini)
2. Not enough iterations for complex sites
3. Vague prompts that don't emphasize accuracy

### Quick Fix

See **[CLONING_GUIDE.md](./CLONING_GUIDE.md)** for detailed instructions on:
- **Recommended models** for accurate cloning (llama-3.3-70b, gpt-4o, gemini-2.0-flash-exp)
- **Optimal .env configuration** (higher token limits, more iterations)
- **How to write prompts** that produce pixel-perfect results
- **Troubleshooting** common issues (wrong colors, missing sections, tiny CSS files)

### Quick Configuration for Better Clones

Add to your `.env`:
```env
# Use a more capable model
GROQ_MODEL=llama-3.3-70b-versatile
# or
LLM_PROVIDER=openrouter
OPENROUTER_MODEL=openai/gpt-4o

# Increase limits
GROQ_MAX_TOKENS=65536
AGENT_MAX_ITERATIONS=150
AGENT_FETCH_MAX_CHARS=600000
```

### Better Prompt Example

Instead of: "Clone Scaler website"

Use:
```
Clone this EXACT page: https://www.scaler.com/

Requirements:
- Extract EXACT colors (hex codes), fonts, spacing from CSS
- Match EXACT layout structure and section order
- Copy exact text from headings and buttons
- This must be pixel-perfect - compare side-by-side with original
- Save to output/scaler-clone/ and open when done
```
