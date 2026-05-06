import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

/**
 * Read entire stdin until EOF (for piped prompts).
 */
export async function readStdinUtf8() {
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf8").trim();
}

/**
 * @param {string[]} argv
 */
export function parseCliArgs(argv) {
  /** @type {{ help: boolean; once: boolean; stdin: boolean; message: string | null; file: string | null }} */
  const out = { help: false, once: false, stdin: false, message: null, file: null };

  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--help" || a === "-h") {
      out.help = true;
    } else if (a === "--once") {
      out.once = true;
    } else if (a === "--stdin") {
      out.stdin = true;
    } else if (a === "-m" || a === "--message") {
      const next = argv[++i];
      if (!next) {
        throw new Error(`${a} requires a value`);
      }
      out.message = next;
    } else if (a === "-f" || a === "--file") {
      const next = argv[++i];
      if (!next) {
        throw new Error(`${a} requires a path`);
      }
      out.file = next;
    } else if (!a.startsWith("-")) {
      if (out.message === null && out.file === null) {
        out.message = a;
      }
    } else {
      throw new Error(`Unknown argument: ${a} (try --help)`);
    }
  }

  return out;
}

export function printCliHelp() {
  console.log(`
Usage:
  npm start
  npm start -- --message "Your instructions here"
  npm start -- -f path/to/prompt.txt
  npm start -- -f prompt.txt --once    (run one task, then exit)

Stdin (full prompt via pipe; then exit — no interactive REPL):
  type prompt.txt | npm start -- --stdin
  Get-Content prompt.txt | npm start -- --stdin

Paste in the terminal: use Ctrl+Shift+V (VS Code / Windows Terminal) or right‑click.
If paste still fails, put your prompt in a file and use --file or pipe with --stdin.

Optional env: AGENT_MAX_ITERATIONS (default 90, max 500) — more steps for big clones.
`.trim());
}

/**
 * @param {string | null} filePath
 * @param {string | null} inlineMessage
 */
export async function resolveInitialUserContent(filePath, inlineMessage) {
  if (filePath && inlineMessage) {
    throw new Error("Use either --file or --message, not both.");
  }
  if (filePath) {
    const abs = resolve(filePath);
    const text = await readFile(abs, "utf8");
    return text.trim();
  }
  if (inlineMessage) {
    return inlineMessage.trim();
  }
  return null;
}

/**
 * @param {{ file: string | null; message: string | null; stdin: boolean }} cli
 */
export async function resolveInitialPrompt(cli) {
  const sources = (cli.file ? 1 : 0) + (cli.message ? 1 : 0) + (cli.stdin ? 1 : 0);
  if (sources > 1) {
    throw new Error("Use only one of: --file, --message / positional argument, or --stdin.");
  }
  if (cli.stdin) {
    const text = await readStdinUtf8();
    if (!text) {
      throw new Error(
        "stdin was empty. Example: Get-Content prompt.txt | npm start -- --stdin"
      );
    }
    return text;
  }
  return resolveInitialUserContent(cli.file, cli.message);
}
