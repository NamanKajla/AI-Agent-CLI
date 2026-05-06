import { mkdir, writeFile, stat, readFile, readdir } from "node:fs/promises";
import { dirname, resolve, isAbsolute } from "node:path";
import { fileURLToPath } from "node:url";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const TOOLS_FILE = fileURLToPath(import.meta.url);
const PACKAGE_ROOT = dirname(dirname(TOOLS_FILE));

/**
 * All relative paths (e.g. output/...) resolve from the project root (folder with package.json), not the shell cwd.
 * Override with AI_AGENT_PROJECT_ROOT if needed.
 */
export function getProjectRoot() {
  const fromEnv = process.env.AI_AGENT_PROJECT_ROOT?.trim();
  if (fromEnv) {
    return resolve(fromEnv);
  }
  return PACKAGE_ROOT;
}

/**
 * @param {string} userPath
 */
export function resolveProjectPath(userPath) {
  const p = String(userPath ?? "").trim();
  if (!p) {
    throw new Error("Path is empty");
  }
  if (isAbsolute(p)) {
    return p;
  }
  return resolve(getProjectRoot(), p);
}

const FETCH_TIMEOUT_MS = Number.parseInt(process.env.AGENT_FETCH_TIMEOUT_MS?.trim() || "", 10) || 35_000;
const FETCH_MAX_BYTES = Number.parseInt(process.env.AGENT_FETCH_MAX_BYTES?.trim() || "", 10) || 3_000_000;

/** Upper bound on characters returned (per request). Override with AGENT_FETCH_MAX_CHARS in .env */
const FETCH_MAX_CHARS_CAP = (() => {
  const raw = process.env.AGENT_FETCH_MAX_CHARS?.trim();
  if (!raw) return 450_000;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n >= 60_000 ? Math.min(n, 900_000) : 450_000;
})();

const FETCH_DEFAULT_MAX_CHARS = Math.min(220_000, FETCH_MAX_CHARS_CAP);

/**
 * Download a URL as text (for layout/HTML reference). Large responses are truncated.
 * @param {string} url
 * @param {{ maxChars?: number } | undefined} options
 */
export async function fetchUrlContent(url, options = {}) {
  const u = String(url ?? "").trim();
  if (!/^https?:\/\//i.test(u)) {
    throw new Error("fetchUrlContent: URL must start with http:// or https://");
  }
  const maxChars = Math.min(options.maxChars ?? FETCH_DEFAULT_MAX_CHARS, FETCH_MAX_CHARS_CAP);

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
  try {
    const res = await fetch(u, {
      signal: ctrl.signal,
      redirect: "follow",
      headers: {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
      },
    });

    const contentType = res.headers.get("content-type") ?? "";
    const ab = await res.arrayBuffer();
    const cappedBytes = ab.byteLength > FETCH_MAX_BYTES ? ab.slice(0, FETCH_MAX_BYTES) : ab;
    let text = new TextDecoder("utf-8", { fatal: false }).decode(cappedBytes);
    let truncated = false;
    if (text.length > maxChars) {
      text = text.slice(0, maxChars);
      truncated = true;
    }

    return {
      ok: res.ok,
      requestedUrl: u,
      responseUrl: res.url,
      status: res.status,
      contentType,
      bytesReceived: ab.byteLength,
      charsReturned: text.length,
      maxCharsLimit: FETCH_MAX_CHARS_CAP,
      truncated: truncated || ab.byteLength > FETCH_MAX_BYTES,
      body: text + (truncated || ab.byteLength > FETCH_MAX_BYTES ? "\n\n...[truncated for model context]" : ""),
    };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Download a URL and save decoded text to a project file in one step.
 * Use this for large CSS/JS so the model does not have to paste the body into writeTextFile JSON.
 * @param {string} url
 * @param {string} filePath project-relative or absolute path
 * @param {{ maxChars?: number } | undefined} options optional cap on saved characters (default: full response up to byte limit)
 */
export async function fetchUrlToFile(url, filePath, options = {}) {
  const u = String(url ?? "").trim();
  if (!/^https?:\/\//i.test(u)) {
    throw new Error("fetchUrlToFile: URL must start with http:// or https://");
  }
  const maxChars =
    typeof options.maxChars === "number" && Number.isFinite(options.maxChars)
      ? Math.min(Math.max(0, options.maxChars), FETCH_MAX_CHARS_CAP)
      : undefined;

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
  try {
    const res = await fetch(u, {
      signal: ctrl.signal,
      redirect: "follow",
      headers: {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
      },
    });

    const contentType = res.headers.get("content-type") ?? "";
    const ab = await res.arrayBuffer();
    const cappedBytes = ab.byteLength > FETCH_MAX_BYTES ? ab.slice(0, FETCH_MAX_BYTES) : ab;
    let text = new TextDecoder("utf-8", { fatal: false }).decode(cappedBytes);
    let truncated = ab.byteLength > FETCH_MAX_BYTES;
    if (maxChars !== undefined && text.length > maxChars) {
      text = text.slice(0, maxChars);
      truncated = true;
    }

    const writeRes = await writeTextFile(filePath, text);

    return {
      ok: res.ok,
      requestedUrl: u,
      responseUrl: res.url,
      status: res.status,
      contentType,
      bytesReceived: ab.byteLength,
      charsWritten: text.length,
      truncated,
      maxCharsApplied: maxChars ?? null,
      path: writeRes.path,
      bytesWritten: writeRes.bytesWritten,
      projectRoot: getProjectRoot(),
      note: "Response body saved to disk only (not repeated in this message). For HTML structure use fetchUrlContent; for verbatim CSS/JS bundles use this tool.",
    };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * @param {string} dirPath
 */
export async function ensureDir(dirPath) {
  const abs = resolveProjectPath(dirPath);
  await mkdir(abs, { recursive: true });
  return { ok: true, path: abs, projectRoot: getProjectRoot() };
}

/**
 * @param {string} filePath
 * @param {string} content
 */
export async function writeTextFile(filePath, content) {
  const abs = resolveProjectPath(filePath);
  await mkdir(dirname(abs), { recursive: true });
  await writeFile(abs, content, "utf8");
  return {
    ok: true,
    path: abs,
    bytesWritten: Buffer.byteLength(content, "utf8"),
    projectRoot: getProjectRoot(),
  };
}

const READ_TEXT_FILE_MAX = 120_000;

/**
 * Read a UTF-8 text file from the project (for verification / inspection).
 * @param {string} filePath
 * @param {number} [maxChars]
 */
export async function readTextFile(filePath, maxChars = READ_TEXT_FILE_MAX) {
  const abs = resolveProjectPath(filePath);
  const cap = Math.min(Math.max(maxChars, 1), READ_TEXT_FILE_MAX);
  const buf = await readFile(abs);
  let text = buf.toString("utf8");
  let truncated = false;
  if (text.length > cap) {
    text = text.slice(0, cap);
    truncated = true;
  }
  return {
    ok: true,
    path: abs,
    lengthChars: text.length,
    truncated,
    content: text,
    projectRoot: getProjectRoot(),
  };
}

/**
 * List files in a project directory (verify index.html exists).
 * @param {string} dirPath
 */
export async function listDir(dirPath) {
  const abs = resolveProjectPath(dirPath);
  const entries = await readdir(abs, { withFileTypes: true });
  const list = entries.map((e) => ({
    name: e.name,
    isDirectory: e.isDirectory(),
  }));
  return { ok: true, path: abs, projectRoot: getProjectRoot(), entries: list };
}

/**
 * Opens a file or http(s) URL in the default application (browser for .html).
 * @param {string} targetPath
 */
export async function openPath(targetPath) {
  const raw = String(targetPath ?? "").trim();
  const platform = process.platform;

  /** @type {string} */
  let absOrUrl;
  if (/^https?:\/\//i.test(raw)) {
    absOrUrl = raw;
  } else {
    absOrUrl = resolveProjectPath(raw);
    try {
      await stat(absOrUrl);
    } catch {
      throw new Error(`openPath: file not found: ${absOrUrl}`);
    }
  }

  if (platform === "win32") {
    const psPath = absOrUrl.replace(/'/g, "''");
    await execFileAsync(
      "powershell.exe",
      ["-NoProfile", "-NonInteractive", "-Command", `Start-Process -FilePath '${psPath}'`],
      { windowsHide: true }
    );
    return { ok: true, opened: absOrUrl, projectRoot: getProjectRoot() };
  }

  if (platform === "darwin") {
    await execFileAsync("open", [absOrUrl], { windowsHide: true });
    return { ok: true, opened: absOrUrl, projectRoot: getProjectRoot() };
  }

  await execFileAsync("xdg-open", [absOrUrl], { windowsHide: true });
  return { ok: true, opened: absOrUrl, projectRoot: getProjectRoot() };
}

/**
 * Runs a shell command; fails on non-zero exit. Use sparingly.
 * On Windows, runs via cmd.exe /c.
 * @param {string} cmd
 */
export async function runShellCommand(cmd) {
  if (typeof cmd !== "string" || !cmd.trim()) {
    throw new Error("runShellCommand requires a non-empty string");
  }

  const platform = process.platform;
  let file;
  /** @type {string[]} */
  let args;
  const shell = process.env.ComSpec || "cmd.exe";

  if (platform === "win32") {
    file = shell;
    args = ["/d", "/s", "/c", cmd];
  } else {
    file = "/bin/sh";
    args = ["-lc", cmd];
  }

  const { stdout, stderr } = await execFileAsync(file, args, {
    encoding: "utf8",
    maxBuffer: 10 * 1024 * 1024,
    windowsHide: true,
  });

  const out = (stdout || "").trim();
  const err = (stderr || "").trim();
  return {
    ok: true,
    stdout: out,
    stderr: err,
    summary: [out && `stdout: ${out}`, err && `stderr: ${err}`].filter(Boolean).join("\n") || "(no output)",
  };
}
