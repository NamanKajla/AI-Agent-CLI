import { fetchUrlContent } from "./src/tools.js";

// Simulate what the agent does
const tool_map = {
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
};

console.log("Testing tool handler with object format...");

try {
  // This is what the LLM is sending
  const tool_args = { url: 'https://www.scaler.com/', maxChars: 400000 };
  const result = await tool_map.fetchUrlContent(tool_args);
  console.log("\n✅ SUCCESS!");
  console.log("Status:", result.status);
  console.log("Chars returned:", result.charsReturned);
} catch (error) {
  console.error("\n❌ FAILED!");
  console.error("Error:", error.message);
}
