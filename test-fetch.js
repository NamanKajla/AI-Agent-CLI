import { fetchUrlContent } from "./src/tools.js";

console.log("Testing fetch to https://www.scaler.com/ ...");

try {
  const result = await fetchUrlContent("https://www.scaler.com/", { maxChars: 400000 });
  console.log("\n✅ SUCCESS!");
  console.log("Status:", result.status);
  console.log("Bytes received:", result.bytesReceived);
  console.log("Chars returned:", result.charsReturned);
  console.log("Truncated:", result.truncated);
  console.log("\nFirst 500 chars of body:");
  console.log(result.body.substring(0, 500));
} catch (error) {
  console.error("\n❌ FAILED!");
  console.error("Error:", error.message);
  console.error("Stack:", error.stack);
}
