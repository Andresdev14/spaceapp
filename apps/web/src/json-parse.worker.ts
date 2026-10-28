/**
 * One-shot JSON parse worker. Receives the raw response bytes as a transferable buffer, decodes
 * and parses them off the main thread, and posts the parsed value back. Like the main-thread
 * path, a body that is not valid JSON is returned as its raw text instead of throwing.
 */
export interface JsonParseWorkerRequest {
  buffer: ArrayBuffer;
}

export interface JsonParseWorkerResponse {
  value: unknown;
}

self.onmessage = (event: MessageEvent<JsonParseWorkerRequest>) => {
  const text = new TextDecoder("utf-8").decode(event.data.buffer);
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    value = text;
  }
  self.postMessage({ value } satisfies JsonParseWorkerResponse);
};
