import type { JsonParseWorkerResponse } from "./json-parse.worker.js";

/** True when a module Worker can be created, so large JSON bodies can be parsed off the main thread. */
export const canParseJsonInWorker = (): boolean => typeof Worker !== "undefined";

/**
 * Parses a UTF-8 JSON body in a dedicated worker. The buffer is transferred, not copied, so the
 * caller must not reuse it. A body that is not valid JSON resolves to its raw text, matching the
 * main-thread path. Rejects only when the worker itself fails.
 */
export function parseJsonInWorker(buffer: ArrayBuffer): Promise<unknown> {
  return new Promise<unknown>((resolve, reject) => {
    const worker = new Worker(new URL("./json-parse.worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (event: MessageEvent<JsonParseWorkerResponse>) => {
      worker.terminate();
      resolve(event.data.value);
    };
    worker.onerror = event => {
      worker.terminate();
      reject(new Error(event.message || "JSON parse worker unavailable"));
    };
    worker.postMessage({ buffer }, [buffer]);
  });
}
