import { afterEach, describe, expect, it, vi } from "vitest";

const workerState = vi.hoisted(() => ({ created: 0, transferred: [] as ArrayBuffer[] }));

/** Minimal in-process Worker double: parses like json-parse.worker.ts, off the caller's call stack. */
class FakeJsonWorker {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  terminated = false;
  constructor() { workerState.created++; }
  postMessage(message: { buffer: ArrayBuffer }, transfer?: ArrayBuffer[]) {
    workerState.transferred.push(...(transfer ?? []));
    const text = new TextDecoder("utf-8").decode(message.buffer);
    let value: unknown;
    try { value = JSON.parse(text); } catch { value = text; }
    queueMicrotask(() => this.onmessage?.({ data: { value } } as MessageEvent));
  }
  terminate() { this.terminated = true; }
}

const jsonResponse = (body: string) => new Response(body, { status: 200, headers: { "content-type": "application/json" } });

describe("memory overview JSON parsing", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    workerState.created = 0;
    workerState.transferred = [];
  });

  it("parses the 50k overview through a worker and transfers the buffer", async () => {
    vi.stubGlobal("Worker", FakeJsonWorker);
    const body = JSON.stringify({ data: { nodes: [{ id: "memory:1" }], edges: [] } });
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(body));
    vi.stubGlobal("fetch", fetchMock);
    const parseSpy = vi.spyOn(JSON, "parse");
    const { api } = await import("../src/live-api.js");

    const result = await api.memoryGraphOverview({ limit: 50_000 });

    expect(result).toEqual({ data: { nodes: [{ id: "memory:1" }], edges: [] } });
    expect(workerState.created).toBe(1);
    expect(workerState.transferred).toHaveLength(1);
    // The fake worker is the only caller of JSON.parse for this payload, i.e. not the request() path.
    expect(parseSpy.mock.calls.filter(([text]) => text === body)).toHaveLength(1);
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/admin/memory/graph/overview?limit=50000");
  });

  it("keeps the default 2,000-node overview on the existing main-thread path", async () => {
    vi.stubGlobal("Worker", FakeJsonWorker);
    vi.stubGlobal("fetch", vi.fn().mockImplementation(async () => jsonResponse(JSON.stringify({ data: { nodes: [], edges: [] } }))));
    const { api } = await import("../src/live-api.js");

    await api.memoryGraphOverview({});
    await api.memoryGraphOverview({ limit: 2000 });

    expect(workerState.created).toBe(0);
  });

  it("falls back to the main thread when Workers are unavailable", async () => {
    vi.stubGlobal("Worker", undefined);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(JSON.stringify({ data: { nodes: [], edges: [] } }))));
    const { api } = await import("../src/live-api.js");

    await expect(api.memoryGraphOverview({ limit: 50_000 })).resolves.toEqual({ data: { nodes: [], edges: [] } });
    expect(workerState.created).toBe(0);
  });

  it("returns non-JSON bodies as raw text from the worker, like the main-thread path", async () => {
    vi.stubGlobal("Worker", FakeJsonWorker);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse("not json")));
    const { api } = await import("../src/live-api.js");

    await expect(api.memoryGraphOverview({ limit: 50_000 })).resolves.toBe("not json");
  });

  it("reports HTTP errors through the normal path without using the worker", async () => {
    vi.stubGlobal("Worker", FakeJsonWorker);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: { code: "UNAUTHENTICATED", message: "no" } }), { status: 401, headers: { "content-type": "application/json" } })));
    const { api } = await import("../src/live-api.js");

    await expect(api.memoryGraphOverview({ limit: 50_000 })).rejects.toThrow();
    expect(workerState.created).toBe(0);
  });
});
