import { createElement } from "react";
import { act, fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  DEFAULT_MEMORY_CAMERA,
  MAX_MEMORY_3D_LABELS,
  buildMemory3DLabelCandidateIndices,
  createMemory3DSpatialIndex,
  createMemoryLayout3DStepper,
  layoutMemoryNetwork3D,
  memory3DVolumeRatio,
  projectMemoryPoint,
  selectMemory3DLabels
} from "../src/features/memory-workspace/memory-3d.js";
import { MemoryGraph3D } from "../src/features/memory-workspace/MemoryGraph3D.js";
import { createMemoryAtlas } from "../src/features/memory-workspace/memory-atlas.js";
import { createSyntheticMemoryGraph, DemoStore } from "../src/demo/DemoStore.js";
import type { MemoryGraphNode, MemoryGraphEdge } from "@space/contracts";

describe("3D memory projection", () => {
  it("rotates all three axes through a full turn and returns to the same view", () => {
    const point = { x: 45, y: 20, z: 60 };
    const initial = projectMemoryPoint(point, DEFAULT_MEMORY_CAMERA, 2, 1000, 700);
    const rotated = projectMemoryPoint(point, { ...DEFAULT_MEMORY_CAMERA, yaw: DEFAULT_MEMORY_CAMERA.yaw + Math.PI / 2 }, 2, 1000, 700);
    const full = projectMemoryPoint(point, { ...DEFAULT_MEMORY_CAMERA, yaw: DEFAULT_MEMORY_CAMERA.yaw + Math.PI * 2, pitch: DEFAULT_MEMORY_CAMERA.pitch + Math.PI * 2 }, 2, 1000, 700);
    expect(rotated.x).not.toBeCloseTo(initial.x);
    expect(rotated.z).not.toBeCloseTo(initial.z);
    expect(full.x).toBeCloseTo(initial.x);
    expect(full.y).toBeCloseTo(initial.y);
    expect(full.z).toBeCloseTo(initial.z);
  });
  const fixture = () => {
    const nodes: MemoryGraphNode[] = [{ id: "root", label: "gemini.md", type: "SOURCE", sourcePath: "/opt/spaceapp/docs/gemini.md", recordId: null }];
    const edges: MemoryGraphEdge[] = [];
    for (let file = 0; file < 12; file++) {
      const id = `source-${file}`;
      nodes.push({ id, label: `${id}.md`, type: "SOURCE", sourcePath: `/opt/spaceapp/docs/${id}.md`, recordId: null });
      edges.push({ id: `file-${file}`, type: "DERIVED_FROM", source: "root", target: id });
      for (let item = 0; item < 30; item++) {
        const memory = `${id}-memory-${item}`;
        nodes.push({ id: memory, label: memory, type: "MEMORY", sourcePath: `/opt/spaceapp/docs/${id}.md`, recordId: memory });
        edges.push({ id: memory, type: "CONTAINS", source: id, target: memory });
      }
    }
    return { nodes, edges };
  };
  it("occupies a volume, including at a side-on camera angle", () => {
    const { nodes, edges } = fixture();
    const layout = layoutMemoryNetwork3D(nodes, edges);
    expect(layout.size).toBe(nodes.length);
    expect(layout.get("root")).toEqual({ x: 0, y: 0, z: 0 });
    expect(memory3DVolumeRatio([...layout.values()])).toBeGreaterThan(0.35);
    for (const yaw of [0, Math.PI / 2, Math.PI]) {
      const points = [...layout.values()].map(p => projectMemoryPoint(p, { yaw, pitch: 0, zoom: 1 }, 1, 1000, 700));
      const width = Math.max(...points.map(p => p.x)) - Math.min(...points.map(p => p.x));
      const height = Math.max(...points.map(p => p.y)) - Math.min(...points.map(p => p.y));
      expect(width / height).toBeGreaterThan(0.5);
      expect(width / height).toBeLessThan(2);
    }
  });
  it("keeps real source links close and is deterministic under input reordering", () => {
    const { nodes, edges } = fixture();
    const layout = layoutMemoryNetwork3D(nodes, edges);
    const reversed = layoutMemoryNetwork3D([...nodes].reverse(), [...edges].reverse());
    expect([...reversed]).toEqual([...layout]);
    const distance = (a: string, b: string) => { const p = layout.get(a)!, q = layout.get(b)!; return Math.hypot(p.x - q.x, p.y - q.y, p.z - q.z); };
    const related = edges.filter(edge => edge.type === "CONTAINS");
    const meanLink = related.reduce((sum, edge) => sum + distance(edge.source, edge.target), 0) / related.length;
    const meanUnrelated = related.reduce((sum, edge) => sum + distance(edge.source === "source-0" ? "source-6" : "source-0", edge.target), 0) / related.length;
    expect(meanLink).toBeLessThan(meanUnrelated * 0.7);
  });
  it("detects flat geometry even if the plane is tilted", () => {
    const plane = Array.from({ length: 100 }, (_, i) => ({ x: i % 10, y: Math.floor(i / 10), z: i % 10 + Math.floor(i / 10) }));
    expect(memory3DVolumeRatio(plane)).toBeCloseTo(0);
    expect(layoutMemoryNetwork3D([], []).size).toBe(0);
  });

  it("streams incremental layout batches to the exact same converged coordinates as batch layout", () => {
    const { nodes, edges } = fixture();
    const batch = layoutMemoryNetwork3D(nodes, edges);
    const stepper = createMemoryLayout3DStepper(nodes, edges);
    const initialPacked = stepper.packForNodes(nodes);
    expect(initialPacked).toHaveLength(nodes.length * 3);
    expect(memory3DVolumeRatio(initialPacked)).toBeGreaterThan(0.35);

    let batches = 0;
    while (stepper.currentStep < stepper.totalSteps) {
      const res = stepper.step(30);
      batches++;
      if (res.done) break;
    }
    expect(batches).toBeGreaterThan(1);
    expect([...stepper.toMap()]).toEqual([...batch]);
  });

  it("picks hovered/clicked nodes in <= 4 ms at 50,000 nodes using the spatial index", () => {
    const { nodes, edges } = createSyntheticMemoryGraph(50_000, 150_000);
    expect(nodes).toHaveLength(50_000);
    expect(edges).toHaveLength(150_000);

    const stepper = createMemoryLayout3DStepper(nodes, edges);
    const packed = stepper.packForNodes(nodes);
    const baseSizes = new Float32Array(nodes.length);
    for (let i = 0; i < nodes.length; i++) {
      baseSizes[i] = nodes[i]!.type === "SOURCE" ? 4.6 : 2.1;
    }

    const index = createMemory3DSpatialIndex(nodes, packed, baseSizes, DEFAULT_MEMORY_CAMERA, 1.5, 1280, 800);
    const rootScreen = projectMemoryPoint({ x: packed[0]!, y: packed[1]!, z: packed[2]! }, DEFAULT_MEMORY_CAMERA, 1.5, 1280, 800);

    const t0 = performance.now();
    const pickedId = index.pick(rootScreen.x, rootScreen.y);
    const elapsedMs = performance.now() - t0;

    expect(pickedId).not.toBeNull();
    expect(elapsedMs).toBeLessThanOrEqual(4);
    expect(index.pick(-100, -100)).toBeNull();
  });

  it("bounds measured and drawn labels per frame via LOD while preserving core and source priority", () => {
    const { nodes, edges } = createSyntheticMemoryGraph(10_000, 30_000);
    const stepper = createMemoryLayout3DStepper(nodes, edges);
    const packed = stepper.packForNodes(nodes);
    const candidates = buildMemory3DLabelCandidateIndices(nodes, edges);

    const zoomedOut = selectMemory3DLabels({
      nodes,
      packedPositions: packed,
      camera: DEFAULT_MEMORY_CAMERA,
      scale: 1.5,
      width: 1280,
      height: 800,
      focusId: "memory:synthetic-00042",
      candidates
    });

    expect(zoomedOut.labels.length).toBeLessThanOrEqual(MAX_MEMORY_3D_LABELS);
    expect(zoomedOut.measuredCount).toBeLessThanOrEqual(96);
    expect(zoomedOut.labels[0]?.id).toBe("memory:synthetic-00042");
    expect(zoomedOut.labels.some(l => l.isCore)).toBe(true);
    expect(zoomedOut.labels.every(l => l.id === "memory:synthetic-00042" || l.id.startsWith("source:"))).toBe(true);

    const zoomedIn = selectMemory3DLabels({
      nodes,
      packedPositions: packed,
      camera: { ...DEFAULT_MEMORY_CAMERA, zoom: 2.2 },
      scale: 1.5,
      width: 1280,
      height: 800,
      focusId: null,
      candidates
    });
    expect(zoomedIn.labels.length).toBeLessThanOrEqual(MAX_MEMORY_3D_LABELS);
    expect(zoomedIn.measuredCount).toBeLessThanOrEqual(96);
  });

  it("supports deterministic synthetic graphs of 2k, 10k, and 50k nodes in DemoStore with overview limit", async () => {
    const store = new DemoStore();
    store.setSyntheticMemoryGraphSize(50_000);
    const defaultCap = await store.api.memoryGraphOverview({ month: "all", relationMode: "RELATIONS" });
    expect(defaultCap.data.nodes).toHaveLength(2000);
    expect(defaultCap.data.edges).toHaveLength(6000);
    expect(defaultCap.data.totalMatchingNodes).toBe(50_000);
    expect(defaultCap.data.totalMatchingEdges).toBe(150_000);
    expect(defaultCap.data.truncated).toBe(true);

    const full3D = await store.api.memoryGraphOverview({ month: "all", relationMode: "RELATIONS", limit: 50_000 });
    expect(full3D.data.nodes).toHaveLength(50_000);
    expect(full3D.data.edges).toHaveLength(150_000);
    expect(full3D.data.truncated).toBe(false);
  });

  it("preserves accessibility attributes, keyboard controls, and falls back cleanly without WebGL2 or on context loss", async () => {
    const { nodes, edges } = fixture();
    const atlas = createMemoryAtlas(nodes, edges, false);
    const coordinates = layoutMemoryNetwork3D(nodes, edges);
    const onSelectNode = vi.fn();
    const onRendererFallbackChange = vi.fn();

    const { container } = render(
      createElement(MemoryGraph3D, {
        nodes: atlas.nodes,
        edges: atlas.edges,
        positions: atlas.positions,
        coordinates,
        showTitles: true,
        selectedNodeId: null,
        onSelectNode,
        loading: false,
        resetToken: 0,
        onRendererFallbackChange
      })
    );

    const canvas = container.querySelector("canvas[role='application']") as HTMLCanvasElement;
    expect(canvas).not.toBeNull();
    expect(canvas.getAttribute("tabindex")).toBe("0");
    expect(canvas.getAttribute("aria-busy")).toBe("false");
    expect(canvas.getAttribute("data-titles-visible")).toBe("true");
    expect(Number(canvas.getAttribute("data-layout-volume"))).toBeGreaterThan(0.35);
    expect(onRendererFallbackChange).toHaveBeenCalledWith(true);

    fireEvent.keyDown(canvas, { key: "Enter" });
    expect(onSelectNode).toHaveBeenCalledWith("root");

    await act(async () => {
      canvas.dispatchEvent(new Event("webglcontextlost", { cancelable: true }));
    });
    expect(onRendererFallbackChange).toHaveBeenLastCalledWith(true);
  });
});
