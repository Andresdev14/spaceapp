import { createMemoryLayout3DStepper, type MemoryPoint3D } from "./memory-3d.js";
import { layoutMemoryNetwork } from "./memory-network-layout.js";
import type { MemoryGraphNode, MemoryGraphEdge } from "@space/contracts";

export interface MemoryLayoutWorkerMessage {
  positions?: Array<[string, { x: number; y: number }]>;
  depth?: Array<[string, MemoryPoint3D]>;
  packedDepth: Float32Array;
  step: number;
  totalSteps: number;
  done: boolean;
}

self.onmessage = (event: MessageEvent<{ nodes: MemoryGraphNode[]; edges: MemoryGraphEdge[]; skip2D?: boolean }>) => {
  const { nodes, edges, skip2D } = event.data;
  const stepper = createMemoryLayout3DStepper(nodes, edges);
  const initialPacked = stepper.packForNodes(nodes);
  self.postMessage(
    {
      packedDepth: initialPacked,
      step: 0,
      totalSteps: stepper.totalSteps,
      done: stepper.totalSteps === 0
    } satisfies MemoryLayoutWorkerMessage,
    { transfer: [initialPacked.buffer] }
  );

  const batchSteps = nodes.length > 10_000 ? 12 : nodes.length > 2000 ? 18 : 60;
  /** Advances the 3D force layout by one batch and posts transferable packed coordinates. */
  const stepNext = () => {
    const result = stepper.step(batchSteps);
    const packed = stepper.packForNodes(nodes);
    if (!result.done) {
      self.postMessage(
        {
          packedDepth: packed,
          step: result.step,
          totalSteps: result.totalSteps,
          done: false
        } satisfies MemoryLayoutWorkerMessage,
        { transfer: [packed.buffer] }
      );
      setTimeout(stepNext, 0);
      return;
    }
    const includeMaps = !skip2D;
    const positions = includeMaps ? [...layoutMemoryNetwork(nodes, edges)] : undefined;
    const depth = nodes.length <= 2000 ? [...stepper.toMap()] : undefined;
    self.postMessage(
      {
        positions,
        depth,
        packedDepth: packed,
        step: result.step,
        totalSteps: result.totalSteps,
        done: true
      } satisfies MemoryLayoutWorkerMessage,
      { transfer: [packed.buffer] }
    );
  };

  if (stepper.totalSteps > 0) {
    setTimeout(stepNext, 0);
  }
};
