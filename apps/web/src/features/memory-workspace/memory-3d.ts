import type { MemoryGraphEdge, MemoryGraphNode } from "@space/contracts";
import { isMemoryCoreLabel } from "./memory-graph-visuals.js";

export type MemoryPoint3D = { x: number; y: number; z: number };
export type MemoryCamera3D = { yaw: number; pitch: number; zoom: number };
export const DEFAULT_MEMORY_CAMERA: MemoryCamera3D = { yaw: 0.3, pitch: -0.15, zoom: 1 };
export const MAX_MEMORY_3D_LABELS = 42;
export const MAX_MEMORY_3D_LABEL_CANDIDATES = 96;

// Avalanche each seed so the three axes are independent even for similar file IDs.
const unit = (seed: string) => {
  let hash = 2166136261;
  for (const character of seed) hash = Math.imul(hash ^ character.charCodeAt(0), 16777619);
  hash = Math.imul(hash ^ (hash >>> 16), 0x7feb352d);
  hash = Math.imul(hash ^ (hash >>> 15), 0x846ca68b);
  return ((hash ^ (hash >>> 16)) >>> 0) / 4294967295;
};

const compareAscii = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);

export interface MemoryLayout3DStepResult {
  step: number;
  totalSteps: number;
  done: boolean;
}

export interface MemoryLayout3DStepper {
  readonly count: number;
  readonly totalSteps: number;
  readonly currentStep: number;
  step: (batchSteps?: number) => MemoryLayout3DStepResult;
  toMap: () => Map<string, MemoryPoint3D>;
  packForNodes: (targetNodes?: MemoryGraphNode[]) => Float32Array;
}

/**
 * Incremental deterministic 3D force layout stepper.
 * For graphs <= 2,000 nodes, produces bit-for-bit identical output to the batch layout.
 * For large graphs (up to 50,000+ nodes), uses a flat spatial hash with bounded per-cell
 * repulsion so streaming steps stay interactive without O(N^2) cell congestion.
 */
export function createMemoryLayout3DStepper(
  nodes: MemoryGraphNode[],
  edges: MemoryGraphEdge[],
  options?: { totalSteps?: number }
): MemoryLayout3DStepper {
  const count = nodes.length;
  const largeGraph = count > 2000;
  const ordered = largeGraph
    ? [...nodes].sort((a, b) => compareAscii(a.id, b.id))
    : [...nodes].sort((a, b) => a.id.localeCompare(b.id));
  const index = new Map<string, number>();
  const isSource = new Uint8Array(count);
  let root = -1;
  for (let i = 0; i < count; i++) {
    const node = ordered[i]!;
    index.set(node.id, i);
    if (node.type === "SOURCE") {
      isSource[i] = 1;
      if (root < 0 && isMemoryCoreLabel(node.label)) root = i;
    }
  }

  const px = new Float64Array(count);
  const py = new Float64Array(count);
  const pz = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    const id = ordered[i]!.id;
    px[i] = (unit(`${id}:x`) - 0.5) * 260;
    py[i] = (unit(`${id}:y`) - 0.5) * 260;
    pz[i] = (unit(`${id}:z`) - 0.5) * 260;
  }
  if (root >= 0) {
    px[root] = 0;
    py[root] = 0;
    pz[root] = 0;
  }

  const sortedEdges = largeGraph
    ? [...edges].sort((a, b) => compareAscii(a.id, b.id))
    : [...edges].sort((a, b) => a.id.localeCompare(b.id));

  const linkA: number[] = [];
  const linkB: number[] = [];
  const linkLength: number[] = [];
  const linkStrength: number[] = [];
  for (let i = 0; i < sortedEdges.length; i++) {
    const edge = sortedEdges[i]!;
    const a = index.get(edge.source);
    const b = index.get(edge.target);
    if (a === undefined || b === undefined || a === b) continue;
    const fileLink = isSource[a] === 1 && isSource[b] === 1;
    linkA.push(a);
    linkB.push(b);
    linkLength.push(fileLink ? 110 : edge.type === "CONTAINS" ? 34 : 55);
    linkStrength.push(fileLink ? 0.035 : edge.type === "CONTAINS" ? 0.028 : 0.004);
  }
  const linkCount = linkA.length;
  const linkATyped = Int32Array.from(linkA);
  const linkBTyped = Int32Array.from(linkB);
  const linkLenTyped = Float64Array.from(linkLength);
  const linkStrTyped = Float64Array.from(linkStrength);

  // Initialize records around their actual file, in a volume rather than a plane.
  for (let i = 0; i < edges.length; i++) {
    const edge = edges[i]!;
    if (edge.type !== "CONTAINS") continue;
    const a = index.get(edge.source);
    const b = index.get(edge.target);
    if (a === undefined || b === undefined || isSource[a] !== 1 || ordered[b]!.type !== "MEMORY") continue;
    const id = ordered[b]!.id;
    px[b] = px[a]! + (unit(`${id}:dx`) - 0.5) * 70;
    py[b] = py[a]! + (unit(`${id}:dy`) - 0.5) * 70;
    pz[b] = pz[a]! + (unit(`${id}:dz`) - 0.5) * 70;
  }

  const fx = new Float64Array(count);
  const fy = new Float64Array(count);
  const fz = new Float64Array(count);
  const totalSteps = options?.totalSteps ?? (count > 10_000 ? 48 : count > 2000 ? 72 : 180);
  let currentStep = 0;

  // Flat hash table for large graphs to avoid per-step Map/Array allocations.
  const hashMask = 16383;
  const cellHead = largeGraph ? new Int32Array(hashMask + 1) : null;
  const cellCount = largeGraph ? new Uint8Array(hashMask + 1) : null;
  const cellNext = largeGraph ? new Int32Array(count) : null;
  const neighborOffsets = [
    [0, 0, 0],
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
  ] as const;

  const runExactStep = (stepIndex: number) => {
    fx.fill(0);
    fy.fill(0);
    fz.fill(0);
    const grid = new Map<string, number[]>();
    for (let i = 0; i < count; i++) {
      const key = `${Math.floor(px[i]! / 32)},${Math.floor(py[i]! / 32)},${Math.floor(pz[i]! / 32)}`;
      let bucket = grid.get(key);
      if (!bucket) {
        bucket = [];
        grid.set(key, bucket);
      }
      bucket.push(i);
    }
    for (let i = 0; i < count; i++) {
      const pxi = px[i]!, pyi = py[i]!, pzi = pz[i]!;
      const gx = Math.floor(pxi / 32), gy = Math.floor(pyi / 32), gz = Math.floor(pzi / 32);
      const srcI = isSource[i] === 1;
      for (let x = gx - 1; x <= gx + 1; x++) {
        for (let y = gy - 1; y <= gy + 1; y++) {
          for (let z = gz - 1; z <= gz + 1; z++) {
            const bucket = grid.get(`${x},${y},${z}`);
            if (!bucket) continue;
            for (let k = 0; k < bucket.length; k++) {
              const j = bucket[k]!;
              if (j <= i) continue;
              const dx = pxi - px[j]!, dy = pyi - py[j]!, dz = pzi - pz[j]!;
              const distanceSquared = Math.max(1, dx * dx + dy * dy + dz * dz);
              const strength = (srcI || isSource[j] === 1 ? 28 : 12) / distanceSquared;
              fx[i]! += dx * strength; fy[i]! += dy * strength; fz[i]! += dz * strength;
              fx[j]! -= dx * strength; fy[j]! -= dy * strength; fz[j]! -= dz * strength;
            }
          }
        }
      }
    }
    for (let k = 0; k < linkCount; k++) {
      const a = linkATyped[k]!, b = linkBTyped[k]!;
      const dx = px[b]! - px[a]!, dy = py[b]! - py[a]!, dz = pz[b]! - pz[a]!;
      const distance = Math.max(0.01, Math.hypot(dx, dy, dz));
      const spring = (distance - linkLenTyped[k]!) * linkStrTyped[k]! / distance;
      fx[a]! += dx * spring; fy[a]! += dy * spring; fz[a]! += dz * spring;
      fx[b]! -= dx * spring; fy[b]! -= dy * spring; fz[b]! -= dz * spring;
    }
    const cooling = 1 - stepIndex / 230;
    for (let i = 0; i < count; i++) {
      if (i === root) continue;
      const dx = fx[i]! - px[i]! * 0.0005, dy = fy[i]! - py[i]! * 0.0005, dz = fz[i]! - pz[i]! * 0.0005;
      const stepScale = Math.min(1, 4 / Math.max(0.01, Math.hypot(dx, dy, dz))) * cooling;
      px[i]! += dx * stepScale; py[i]! += dy * stepScale; pz[i]! += dz * stepScale;
    }
  };

  const runLargeGraphStep = (stepIndex: number) => {
    fx.fill(0);
    fy.fill(0);
    fz.fill(0);
    cellHead!.fill(-1);
    cellCount!.fill(0);
    const maxPerCell = 6;
    for (let i = count - 1; i >= 0; i--) {
      const gx = Math.floor(px[i]! / 32), gy = Math.floor(py[i]! / 32), gz = Math.floor(pz[i]! / 32);
      const h = (Math.imul(gx, 73856093) ^ Math.imul(gy, 19349663) ^ Math.imul(gz, 83492791)) & hashMask;
      if (cellCount![h]! >= maxPerCell) continue;
      cellCount![h]!++;
      cellNext![i] = cellHead![h]!;
      cellHead![h] = i;
    }
    for (let i = 0; i < count; i++) {
      const pxi = px[i]!, pyi = py[i]!, pzi = pz[i]!;
      const gx = Math.floor(pxi / 32), gy = Math.floor(pyi / 32), gz = Math.floor(pzi / 32);
      const srcI = isSource[i] === 1;
      for (let n = 0; n < neighborOffsets.length; n++) {
        const off = neighborOffsets[n]!;
        const h = (Math.imul(gx + off[0], 73856093) ^ Math.imul(gy + off[1], 19349663) ^ Math.imul(gz + off[2], 83492791)) & hashMask;
        for (let j = cellHead![h]!; j !== -1; j = cellNext![j]!) {
          if (j <= i) continue;
          const dx = pxi - px[j]!, dy = pyi - py[j]!, dz = pzi - pz[j]!;
          const distanceSquared = Math.max(1, dx * dx + dy * dy + dz * dz);
          const strength = (srcI || isSource[j] === 1 ? 28 : 12) / distanceSquared;
          fx[i]! += dx * strength; fy[i]! += dy * strength; fz[i]! += dz * strength;
          fx[j]! -= dx * strength; fy[j]! -= dy * strength; fz[j]! -= dz * strength;
        }
      }
    }
    for (let k = 0; k < linkCount; k++) {
      const a = linkATyped[k]!, b = linkBTyped[k]!;
      const dx = px[b]! - px[a]!, dy = py[b]! - py[a]!, dz = pz[b]! - pz[a]!;
      const distance = Math.max(0.01, Math.hypot(dx, dy, dz));
      const spring = (distance - linkLenTyped[k]!) * linkStrTyped[k]! / distance;
      fx[a]! += dx * spring; fy[a]! += dy * spring; fz[a]! += dz * spring;
      fx[b]! -= dx * spring; fy[b]! -= dy * spring; fz[b]! -= dz * spring;
    }
    const cooling = 1 - stepIndex / Math.max(totalSteps + 20, 68);
    for (let i = 0; i < count; i++) {
      if (i === root) continue;
      const dx = fx[i]! - px[i]! * 0.0005, dy = fy[i]! - py[i]! * 0.0005, dz = fz[i]! - pz[i]! * 0.0005;
      const stepScale = Math.min(1, 4 / Math.max(0.01, Math.hypot(dx, dy, dz))) * cooling;
      px[i]! += dx * stepScale; py[i]! += dy * stepScale; pz[i]! += dz * stepScale;
    }
  };

  return {
    count,
    totalSteps,
    get currentStep() {
      return currentStep;
    },
    step(batchSteps = totalSteps): MemoryLayout3DStepResult {
      const target = Math.min(totalSteps, currentStep + Math.max(1, Math.trunc(batchSteps)));
      while (currentStep < target) {
        if (largeGraph) runLargeGraphStep(currentStep);
        else runExactStep(currentStep);
        currentStep++;
      }
      return {
        step: currentStep,
        totalSteps,
        done: currentStep >= totalSteps
      };
    },
    toMap(): Map<string, MemoryPoint3D> {
      const result = new Map<string, MemoryPoint3D>();
      for (let i = 0; i < count; i++) {
        result.set(ordered[i]!.id, { x: px[i]!, y: py[i]!, z: pz[i]! });
      }
      return result;
    },
    packForNodes(targetNodes: MemoryGraphNode[] = nodes): Float32Array {
      const out = new Float32Array(targetNodes.length * 3);
      for (let i = 0; i < targetNodes.length; i++) {
        const idx = index.get(targetNodes[i]!.id);
        const offset = i * 3;
        if (idx !== undefined) {
          out[offset] = px[idx]!;
          out[offset + 1] = py[idx]!;
          out[offset + 2] = pz[idx]!;
        }
      }
      return out;
    }
  };
}

/** Independent XYZ force layout. No coordinates are inherited from the 2D map. */
export function layoutMemoryNetwork3D(nodes: MemoryGraphNode[], edges: MemoryGraphEdge[]): Map<string, MemoryPoint3D> {
  if (!nodes.length) return new Map();
  const stepper = createMemoryLayout3DStepper(nodes, edges);
  stepper.step(stepper.totalSteps);
  return stepper.toMap();
}

/** Projects a 3D memory coordinate into 2D viewport screen coordinates and perspective depth. */
export function projectMemoryPoint(point: MemoryPoint3D, camera: MemoryCamera3D, scale: number, width: number, height: number) {
  const x = point.x * Math.cos(camera.yaw) + point.z * Math.sin(camera.yaw);
  const depth = -point.x * Math.sin(camera.yaw) + point.z * Math.cos(camera.yaw);
  const y = point.y * Math.cos(camera.pitch) - depth * Math.sin(camera.pitch);
  const z = point.y * Math.sin(camera.pitch) + depth * Math.cos(camera.pitch);
  const perspective = 650 / Math.max(120, 650 + z * scale);
  return { x: width / 2 + x * scale * camera.zoom * perspective, y: height / 2 - y * scale * camera.zoom * perspective, z, perspective };
}

/** Normalized covariance determinant: 0 for a plane, 1 for equal spread in XYZ. */
export function memory3DVolumeRatio(points: MemoryPoint3D[] | Float32Array): number {
  if (points instanceof Float32Array) {
    const count = Math.floor(points.length / 3);
    if (count < 4) return 0;
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < points.length; i += 3) {
      cx += points[i]!;
      cy += points[i + 1]!;
      cz += points[i + 2]!;
    }
    cx /= count; cy /= count; cz /= count;
    let xx = 0, yy = 0, zz = 0, xy = 0, xz = 0, yz = 0;
    for (let i = 0; i < points.length; i += 3) {
      const x = points[i]! - cx, y = points[i + 1]! - cy, z = points[i + 2]! - cz;
      xx += x * x; yy += y * y; zz += z * z; xy += x * y; xz += x * z; yz += y * z;
    }
    const determinant = xx * yy * zz + 2 * xy * xz * yz - xx * yz * yz - yy * xz * xz - zz * xy * xy;
    return Math.max(0, determinant / Math.max(1e-12, ((xx + yy + zz) / 3) ** 3));
  }
  if (points.length < 4) return 0;
  const center = points.reduce((sum, p) => ({ x: sum.x + p.x / points.length, y: sum.y + p.y / points.length, z: sum.z + p.z / points.length }), { x: 0, y: 0, z: 0 });
  let xx = 0, yy = 0, zz = 0, xy = 0, xz = 0, yz = 0;
  for (const p of points) {
    const x = p.x - center.x, y = p.y - center.y, z = p.z - center.z;
    xx += x * x; yy += y * y; zz += z * z; xy += x * y; xz += x * z; yz += y * z;
  }
  const determinant = xx * yy * zz + 2 * xy * xz * yz - xx * yz * yz - yy * xz * xz - zz * xy * xy;
  return Math.max(0, determinant / Math.max(1e-12, ((xx + yy + zz) / 3) ** 3));
}

/** Computes the maximum radial extent of packed XYZ coordinates for viewport scaling. */
export function computeMemory3DExtent(packed: Float32Array): number {
  let maxExtent = 60;
  for (let i = 0; i < packed.length; i += 3) {
    const d = Math.hypot(packed[i]!, packed[i + 1]!, packed[i + 2]!);
    if (d > maxExtent) maxExtent = d;
  }
  return maxExtent;
}

export interface Memory3DSpatialIndex {
  pick: (x: number, y: number, focusId?: string | null) => string | null;
}

/**
 * Uniform 2D screen-space grid index for sub-millisecond hover/click picking at 50,000+ nodes.
 */
export function createMemory3DSpatialIndex(
  nodes: MemoryGraphNode[],
  packedPositions: Float32Array,
  baseSizes: Float32Array,
  camera: MemoryCamera3D,
  scale: number,
  width: number,
  height: number
): Memory3DSpatialIndex {
  const count = Math.min(nodes.length, Math.floor(packedPositions.length / 3));
  const cellSize = 8;
  const maxPerCell = 4;
  const cols = Math.max(1, Math.ceil(width / cellSize));
  const rows = Math.max(1, Math.ceil(height / cellSize));
  const totalCells = cols * rows;
  const cellHead = new Int32Array(totalCells).fill(-1);
  const cellCount = new Uint8Array(totalCells);
  const cellMaxZ = new Float32Array(totalCells);
  const cellNext = new Int32Array(count).fill(-1);
  const screenX = new Float32Array(count);
  const screenY = new Float32Array(count);
  const screenZ = new Float32Array(count);
  const baseRadius = new Float32Array(count);

  const cosYaw = Math.cos(camera.yaw), sinYaw = Math.sin(camera.yaw);
  const cosPitch = Math.cos(camera.pitch), sinPitch = Math.sin(camera.pitch);
  const zoomScale = scale * camera.zoom;
  const sqrtZoom = Math.sqrt(camera.zoom);
  const halfW = width / 2, halfH = height / 2;

  for (let i = 0; i < count; i++) {
    const offset = i * 3;
    const px = packedPositions[offset]!, py = packedPositions[offset + 1]!, pz = packedPositions[offset + 2]!;
    const rx = px * cosYaw + pz * sinYaw;
    const depth = -px * sinYaw + pz * cosYaw;
    const ry = py * cosPitch - depth * sinPitch;
    const rz = py * sinPitch + depth * cosPitch;
    const perspective = 650 / Math.max(120, 650 + rz * scale);
    const sx = halfW + rx * zoomScale * perspective;
    const sy = halfH - ry * zoomScale * perspective;
    screenX[i] = sx;
    screenY[i] = sy;
    screenZ[i] = rz;
    const r = Math.max(1, (baseSizes[i] ?? 2.1) * perspective * sqrtZoom);
    baseRadius[i] = r;

    if (sx < -24 || sx > width + 24 || sy < -24 || sy > height + 24) continue;
    const cx = Math.max(0, Math.min(cols - 1, Math.floor(sx / cellSize)));
    const cy = Math.max(0, Math.min(rows - 1, Math.floor(sy / cellSize)));
    const cellIdx = cy * cols + cx;
    const cCount = cellCount[cellIdx]!;
    if (cCount < maxPerCell) {
      cellNext[i] = cellHead[cellIdx]!;
      cellHead[cellIdx] = i;
      cellCount[cellIdx] = cCount + 1;
      if (cCount === 0 || rz > cellMaxZ[cellIdx]!) cellMaxZ[cellIdx] = rz;
    } else if (rz < cellMaxZ[cellIdx]! || r > 4.0) {
      // Replace the farthest node in this cell with the closer/larger node
      let prev = -1;
      let curr = cellHead[cellIdx]!;
      let worstPrev = -1;
      let worstCurr = curr;
      let worstZ = -Infinity;
      while (curr !== -1) {
        const cz = screenZ[curr]!;
        if (cz > worstZ && baseRadius[curr]! <= r) {
          worstZ = cz;
          worstPrev = prev;
          worstCurr = curr;
        }
        prev = curr;
        curr = cellNext[curr]!;
      }
      if (worstCurr !== -1 && (rz < worstZ || r > 4.0)) {
        if (worstPrev === -1) cellHead[cellIdx] = cellNext[worstCurr]!;
        else cellNext[worstPrev] = cellNext[worstCurr]!;
        cellNext[i] = cellHead[cellIdx]!;
        cellHead[cellIdx] = i;
        let newMaxZ = -Infinity;
        for (let nodeIdx = cellHead[cellIdx]!; nodeIdx !== -1; nodeIdx = cellNext[nodeIdx]!) {
          if (screenZ[nodeIdx]! > newMaxZ) newMaxZ = screenZ[nodeIdx]!;
        }
        cellMaxZ[cellIdx] = newMaxZ;
      }
    }
  }

  return {
    pick(x: number, y: number, focusId: string | null = null): string | null {
      if (count === 0 || x < 0 || x > width || y < 0 || y > height) return null;
      const cx = Math.max(0, Math.min(cols - 1, Math.floor(x / cellSize)));
      const cy = Math.max(0, Math.min(rows - 1, Math.floor(y / cellSize)));
      const searchSpan = 3;
      const minGx = Math.max(0, cx - searchSpan);
      const maxGx = Math.min(cols - 1, cx + searchSpan);
      const minGy = Math.max(0, cy - searchSpan);
      const maxGy = Math.min(rows - 1, cy + searchSpan);
      let bestIndex = -1;
      let bestZ = Infinity;

      for (let gy = minGy; gy <= maxGy; gy++) {
        const rowOffset = gy * cols;
        for (let gx = minGx; gx <= maxGx; gx++) {
          for (let i = cellHead[rowOffset + gx]!; i !== -1; i = cellNext[i]!) {
            const r = baseRadius[i]! * (focusId !== null && nodes[i]!.id === focusId ? 1.5 : 1);
            const hitRadius = Math.max(7, r + 3);
            const dx = screenX[i]! - x;
            const dy = screenY[i]! - y;
            if (dx * dx + dy * dy < hitRadius * hitRadius) {
              const z = screenZ[i]!;
              if (z < bestZ || (z === bestZ && i > bestIndex)) {
                bestZ = z;
                bestIndex = i;
              }
            }
          }
        }
      }
      return bestIndex >= 0 ? nodes[bestIndex]!.id : null;
    }
  };
}

export interface Memory3DLabelPlacement {
  id: string;
  label: string;
  x: number;
  y: number;
  width: number;
  isCore: boolean;
}

/**
 * Precomputes bounded priority candidate indices for LOD label selection so per-frame
 * label layout never scans or sorts all 50,000 nodes.
 */
export function buildMemory3DLabelCandidateIndices(
  nodes: MemoryGraphNode[],
  edges: MemoryGraphEdge[],
  maxCandidates = MAX_MEMORY_3D_LABEL_CANDIDATES
): { sourceIndices: Int32Array; zoomIndices: Int32Array; indexById: Map<string, number> } {
  const indexById = new Map<string, number>();
  const degree = new Int32Array(nodes.length);
  for (let i = 0; i < nodes.length; i++) indexById.set(nodes[i]!.id, i);
  for (let i = 0; i < edges.length; i++) {
    const e = edges[i]!;
    const a = indexById.get(e.source);
    const b = indexById.get(e.target);
    if (a !== undefined) degree[a]!++;
    if (b !== undefined) degree[b]!++;
  }

  const sources: number[] = [];
  const memories: number[] = [];
  for (let i = 0; i < nodes.length; i++) {
    if (nodes[i]!.type === "SOURCE" || isMemoryCoreLabel(nodes[i]!.label)) sources.push(i);
    else memories.push(i);
  }

  sources.sort((a, b) =>
    Number(isMemoryCoreLabel(nodes[b]!.label)) - Number(isMemoryCoreLabel(nodes[a]!.label)) ||
    degree[b]! - degree[a]! ||
    compareAscii(nodes[a]!.id, nodes[b]!.id)
  );

  memories.sort((a, b) =>
    degree[b]! - degree[a]! ||
    compareAscii(nodes[a]!.id, nodes[b]!.id)
  );

  return {
    sourceIndices: Int32Array.from(sources.slice(0, maxCandidates)),
    zoomIndices: Int32Array.from(memories.slice(0, maxCandidates)),
    indexById
  };
}

/**
 * Selects non-overlapping LOD labels with a hard per-frame cap on both measured candidates
 * and drawn labels. Preserves core -> SOURCE -> zoomed memory label priority.
 */
export function selectMemory3DLabels(options: {
  nodes: MemoryGraphNode[];
  packedPositions: Float32Array;
  camera: MemoryCamera3D;
  scale: number;
  width: number;
  height: number;
  focusId: string | null;
  neighborIndices?: Iterable<number>;
  candidates: { sourceIndices: Int32Array; zoomIndices: Int32Array; indexById: Map<string, number> };
  measureTextWidth?: (text: string) => number;
  maxLabels?: number;
  maxCandidates?: number;
}): { labels: Memory3DLabelPlacement[]; measuredCount: number } {
  const {
    nodes,
    packedPositions,
    camera,
    scale,
    width,
    height,
    focusId,
    neighborIndices,
    candidates,
    measureTextWidth = (text: string) => text.length * 6.2,
    maxLabels = MAX_MEMORY_3D_LABELS,
    maxCandidates = MAX_MEMORY_3D_LABEL_CANDIDATES
  } = options;

  const order: number[] = [];
  const seen = new Set<number>();
  const pushCandidate = (idx: number | undefined) => {
    if (idx === undefined || idx < 0 || idx >= nodes.length || seen.has(idx) || order.length >= maxCandidates) return;
    seen.add(idx);
    order.push(idx);
  };

  if (focusId) pushCandidate(candidates.indexById.get(focusId));
  for (let i = 0; i < candidates.sourceIndices.length && order.length < maxCandidates; i++) {
    pushCandidate(candidates.sourceIndices[i]!);
  }
  if (camera.zoom >= 1.6) {
    if (neighborIndices) {
      for (const idx of neighborIndices) {
        if (order.length >= maxCandidates) break;
        pushCandidate(idx);
      }
    }
    for (let i = 0; i < candidates.zoomIndices.length && order.length < maxCandidates; i++) {
      pushCandidate(candidates.zoomIndices[i]!);
    }
  }

  const cosYaw = Math.cos(camera.yaw), sinYaw = Math.sin(camera.yaw);
  const cosPitch = Math.cos(camera.pitch), sinPitch = Math.sin(camera.pitch);
  const zoomScale = scale * camera.zoom;
  const halfW = width / 2, halfH = height / 2;

  const projectedCandidates = order.map(idx => {
    const offset = idx * 3;
    const px = packedPositions[offset] ?? 0;
    const py = packedPositions[offset + 1] ?? 0;
    const pz = packedPositions[offset + 2] ?? 0;
    const rx = px * cosYaw + pz * sinYaw;
    const depth = -px * sinYaw + pz * cosYaw;
    const ry = py * cosPitch - depth * sinPitch;
    const rz = py * sinPitch + depth * cosPitch;
    const perspective = 650 / Math.max(120, 650 + rz * scale);
    return {
      idx,
      x: halfW + rx * zoomScale * perspective,
      y: halfH - ry * zoomScale * perspective,
      z: rz
    };
  });

  projectedCandidates.sort((a, b) => {
    const nodeA = nodes[a.idx]!, nodeB = nodes[b.idx]!;
    return (
      Number(nodeB.id === focusId) - Number(nodeA.id === focusId) ||
      Number(isMemoryCoreLabel(nodeB.label)) - Number(isMemoryCoreLabel(nodeA.label)) ||
      Number(nodeB.type === "SOURCE") - Number(nodeA.type === "SOURCE") ||
      a.z - b.z
    );
  });

  const labelBoxes: Array<{ x: number; y: number; w: number }> = [];
  const labels: Memory3DLabelPlacement[] = [];
  let measuredCount = 0;

  for (let i = 0; i < projectedCandidates.length; i++) {
    if (labels.length >= maxLabels || measuredCount >= maxCandidates) break;
    const item = projectedCandidates[i]!;
    const node = nodes[item.idx]!;
    if (node.type !== "SOURCE" && node.id !== focusId && camera.zoom < 1.6) continue;
    const x = item.x + 9;
    const y = item.y;
    if (x < 0 || x >= width || y < 14 || y > height - 5) continue;
    const text = node.label.length > 48 ? `${node.label.slice(0, 45)}…` : node.label;
    measuredCount++;
    const w = measureTextWidth(text);
    if (
      x + w > width ||
      labelBoxes.some(box => Math.abs(box.y - y) < 16 && x < box.x + box.w + 5 && x + w + 5 > box.x)
    ) {
      continue;
    }
    labelBoxes.push({ x, y, w });
    labels.push({
      id: node.id,
      label: text,
      x,
      y,
      width: w,
      isCore: isMemoryCoreLabel(node.label)
    });
  }

  return { labels, measuredCount };
}
