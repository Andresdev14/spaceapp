import { useEffect, useMemo, useRef } from "react";
import type { MemoryGraphNode, MemoryGraphEdge } from "@space/contracts";
import type { MemoryAtlasPosition } from "./memory-atlas.js";
import { isMemoryCoreLabel } from "./memory-graph-visuals.js";
import {
  DEFAULT_MEMORY_CAMERA,
  MAX_MEMORY_3D_LABELS,
  buildMemory3DLabelCandidateIndices,
  computeMemory3DExtent,
  createMemory3DSpatialIndex,
  memory3DVolumeRatio,
  projectMemoryPoint,
  selectMemory3DLabels,
  type Memory3DSpatialIndex,
  type MemoryCamera3D,
  type MemoryPoint3D
} from "./memory-3d.js";

const FALLBACK_NODE_CAP = 2000;
const FALLBACK_EDGE_CAP = 6000;

const EDGE_VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location = 0) in vec3 a_position;
layout(location = 1) in float a_kind;

uniform float u_cosYaw;
uniform float u_sinYaw;
uniform float u_cosPitch;
uniform float u_sinPitch;
uniform float u_scale;
uniform float u_zoom;
uniform vec2 u_viewport;
uniform float u_mode;

out vec4 v_color;

void main() {
  float rx = a_position.x * u_cosYaw + a_position.z * u_sinYaw;
  float depth = -a_position.x * u_sinYaw + a_position.z * u_cosYaw;
  float ry = a_position.y * u_cosPitch - depth * u_sinPitch;
  float rz = a_position.y * u_sinPitch + depth * u_cosPitch;
  float perspective = 650.0 / max(120.0, 650.0 + rz * u_scale);
  float sx = u_viewport.x * 0.5 + rx * u_scale * u_zoom * perspective;
  float sy = u_viewport.y * 0.5 - ry * u_scale * u_zoom * perspective;
  vec2 clip = (vec2(sx, sy) / u_viewport) * 2.0 - 1.0;
  clip.y = -clip.y;
  float ndcZ = clamp((rz * u_scale) / 1800.0 + 0.015, -0.98, 0.98);
  if (u_mode > 1.5) {
    ndcZ -= 0.02;
    v_color = vec4(0.596, 0.855, 0.980, 0.92);
  } else if (u_mode > 0.5) {
    v_color = vec4(0.078, 0.125, 0.169, 0.55);
  } else if (a_kind > 0.5) {
    v_color = vec4(0.220, 0.314, 0.408, 0.75);
  } else {
    v_color = vec4(0.125, 0.200, 0.263, 0.75);
  }
  gl_Position = vec4(clip, ndcZ, 1.0);
}
`;

const EDGE_FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec4 v_color;
out vec4 outColor;
void main() {
  outColor = v_color;
}
`;

const NODE_VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location = 0) in vec2 a_quad;
layout(location = 1) in vec3 a_nodePos;
layout(location = 2) in vec3 a_nodeColor;
layout(location = 3) in vec2 a_sizeAndRoot;
layout(location = 4) in float a_focusState;

uniform float u_cosYaw;
uniform float u_sinYaw;
uniform float u_cosPitch;
uniform float u_sinPitch;
uniform float u_scale;
uniform float u_zoom;
uniform float u_sqrtZoom;
uniform vec2 u_viewport;
uniform float u_hasFocus;
uniform float u_glowPass;

out vec2 v_local;
out vec3 v_color;
out float v_alpha;

void main() {
  bool highlighted = a_focusState > 1.5;
  bool isNeighbor = a_focusState > 0.5;
  bool isRoot = a_sizeAndRoot.y > 0.5;
  if (u_glowPass > 0.5 && !isRoot && !highlighted) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    v_local = vec2(0.0);
    v_color = vec3(0.0);
    v_alpha = 0.0;
    return;
  }
  float rx = a_nodePos.x * u_cosYaw + a_nodePos.z * u_sinYaw;
  float depth = -a_nodePos.x * u_sinYaw + a_nodePos.z * u_cosYaw;
  float ry = a_nodePos.y * u_cosPitch - depth * u_sinPitch;
  float rz = a_nodePos.y * u_sinPitch + depth * u_cosPitch;
  float perspective = 650.0 / max(120.0, 650.0 + rz * u_scale);
  float sx = u_viewport.x * 0.5 + rx * u_scale * u_zoom * perspective;
  float sy = u_viewport.y * 0.5 - ry * u_scale * u_zoom * perspective;

  float baseRadius = max(1.0, a_sizeAndRoot.x * perspective * u_sqrtZoom) * (highlighted ? 1.5 : 1.0);
  float quadRadius = u_glowPass > 0.5 ? baseRadius + 9.0 : baseRadius + 0.75;
  vec2 pixel = vec2(sx, sy) + a_quad * quadRadius;
  vec2 clip = (pixel / u_viewport) * 2.0 - 1.0;
  clip.y = -clip.y;
  float ndcZ = clamp((rz * u_scale) / 1800.0, -0.96, 0.96);
  if (highlighted) ndcZ -= 0.025;
  else if (isRoot) ndcZ -= 0.01;

  gl_Position = vec4(clip, ndcZ, 1.0);
  v_local = a_quad;
  v_color = highlighted ? vec3(1.0, 1.0, 1.0) : a_nodeColor;
  float depthAlpha = clamp(perspective, 0.4, 1.0);
  v_alpha = (u_hasFocus > 0.5 && !highlighted && !isNeighbor) ? 0.2 : depthAlpha;
}
`;

const NODE_FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec2 v_local;
in vec3 v_color;
in float v_alpha;
uniform float u_glowPass;
out vec4 outColor;

void main() {
  float r = length(v_local);
  if (r > 1.0) discard;
  if (u_glowPass > 0.5) {
    float glow = exp(-3.5 * r * r) * 0.36;
    if (glow < 0.015) discard;
    outColor = vec4(v_color, glow * v_alpha);
    return;
  }
  float aa = 1.0 - smoothstep(0.82, 1.0, r);
  outColor = vec4(v_color, v_alpha * aa);
}
`;

function parseHexRgb(hex: string): [number, number, number] {
  const clean = hex.startsWith("#") ? hex.slice(1) : hex;
  if (clean.length === 6) {
    const num = Number.parseInt(clean, 16);
    if (!Number.isNaN(num)) return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
  }
  return [115, 149, 173];
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function createProgram(gl: WebGL2RenderingContext, vsSource: string, fsSource: string): WebGLProgram | null {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSource);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
  if (!vs || !fs) {
    if (vs) gl.deleteShader(vs);
    if (fs) gl.deleteShader(fs);
    return null;
  }
  const program = gl.createProgram();
  if (!program) return null;
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.deleteProgram(program);
    return null;
  }
  return program;
}

interface WebGLResources {
  gl: WebGL2RenderingContext;
  edgeProgram: WebGLProgram;
  nodeProgram: WebGLProgram;
  edgeVao: WebGLVertexArrayObject;
  activeEdgeVao: WebGLVertexArrayObject;
  nodeVao: WebGLVertexArrayObject;
  quadBuffer: WebGLBuffer;
  edgePosBuffer: WebGLBuffer;
  edgeKindBuffer: WebGLBuffer;
  activeEdgePosBuffer: WebGLBuffer;
  nodePosBuffer: WebGLBuffer;
  nodeColorBuffer: WebGLBuffer;
  nodeSizeBuffer: WebGLBuffer;
  nodeFocusBuffer: WebGLBuffer;
  edgeUniforms: Record<string, WebGLUniformLocation | null>;
  nodeUniforms: Record<string, WebGLUniformLocation | null>;
}

function initWebGLResources(gl: WebGL2RenderingContext): WebGLResources | null {
  const edgeProgram = createProgram(gl, EDGE_VERTEX_SHADER, EDGE_FRAGMENT_SHADER);
  const nodeProgram = createProgram(gl, NODE_VERTEX_SHADER, NODE_FRAGMENT_SHADER);
  if (!edgeProgram || !nodeProgram) return null;

  const edgeVao = gl.createVertexArray();
  const activeEdgeVao = gl.createVertexArray();
  const nodeVao = gl.createVertexArray();
  const quadBuffer = gl.createBuffer();
  const edgePosBuffer = gl.createBuffer();
  const edgeKindBuffer = gl.createBuffer();
  const activeEdgePosBuffer = gl.createBuffer();
  const nodePosBuffer = gl.createBuffer();
  const nodeColorBuffer = gl.createBuffer();
  const nodeSizeBuffer = gl.createBuffer();
  const nodeFocusBuffer = gl.createBuffer();

  if (
    !edgeVao || !activeEdgeVao || !nodeVao || !quadBuffer ||
    !edgePosBuffer || !edgeKindBuffer || !activeEdgePosBuffer ||
    !nodePosBuffer || !nodeColorBuffer || !nodeSizeBuffer || !nodeFocusBuffer
  ) {
    return null;
  }

  gl.bindVertexArray(edgeVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, edgePosBuffer);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, edgeKindBuffer);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0);

  gl.bindVertexArray(activeEdgeVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, activeEdgePosBuffer);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.vertexAttrib1f(1, 0);

  gl.bindVertexArray(nodeVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, nodePosBuffer);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(1, 1);

  gl.bindBuffer(gl.ARRAY_BUFFER, nodeColorBuffer);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 3, gl.UNSIGNED_BYTE, true, 0, 0);
  gl.vertexAttribDivisor(2, 1);

  gl.bindBuffer(gl.ARRAY_BUFFER, nodeSizeBuffer);
  gl.enableVertexAttribArray(3);
  gl.vertexAttribPointer(3, 2, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(3, 1);

  gl.bindBuffer(gl.ARRAY_BUFFER, nodeFocusBuffer);
  gl.enableVertexAttribArray(4);
  gl.vertexAttribPointer(4, 1, gl.UNSIGNED_BYTE, false, 0, 0);
  gl.vertexAttribDivisor(4, 1);

  gl.bindVertexArray(null);

  const edgeUniformNames = ["u_cosYaw", "u_sinYaw", "u_cosPitch", "u_sinPitch", "u_scale", "u_zoom", "u_viewport", "u_mode"];
  const nodeUniformNames = ["u_cosYaw", "u_sinYaw", "u_cosPitch", "u_sinPitch", "u_scale", "u_zoom", "u_sqrtZoom", "u_viewport", "u_hasFocus", "u_glowPass"];
  const edgeUniforms: Record<string, WebGLUniformLocation | null> = {};
  const nodeUniforms: Record<string, WebGLUniformLocation | null> = {};
  for (const name of edgeUniformNames) edgeUniforms[name] = gl.getUniformLocation(edgeProgram, name);
  for (const name of nodeUniformNames) nodeUniforms[name] = gl.getUniformLocation(nodeProgram, name);

  return {
    gl,
    edgeProgram,
    nodeProgram,
    edgeVao,
    activeEdgeVao,
    nodeVao,
    quadBuffer,
    edgePosBuffer,
    edgeKindBuffer,
    activeEdgePosBuffer,
    nodePosBuffer,
    nodeColorBuffer,
    nodeSizeBuffer,
    nodeFocusBuffer,
    edgeUniforms,
    nodeUniforms
  };
}

export function MemoryGraph3D({
  nodes,
  edges,
  positions,
  coordinates,
  packedCoordinates,
  showTitles,
  selectedNodeId,
  onSelectNode,
  loading,
  resetToken,
  onRendererFallbackChange
}: {
  nodes: MemoryGraphNode[];
  edges: MemoryGraphEdge[];
  positions: Map<string, MemoryAtlasPosition>;
  coordinates: Map<string, MemoryPoint3D>;
  packedCoordinates?: Float32Array | null;
  showTitles: boolean;
  selectedNodeId: string | null;
  onSelectNode: (id: string) => void;
  loading: boolean;
  resetToken: number;
  onRendererFallbackChange?: (fallback: boolean) => void;
}) {
  const packed = useMemo(() => {
    if (packedCoordinates && packedCoordinates.length === nodes.length * 3) {
      return packedCoordinates;
    }
    const array = new Float32Array(nodes.length * 3);
    for (let i = 0; i < nodes.length; i++) {
      const p = coordinates.get(nodes[i]!.id);
      if (p) {
        const offset = i * 3;
        array[offset] = p.x;
        array[offset + 1] = p.y;
        array[offset + 2] = p.z;
      }
    }
    return array;
  }, [packedCoordinates, coordinates, nodes]);

  const volumeRatio = useMemo(() => memory3DVolumeRatio(packed), [packed]);
  const extent = useMemo(() => computeMemory3DExtent(packed), [packed]);

  const staticGraphData = useMemo(() => {
    const count = nodes.length;
    const indexById = new Map<string, number>();
    const colors = new Uint8Array(count * 3);
    const sizeAndRoot = new Float32Array(count * 2);
    const baseSizes = new Float32Array(count);
    const neighborsByNode: number[][] = Array.from({ length: count }, () => []);
    const incidentEdgesByNode: number[][] = Array.from({ length: count }, () => []);

    for (let i = 0; i < count; i++) {
      const node = nodes[i]!;
      indexById.set(node.id, i);
      const colorHex = positions.get(node.id)?.color ?? "#7395ad";
      const [r, g, b] = parseHexRgb(colorHex);
      const cOff = i * 3;
      colors[cOff] = r;
      colors[cOff + 1] = g;
      colors[cOff + 2] = b;

      const root = node.type === "SOURCE" && isMemoryCoreLabel(node.label);
      const baseSize = root ? 10 : node.type === "SOURCE" ? 4.6 : 2.1;
      baseSizes[i] = baseSize;
      sizeAndRoot[i * 2] = baseSize;
      sizeAndRoot[i * 2 + 1] = root ? 1 : 0;
    }

    const validEdgeEndpoints: number[] = [];
    const validEdgeKinds: number[] = [];
    for (let i = 0; i < edges.length; i++) {
      const edge = edges[i]!;
      const a = indexById.get(edge.source);
      const b = indexById.get(edge.target);
      if (a === undefined || b === undefined) continue;
      const edgeIdx = validEdgeEndpoints.length / 2;
      validEdgeEndpoints.push(a, b);
      const kind = edge.type === "DERIVED_FROM" ? 1 : 0;
      validEdgeKinds.push(kind, kind);
      neighborsByNode[a]!.push(b);
      neighborsByNode[b]!.push(a);
      incidentEdgesByNode[a]!.push(edgeIdx);
      incidentEdgesByNode[b]!.push(edgeIdx);
    }

    const totalValidEdges = validEdgeEndpoints.length / 2;
    const maxBackgroundEdges = 18_000;
    let renderEdgeEndpoints: Int32Array;
    let renderEdgeKinds: Float32Array;
    if (totalValidEdges <= maxBackgroundEdges) {
      renderEdgeEndpoints = Int32Array.from(validEdgeEndpoints);
      renderEdgeKinds = Float32Array.from(validEdgeKinds);
    } else {
      const sampledEndpoints: number[] = [];
      const sampledKinds: number[] = [];
      const stride = Math.max(1, Math.ceil(totalValidEdges / maxBackgroundEdges));
      for (let i = 0; i < totalValidEdges; i++) {
        const kind = validEdgeKinds[i * 2]!;
        if (kind > 0.5 || i % stride === 0) {
          sampledEndpoints.push(validEdgeEndpoints[i * 2]!, validEdgeEndpoints[i * 2 + 1]!);
          sampledKinds.push(kind, kind);
        }
      }
      renderEdgeEndpoints = Int32Array.from(sampledEndpoints);
      renderEdgeKinds = Float32Array.from(sampledKinds);
    }

    const labelCandidates = buildMemory3DLabelCandidateIndices(nodes, edges);

    return {
      indexById,
      colors,
      sizeAndRoot,
      baseSizes,
      neighborsByNode,
      incidentEdgesByNode,
      totalEdgeCount: totalValidEdges,
      allEdgeEndpoints: Int32Array.from(validEdgeEndpoints),
      edgeEndpoints: renderEdgeEndpoints,
      edgeKinds: renderEdgeKinds,
      labelCandidates
    };
  }, [nodes, edges, positions]);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const cameraRef = useRef<MemoryCamera3D>({ ...DEFAULT_MEMORY_CAMERA });
  const redrawRef = useRef<() => void>(() => {});
  const uploadGraphRef = useRef<() => void>(() => {});
  const uploadPositionsRef = useRef<() => void>(() => {});

  const current = useRef({
    nodes,
    edges,
    positions,
    packed,
    extent,
    staticGraphData,
    showTitles,
    selectedNodeId,
    onSelectNode,
    onRendererFallbackChange
  });
  current.current = {
    nodes,
    edges,
    positions,
    packed,
    extent,
    staticGraphData,
    showTitles,
    selectedNodeId,
    onSelectNode,
    onRendererFallbackChange
  };

  useEffect(() => {
    cameraRef.current = { ...DEFAULT_MEMORY_CAMERA };
    redrawRef.current();
  }, [resetToken]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;
    if (!canvas) return;

    let glResources: WebGLResources | null = null;
    let contextLost = false;
    let fallback2DContext: CanvasRenderingContext2D | null = null;
    const overlayContext = overlayCanvas?.getContext("2d") ?? null;

    const tryInitWebGL2 = () => {
      if (contextLost) return false;
      try {
        const forceFallback = typeof window !== "undefined" && new URLSearchParams(window.location.search).get("webgl") === "0";
        if (forceFallback) return false;
        const gl = canvas.getContext("webgl2", {
          alpha: true,
          antialias: false,
          depth: true,
          powerPreference: "high-performance",
          preserveDrawingBuffer: false
        });
        if (!gl) return false;
        const resources = initWebGLResources(gl);
        if (!resources) return false;
        glResources = resources;
        return true;
      } catch {
        return false;
      }
    };

    if (!tryInitWebGL2()) {
      try {
        fallback2DContext = canvas.getContext("2d");
      } catch {
        fallback2DContext = null;
      }
      current.current.onRendererFallbackChange?.(true);
    } else {
      current.current.onRendererFallbackChange?.(false);
    }

    let frame = 0;
    let width = 1;
    let height = 1;
    let dpr = 1;
    let hover: string | null = null;
    const pointers = new Map<number, { x: number; y: number }>();
    let dragged = false;
    let downX = 0;
    let downY = 0;

    let uploadedStaticRef: typeof staticGraphData | null = null;
    let uploadedPackedRef: Float32Array | null = null;
    let edgePositionsBufferData = new Float32Array(0);
    let focusStateData = new Uint8Array(0);
    let lastUploadedFocus: string | null | undefined = undefined;
    let activeEdgeVertexCount = 0;

    let cachedSpatialIndex: Memory3DSpatialIndex | null = null;
    let cachedSpatialKey = "";
    const frameTimes: number[] = [];
    let lastPickMs = 0;

    const syncStaticBuffers = () => {
      if (!glResources) return;
      const { gl } = glResources;
      const { staticGraphData: data } = current.current;
      if (uploadedStaticRef === data) return;
      uploadedStaticRef = data;
      uploadedPackedRef = null;
      lastUploadedFocus = undefined;

      gl.bindBuffer(gl.ARRAY_BUFFER, glResources.nodeColorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, data.colors, gl.STATIC_DRAW);

      gl.bindBuffer(gl.ARRAY_BUFFER, glResources.nodeSizeBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, data.sizeAndRoot, gl.STATIC_DRAW);

      focusStateData = new Uint8Array(current.current.nodes.length);
      gl.bindBuffer(gl.ARRAY_BUFFER, glResources.nodeFocusBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, focusStateData, gl.DYNAMIC_DRAW);

      gl.bindBuffer(gl.ARRAY_BUFFER, glResources.edgeKindBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, data.edgeKinds, gl.STATIC_DRAW);

      const edgeCount = data.edgeEndpoints.length / 2;
      edgePositionsBufferData = new Float32Array(edgeCount * 6);
      gl.bindBuffer(gl.ARRAY_BUFFER, glResources.edgePosBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, edgePositionsBufferData.byteLength, gl.DYNAMIC_DRAW);
    };

    const syncPositionBuffers = () => {
      if (!glResources) return;
      const { gl } = glResources;
      const { packed: pos, staticGraphData: data } = current.current;
      if (uploadedPackedRef === pos) return;
      uploadedPackedRef = pos;
      lastUploadedFocus = undefined;
      cachedSpatialIndex = null;

      gl.bindBuffer(gl.ARRAY_BUFFER, glResources.nodePosBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);

      const endpoints = data.edgeEndpoints;
      const edgeCount = endpoints.length / 2;
      if (edgePositionsBufferData.length !== edgeCount * 6) {
        edgePositionsBufferData = new Float32Array(edgeCount * 6);
      }
      for (let i = 0; i < edgeCount; i++) {
        const aOff = endpoints[i * 2]! * 3;
        const bOff = endpoints[i * 2 + 1]! * 3;
        const outOff = i * 6;
        edgePositionsBufferData[outOff] = pos[aOff]!;
        edgePositionsBufferData[outOff + 1] = pos[aOff + 1]!;
        edgePositionsBufferData[outOff + 2] = pos[aOff + 2]!;
        edgePositionsBufferData[outOff + 3] = pos[bOff]!;
        edgePositionsBufferData[outOff + 4] = pos[bOff + 1]!;
        edgePositionsBufferData[outOff + 5] = pos[bOff + 2]!;
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, glResources.edgePosBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, edgePositionsBufferData, gl.DYNAMIC_DRAW);
    };

    const syncFocusBuffers = (focusId: string | null) => {
      if (!glResources) return;
      if (lastUploadedFocus === focusId) return;
      lastUploadedFocus = focusId;
      const { gl } = glResources;
      const { staticGraphData: data, packed: pos, nodes: nodeList } = current.current;

      if (focusStateData.length !== nodeList.length) {
        focusStateData = new Uint8Array(nodeList.length);
      } else {
        focusStateData.fill(0);
      }
      activeEdgeVertexCount = 0;

      if (focusId !== null) {
        const focusIdx = data.indexById.get(focusId);
        if (focusIdx !== undefined) {
          const neighbors = data.neighborsByNode[focusIdx] ?? [];
          for (let i = 0; i < neighbors.length; i++) {
            focusStateData[neighbors[i]!] = 1;
          }
          focusStateData[focusIdx] = 2;

          const incident = data.incidentEdgesByNode[focusIdx] ?? [];
          const activeCoords = new Float32Array(incident.length * 6);
          for (let i = 0; i < incident.length; i++) {
            const edgeIdx = incident[i]!;
            const aOff = data.allEdgeEndpoints[edgeIdx * 2]! * 3;
            const bOff = data.allEdgeEndpoints[edgeIdx * 2 + 1]! * 3;
            const outOff = i * 6;
            activeCoords[outOff] = pos[aOff]!;
            activeCoords[outOff + 1] = pos[aOff + 1]!;
            activeCoords[outOff + 2] = pos[aOff + 2]!;
            activeCoords[outOff + 3] = pos[bOff]!;
            activeCoords[outOff + 4] = pos[bOff + 1]!;
            activeCoords[outOff + 5] = pos[bOff + 2]!;
          }
          gl.bindBuffer(gl.ARRAY_BUFFER, glResources.activeEdgePosBuffer);
          gl.bufferData(gl.ARRAY_BUFFER, activeCoords, gl.DYNAMIC_DRAW);
          activeEdgeVertexCount = incident.length * 2;
        }
      }

      gl.bindBuffer(gl.ARRAY_BUFFER, glResources.nodeFocusBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, focusStateData, gl.DYNAMIC_DRAW);
    };

    uploadGraphRef.current = syncStaticBuffers;
    uploadPositionsRef.current = syncPositionBuffers;

    const getSpatialIndex = (): Memory3DSpatialIndex => {
      const { nodes: nodeList, packed: pos, extent: ext, staticGraphData: data } = current.current;
      const scale = Math.min(width, height) * 0.4 / ext;
      const cam = cameraRef.current;
      const key = `${nodeList.length}:${cam.yaw.toFixed(4)}:${cam.pitch.toFixed(4)}:${cam.zoom.toFixed(4)}:${width}:${height}:${scale.toFixed(4)}`;
      if (!cachedSpatialIndex || cachedSpatialKey !== key) {
        cachedSpatialKey = key;
        const activeNodes = glResources && !contextLost ? nodeList : nodeList.slice(0, FALLBACK_NODE_CAP);
        cachedSpatialIndex = createMemory3DSpatialIndex(activeNodes, pos, data.baseSizes, cam, scale, width, height);
      }
      return cachedSpatialIndex;
    };

    const drawLabelsOverlay = (
      ctx: CanvasRenderingContext2D,
      scale: number,
      focusId: string | null,
      clearFirst: boolean
    ): number => {
      if (clearFirst) ctx.clearRect(0, 0, width, height);
      if (!current.current.showTitles) return 0;
      const { nodes: nodeList, packed: pos, staticGraphData: data } = current.current;
      const focusIdx = focusId ? data.indexById.get(focusId) : undefined;
      const neighborIndices = focusIdx !== undefined ? data.neighborsByNode[focusIdx] : undefined;
      ctx.font = "11px system-ui, sans-serif";
      const { labels, measuredCount } = selectMemory3DLabels({
        nodes: nodeList,
        packedPositions: pos,
        camera: cameraRef.current,
        scale,
        width,
        height,
        focusId,
        neighborIndices,
        candidates: data.labelCandidates,
        measureTextWidth: text => ctx.measureText(text).width,
        maxLabels: MAX_MEMORY_3D_LABELS
      });
      for (let i = 0; i < labels.length; i++) {
        const item = labels[i]!;
        ctx.fillStyle = "#09111bea";
        ctx.fillRect(item.x - 3, item.y - 11, item.width + 6, 16);
        ctx.fillStyle = item.isCore ? "#fff0b0" : "#c6dfed";
        ctx.fillText(item.label, item.x, item.y + 1);
      }
      return measuredCount;
    };

    const renderFallback2D = (ctx: CanvasRenderingContext2D, scale: number, focusId: string | null) => {
      const { nodes: allNodes, edges: allEdges, positions: posMap, packed: pos, staticGraphData: data } = current.current;
      const cappedNodes = allNodes.length > FALLBACK_NODE_CAP ? allNodes.slice(0, FALLBACK_NODE_CAP) : allNodes;
      const cappedNodeIds = new Set(cappedNodes.map(n => n.id));
      const cappedEdges = allEdges.filter(e => cappedNodeIds.has(e.source) && cappedNodeIds.has(e.target)).slice(0, FALLBACK_EDGE_CAP);

      ctx.clearRect(0, 0, width, height);
      const projected = new Map<string, { x: number; y: number; z: number; perspective: number }>();
      for (let i = 0; i < cappedNodes.length; i++) {
        const node = cappedNodes[i]!;
        const idx = data.indexById.get(node.id) ?? i;
        const offset = idx * 3;
        projected.set(
          node.id,
          projectMemoryPoint(
            { x: pos[offset] ?? 0, y: pos[offset + 1] ?? 0, z: pos[offset + 2] ?? 0 },
            cameraRef.current,
            scale,
            width,
            height
          )
        );
      }
      const neighbors = new Set<string>();
      if (focusId) {
        for (const edge of cappedEdges) {
          if (edge.source === focusId) neighbors.add(edge.target);
          if (edge.target === focusId) neighbors.add(edge.source);
        }
      }
      ctx.lineWidth = 0.55;
      for (const edge of cappedEdges) {
        const a = projected.get(edge.source), b = projected.get(edge.target);
        if (!a || !b) continue;
        const incident = edge.source === focusId || edge.target === focusId;
        ctx.strokeStyle = incident ? "#98dafa" : focusId ? "#14202b" : edge.type === "DERIVED_FROM" ? "#385068" : "#203343";
        ctx.lineWidth = incident ? 1.15 : 0.55;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
      const ordered = [...cappedNodes].sort((a, b) => projected.get(b.id)!.z - projected.get(a.id)!.z);
      for (const node of ordered) {
        const p = projected.get(node.id)!;
        const highlighted = node.id === focusId;
        const root = node.type === "SOURCE" && isMemoryCoreLabel(node.label);
        const radius = Math.max(1, (root ? 10 : node.type === "SOURCE" ? 4.6 : 2.1) * p.perspective * Math.sqrt(cameraRef.current.zoom)) * (highlighted ? 1.5 : 1);
        ctx.globalAlpha = focusId && !highlighted && !neighbors.has(node.id) ? 0.2 : Math.max(0.4, Math.min(1, p.perspective));
        ctx.fillStyle = highlighted ? "#ffffff" : posMap.get(node.id)?.color ?? "#7395ad";
        if (root || highlighted) {
          ctx.shadowColor = ctx.fillStyle;
          ctx.shadowBlur = 13;
        }
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }
      ctx.globalAlpha = 1;
      drawLabelsOverlay(ctx, scale, focusId, false);
    };

    const render = () => {
      frame = 0;
      const t0 = performance.now();
      const { nodes: nodeList, extent: ext, selectedNodeId: selectedId, staticGraphData: data } = current.current;
      const scale = Math.min(width, height) * 0.4 / ext;
      const focus = hover ?? selectedId;
      let measuredLabels = 0;

      if (glResources && !contextLost) {
        syncStaticBuffers();
        syncPositionBuffers();
        syncFocusBuffers(focus);

        const { gl, edgeProgram, nodeProgram, edgeVao, activeEdgeVao, nodeVao, edgeUniforms, nodeUniforms } = glResources;
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        const cosYaw = Math.cos(cameraRef.current.yaw);
        const sinYaw = Math.sin(cameraRef.current.yaw);
        const cosPitch = Math.cos(cameraRef.current.pitch);
        const sinPitch = Math.sin(cameraRef.current.pitch);
        const zoom = cameraRef.current.zoom;
        const edgeVertexCount = data.edgeEndpoints.length;

        if (edgeVertexCount > 0) {
          gl.disable(gl.DEPTH_TEST);
          gl.depthMask(false);
          gl.useProgram(edgeProgram);
          gl.uniform1f(edgeUniforms.u_cosYaw!, cosYaw);
          gl.uniform1f(edgeUniforms.u_sinYaw!, sinYaw);
          gl.uniform1f(edgeUniforms.u_cosPitch!, cosPitch);
          gl.uniform1f(edgeUniforms.u_sinPitch!, sinPitch);
          gl.uniform1f(edgeUniforms.u_scale!, scale);
          gl.uniform1f(edgeUniforms.u_zoom!, zoom);
          gl.uniform2f(edgeUniforms.u_viewport!, width, height);
          gl.uniform1f(edgeUniforms.u_mode!, focus ? 1 : 0);

          gl.bindVertexArray(edgeVao);
          gl.drawArrays(gl.LINES, 0, edgeVertexCount);

          if (focus && activeEdgeVertexCount > 0) {
            gl.uniform1f(edgeUniforms.u_mode!, 2);
            gl.bindVertexArray(activeEdgeVao);
            gl.drawArrays(gl.LINES, 0, activeEdgeVertexCount);
          }
        }

        if (nodeList.length > 0) {
          gl.useProgram(nodeProgram);
          gl.uniform1f(nodeUniforms.u_cosYaw!, cosYaw);
          gl.uniform1f(nodeUniforms.u_sinYaw!, sinYaw);
          gl.uniform1f(nodeUniforms.u_cosPitch!, cosPitch);
          gl.uniform1f(nodeUniforms.u_sinPitch!, sinPitch);
          gl.uniform1f(nodeUniforms.u_scale!, scale);
          gl.uniform1f(nodeUniforms.u_zoom!, zoom);
          gl.uniform1f(nodeUniforms.u_sqrtZoom!, Math.sqrt(zoom));
          gl.uniform2f(nodeUniforms.u_viewport!, width, height);
          gl.uniform1f(nodeUniforms.u_hasFocus!, focus ? 1 : 0);

          gl.bindVertexArray(nodeVao);
          if (nodeList.length <= 2_000) {
            // Glow pass for root and focused nodes
            gl.depthMask(false);
            gl.uniform1f(nodeUniforms.u_glowPass!, 1);
            gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, nodeList.length);
          }

          // Solid circle pass with hardware depth test
          gl.enable(gl.DEPTH_TEST);
          gl.depthFunc(gl.LEQUAL);
          gl.depthMask(true);
          gl.uniform1f(nodeUniforms.u_glowPass!, 0);
          gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, nodeList.length);
          gl.bindVertexArray(null);
        }

        if (overlayContext) {
          measuredLabels = drawLabelsOverlay(overlayContext, scale, focus, true);
        }
        canvas.dataset.renderer = "webgl2";
      } else {
        const target2D = fallback2DContext ?? overlayContext;
        if (target2D) {
          renderFallback2D(target2D, scale, focus);
        }
        canvas.dataset.renderer = contextLost ? "canvas2d-fallback" : "canvas2d";
      }

      const elapsed = performance.now() - t0;
      frameTimes.push(elapsed);
      if (frameTimes.length > 180) frameTimes.shift();
      canvas.dataset.rotation = `${cameraRef.current.yaw.toFixed(3)},${cameraRef.current.pitch.toFixed(3)}`;
      canvas.dataset.lastFrameMs = elapsed.toFixed(2);
      if (typeof window !== "undefined") {
        const sorted = [...frameTimes].sort((a, b) => a - b);
        const p50 = sorted[Math.floor(sorted.length * 0.5)] ?? elapsed;
        const p95 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))] ?? elapsed;
        (window as unknown as Record<string, unknown>).__spaceMemory3DStats = {
          renderer: canvas.dataset.renderer,
          nodeCount: glResources && !contextLost ? nodeList.length : Math.min(nodeList.length, FALLBACK_NODE_CAP),
          edgeCount: glResources && !contextLost ? data.totalEdgeCount : Math.min(data.totalEdgeCount, FALLBACK_EDGE_CAP),
          lastFrameMs: Number(elapsed.toFixed(3)),
          p50FrameMs: Number(p50.toFixed(3)),
          p95FrameMs: Number(p95.toFixed(3)),
          sampleCount: frameTimes.length,
          lastPickMs: Number(lastPickMs.toFixed(3)),
          measuredLabels
        };
      }
    };

    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(render);
    };
    redrawRef.current = schedule;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      width = Math.max(1, rect.width);
      height = Math.max(1, rect.height);
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      if (overlayCanvas) {
        overlayCanvas.width = Math.round(width * dpr);
        overlayCanvas.height = Math.round(height * dpr);
      }
      if (fallback2DContext) fallback2DContext.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (overlayContext) overlayContext.setTransform(dpr, 0, 0, dpr, 0, 0);
      cachedSpatialIndex = null;
      schedule();
    };

    const hit = (x: number, y: number) => {
      const t0 = performance.now();
      const focus = hover ?? current.current.selectedNodeId;
      const result = getSpatialIndex().pick(x, y, focus);
      lastPickMs = performance.now() - t0;
      return result;
    };

    const local = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    };

    const down = (event: PointerEvent) => {
      canvas.focus();
      canvas.setPointerCapture(event.pointerId);
      const p = local(event);
      pointers.set(event.pointerId, p);
      downX = p.x;
      downY = p.y;
      dragged = false;
      hover = null;
    };

    const move = (event: PointerEvent) => {
      const p = local(event);
      const previous = pointers.get(event.pointerId);
      if (previous) {
        if (Math.hypot(p.x - downX, p.y - downY) > 4) dragged = true;
        if (pointers.size === 2) {
          const other = [...pointers].find(([id]) => id !== event.pointerId)![1];
          const before = Math.hypot(previous.x - other.x, previous.y - other.y);
          const after = Math.hypot(p.x - other.x, p.y - other.y);
          if (before > 1) cameraRef.current.zoom = Math.max(0.35, Math.min(5, cameraRef.current.zoom * after / before));
        } else {
          cameraRef.current.yaw += (p.x - previous.x) * 0.009;
          cameraRef.current.pitch += (p.y - previous.y) * 0.009;
        }
        pointers.set(event.pointerId, p);
        schedule();
      } else {
        const nextHover = hit(p.x, p.y);
        canvas.style.cursor = nextHover ? "pointer" : "grab";
        if (nextHover !== hover) {
          hover = nextHover;
          schedule();
        }
      }
    };

    const up = (event: PointerEvent) => {
      const p = local(event);
      if (!dragged && pointers.size === 1) {
        const id = hit(p.x, p.y);
        if (id) current.current.onSelectNode(id);
      }
      pointers.delete(event.pointerId);
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    };

    const cancel = (event: PointerEvent) => {
      pointers.delete(event.pointerId);
      dragged = true;
    };

    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      cameraRef.current.zoom = Math.max(0.35, Math.min(5, cameraRef.current.zoom * Math.exp(-event.deltaY * 0.001)));
      schedule();
    };

    const key = (event: KeyboardEvent) => {
      if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
        event.preventDefault();
        cameraRef.current.yaw += event.key === "ArrowLeft" ? -0.15 : event.key === "ArrowRight" ? 0.15 : 0;
        cameraRef.current.pitch += event.key === "ArrowUp" ? -0.15 : event.key === "ArrowDown" ? 0.15 : 0;
        schedule();
      } else if (event.key === "Enter") {
        const node =
          current.current.nodes.find(n => n.id === current.current.selectedNodeId) ??
          current.current.nodes.find(n => isMemoryCoreLabel(n.label));
        if (node) current.current.onSelectNode(node.id);
      }
    };

    const handleContextLost = (event: Event) => {
      event.preventDefault();
      contextLost = true;
      glResources = null;
      uploadedStaticRef = null;
      uploadedPackedRef = null;
      lastUploadedFocus = undefined;
      current.current.onRendererFallbackChange?.(true);
      schedule();
    };

    const handleContextRestored = () => {
      contextLost = false;
      uploadedStaticRef = null;
      uploadedPackedRef = null;
      lastUploadedFocus = undefined;
      if (tryInitWebGL2()) {
        current.current.onRendererFallbackChange?.(false);
        if (overlayContext) overlayContext.clearRect(0, 0, width, height);
      }
      schedule();
    };

    const observer = typeof ResizeObserver !== "undefined" ? new ResizeObserver(resize) : null;
    observer?.observe(canvas);
    resize();

    canvas.addEventListener("pointerdown", down);
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerup", up);
    canvas.addEventListener("pointercancel", cancel);
    canvas.addEventListener("wheel", wheel, { passive: false });
    canvas.addEventListener("keydown", key);
    canvas.addEventListener("webglcontextlost", handleContextLost);
    canvas.addEventListener("webglcontextrestored", handleContextRestored);

    return () => {
      observer?.disconnect();
      cancelAnimationFrame(frame);
      redrawRef.current = () => {};
      uploadGraphRef.current = () => {};
      uploadPositionsRef.current = () => {};
      canvas.removeEventListener("pointerdown", down);
      canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerup", up);
      canvas.removeEventListener("pointercancel", cancel);
      canvas.removeEventListener("wheel", wheel);
      canvas.removeEventListener("keydown", key);
      canvas.removeEventListener("webglcontextlost", handleContextLost);
      canvas.removeEventListener("webglcontextrestored", handleContextRestored);
    };
  }, []);

  useEffect(() => {
    uploadGraphRef.current();
    redrawRef.current();
  }, [staticGraphData]);

  useEffect(() => {
    uploadPositionsRef.current();
    redrawRef.current();
  }, [packed]);

  useEffect(() => {
    redrawRef.current();
  }, [showTitles, selectedNodeId]);

  return (
    <div className="memory-3d-surface">
      <canvas
        ref={canvasRef}
        tabIndex={0}
        role="application"
        aria-label="3D memory graph. Drag or use arrow keys to rotate 360 degrees. Scroll or pinch to zoom. Click a node to open."
        aria-busy={loading}
        data-titles-visible={showTitles}
        data-layout-volume={volumeRatio.toFixed(4)}
      />
      <canvas
        ref={overlayCanvasRef}
        className="memory-3d-labels-overlay"
        aria-hidden="true"
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
      />
      <span className="memory-3d-hint">Drag to rotate 360° · Scroll or pinch to zoom · Click to explore</span>
    </div>
  );
}
