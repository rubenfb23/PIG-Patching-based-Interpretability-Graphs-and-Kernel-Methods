import { useMemo } from "react";
import { Canvas, ThreeEvent } from "@react-three/fiber";
import { Line, OrbitControls, Text } from "@react-three/drei";
import * as THREE from "three";
import { CausalEdgeData, GraphPayload } from "../types";

// Color scale endpoints for causal classification
const CAUSAL_MEDIATED_LOW  = "#7c2d12";   // dark orange
const CAUSAL_MEDIATED_HIGH = "#f97316";   // bright orange
const CAUSAL_PARALLEL_LOW  = "#1e3a5f";   // dark blue
const CAUSAL_PARALLEL_HIGH = "#38bdf8";   // bright sky-blue
const CAUSAL_COLOR_INSIG   = "#374151";   // dark gray

type Props = {
  graph: GraphPayload | null;
  selectedNodeId: number | null;
  localNodeIds: Set<number>;
  localEdgeKeys: Set<string>;
  isolateLocalCircuit: boolean;
  onSelectNode: (nodeId: number) => void;
  causalEdges: CausalEdgeData[];
  causalSigOnly: boolean;
  causalOnlyMode: boolean;
};

type Point3 = [number, number, number];

const ATTENTION_BASE_Z = -6;
const ATTENTION_HEAD_STEP_Z = 0.5;
const MLP_Z = 1;
const RESIDUAL_Z = 2;
const LAYER_SPACING = 1.45;

const COMPONENT_LEGEND_ENTRIES: Array<{ label: string; z: number; color: string }> = [
  { label: "res", z: RESIDUAL_Z, color: "#a78bfa" },
  { label: "mlp", z: MLP_Z, color: "#fb923c" },
  { label: "att[0]", z: ATTENTION_BASE_Z, color: "#67e8f9" },
];

const componentOffset = (type: string, head: number | null): number => {
  if (type === "att") {
    const safeHead = head ?? 0;
    return ATTENTION_BASE_Z + safeHead * ATTENTION_HEAD_STEP_Z;
  }
  if (type === "mlp") {
    return MLP_Z;
  }
  return RESIDUAL_Z;
};

const toPosition = (
  layer: number,
  token: number,
  type: string,
  head: number | null,
): Point3 => [
    token,
    layer * LAYER_SPACING,
    componentOffset(type, head),
  ];

export function GraphScene({
  graph,
  selectedNodeId,
  localNodeIds,
  localEdgeKeys,
  isolateLocalCircuit,
  onSelectNode,
  causalEdges,
  causalSigOnly,
  causalOnlyMode,
}: Props) {
  const nodeMap = useMemo(() => {
    const map = new Map<number, Point3>();
    if (!graph) {
      return map;
    }
    for (const node of graph.nodes) {
      map.set(node.id, toPosition(node.layer, node.token, node.type, node.head));
    }
    return map;
  }, [graph]);

  const edges = useMemo(() => {
    if (!graph) {
      return [] as { id: string; points: Point3[]; weight: number }[];
    }
    return graph.edges
      .map((edge) => {
        const src = nodeMap.get(edge.src);
        const dst = nodeMap.get(edge.dst);
        if (!src || !dst) {
          return null;
        }
        return {
          id: `${edge.src}-${edge.dst}`,
          points: [src, dst],
          weight: edge.weight,
        };
      })
      .filter((entry): entry is { id: string; points: Point3[]; weight: number } =>
        entry !== null,
      );
  }, [graph, nodeMap]);

  // Compute causal edge geometry directly from node coordinates (no graph lookup needed)
  const causalEdgeGeoms = useMemo(() => {
    const visible = causalSigOnly ? causalEdges.filter((e) => e.significant) : causalEdges;
    return visible.map((e) => {
      const srcPos = toPosition(e.src.layer, e.src.token, e.src.type, e.src.head);
      const dstPos = toPosition(e.dst.layer, e.dst.token, e.dst.type, e.dst.head);
      const iNorm = e.significant ? Math.min(e.I_mean / 0.08, 1) : 0;
      let color: string;
      if (!e.significant) {
        color = CAUSAL_COLOR_INSIG;
      } else if (e.classification === "mediated") {
        color = "#" + new THREE.Color(CAUSAL_MEDIATED_LOW)
          .lerp(new THREE.Color(CAUSAL_MEDIATED_HIGH), iNorm)
          .getHexString();
      } else {
        color = "#" + new THREE.Color(CAUSAL_PARALLEL_LOW)
          .lerp(new THREE.Color(CAUSAL_PARALLEL_HIGH), iNorm)
          .getHexString();
      }
      const lineWidth = e.significant ? 2.0 : 0.5;
      const opacity = e.significant ? 0.9 : 0.15;
      return { key: e.edge_id, points: [srcPos, dstPos] as Point3[], color, lineWidth, opacity };
    });
  }, [causalEdges, causalSigOnly]);

  const nodes = graph?.nodes ?? [];
  const tokenLabels = graph?.token_labels ?? [];
  const axisTokenEnd = Math.max(3, (graph?.num_tokens ?? 0) + 1);
  const axisLayerEnd = Math.max(3, (graph?.num_layers ?? 0) * LAYER_SPACING + 0.5);
  const maxAttentionHead = Math.max(
    0,
    ...nodes.map((node) => (node.type === "att" ? (node.head ?? 0) : 0)),
  );
  const axisComponentEnd = Math.max(
    RESIDUAL_Z + 0.4,
    ATTENTION_BASE_Z + maxAttentionHead * ATTENTION_HEAD_STEP_Z + 0.4,
  );

  return (
    <Canvas
      camera={{ position: [15, 12, 15], fov: 55 }}
      onContextMenu={(event) => event.preventDefault()}
    >
      <ambientLight intensity={0.5} />
      <pointLight position={[20, 20, 20]} intensity={1.2} />
      <axesHelper args={[6]} />

      <Text
        position={[axisTokenEnd, -0.35, 0]}
        fontSize={0.25}
        color="#ef4444"
        anchorX="left"
        anchorY="middle"
      >
        X: Token
      </Text>
      <Text
        position={[-0.25, axisLayerEnd, 0]}
        fontSize={0.25}
        color="#22c55e"
        anchorX="right"
        anchorY="middle"
      >
        Y: Layer
      </Text>
      <Text
        position={[-0.2, 0.2, axisComponentEnd]}
        fontSize={0.25}
        color="#3b82f6"
        anchorX="right"
        anchorY="middle"
      >
        Z: Component
      </Text>

      {/* Token tick labels */}
      {Array.from({ length: graph?.num_tokens ?? 0 }, (_, tokenIndex) => {
        const raw = tokenLabels[tokenIndex] ?? String(tokenIndex);
        const label = `t${tokenIndex + 1} (${raw})`;
        return (
          <Text
            key={`tok-${tokenIndex}`}
            position={[tokenIndex, -0.65, 0]}
            fontSize={0.18}
            color="#ef4444"
            anchorX="center"
            anchorY="top"
            rotation={[0, 0, -Math.PI / 4]}
          >
            {label}
          </Text>
        );
      })}

      {/* Layer tick labels */}
      {Array.from({ length: graph?.num_layers ?? 0 }, (_, layerIndex) => (
        <Text
          key={`lay-${layerIndex}`}
          position={[-0.55, layerIndex * LAYER_SPACING, 0]}
          fontSize={0.18}
          color="#22c55e"
          anchorX="right"
          anchorY="middle"
        >
          {`L${layerIndex}`}
        </Text>
      ))}

      {/* Component (Z-axis) legend */}
      {COMPONENT_LEGEND_ENTRIES.map((entry) => (
        <Text
          key={`comp-${entry.label}`}
          position={[-0.2, 0.2, entry.z]}
          fontSize={0.2}
          color={entry.color}
          anchorX="right"
          anchorY="middle"
        >
          {entry.label}
        </Text>
      ))}

      {nodes.map((node) => {
        const position = nodeMap.get(node.id);
        if (!position) {
          return null;
        }
        const isSelected = node.id === selectedNodeId;
        const isInLocalCircuit = localNodeIds.has(node.id);
        const shouldFade = isolateLocalCircuit && selectedNodeId !== null && !isInLocalCircuit;

        const nodeColor = isSelected
          ? "#fde047"
          : shouldFade
            ? "#3f4c5f"
            : "#7dd3fc";
        const nodeOpacity = shouldFade ? 0.22 : 0.95;
        const nodeRadius = isSelected ? 0.14 : 0.08;

        const handleSelect = (event: ThreeEvent<PointerEvent>) => {
          event.stopPropagation();
          onSelectNode(node.id);
        };

        return (
          <mesh key={node.id} position={position} onPointerDown={handleSelect}>
            <sphereGeometry args={[nodeRadius, 10, 10]} />
            <meshStandardMaterial
              color={nodeColor}
              transparent
              opacity={nodeOpacity}
            />
            <mesh onPointerDown={handleSelect}>
              <sphereGeometry args={[Math.max(0.16, nodeRadius * 2.25), 8, 8]} />
              <meshBasicMaterial transparent opacity={0} depthWrite={false} />
            </mesh>
          </mesh>
        );
      })}

      {edges.map((edge) => {
        const isInLocalCircuit = localEdgeKeys.has(edge.id);
        const shouldFade = isolateLocalCircuit && selectedNodeId !== null && !isInLocalCircuit;
        const baseOpacity = causalOnlyMode ? 0.12 : 0.5;
        const opacity = shouldFade ? 0.05 : baseOpacity;
        const lineWidth = isInLocalCircuit ? 1.25 : 0.8;

        return (
          <Line
            key={edge.id}
            points={edge.points}
            color={edge.weight >= 0 ? "#22c55e" : "#f43f5e"}
            lineWidth={lineWidth}
            transparent
            opacity={opacity}
            raycast={() => null}
          />
        );
      })}

      {/* Causal overlay edges */}
      {causalEdgeGeoms.map((ce) => (
        <Line
          key={`causal-${ce.key}`}
          points={ce.points}
          color={ce.color}
          lineWidth={ce.lineWidth}
          transparent
          opacity={ce.opacity}
          raycast={() => null}
        />
      ))}

      <OrbitControls
        makeDefault
        enablePan
        screenSpacePanning
        panSpeed={1.15}
        rotateSpeed={0.9}
        zoomSpeed={1.0}
        keyPanSpeed={10}
        mouseButtons={{
          LEFT: THREE.MOUSE.ROTATE,
          MIDDLE: THREE.MOUSE.PAN,
          RIGHT: THREE.MOUSE.PAN,
        }}
      />
    </Canvas>
  );
}
