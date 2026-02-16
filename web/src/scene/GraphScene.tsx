import { useMemo } from "react";
import { Canvas, ThreeEvent } from "@react-three/fiber";
import { Line, OrbitControls, Text } from "@react-three/drei";
import * as THREE from "three";
import { GraphPayload } from "../types";

type Props = {
  graph: GraphPayload | null;
  selectedNodeId: number | null;
  localNodeIds: Set<number>;
  localEdgeKeys: Set<string>;
  isolateLocalCircuit: boolean;
  onSelectNode: (nodeId: number) => void;
};

type Point3 = [number, number, number];

const ATTENTION_BASE_Z = -6;
const ATTENTION_HEAD_STEP_Z = 0.5;
const MLP_Z = 1;
const RESIDUAL_Z = 2;
const LAYER_SPACING = 1.45;

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

  const nodes = graph?.nodes ?? [];
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
        Z: Component (heads + mlp/res)
      </Text>

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
        const opacity = shouldFade ? 0.07 : 0.5;
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
