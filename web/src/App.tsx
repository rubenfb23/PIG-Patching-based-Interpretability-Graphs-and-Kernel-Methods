import { useEffect, useMemo, useRef, useState } from "react";
import { ControlsPanel } from "./components/ControlsPanel";
import { GraphScene } from "./scene/GraphScene";
import {
  GraphEdge,
  GraphFilter,
  GraphNode,
  GraphPayload,
  ViewerMessage,
} from "./types";
import { ViewerClient } from "./ws/client";

const DEFAULT_WS_URL = "ws://127.0.0.1:8765";

export default function App() {
  const [slices, setSlices] = useState<string[]>([]);
  const [graph, setGraph] = useState<GraphPayload | null>(null);
  const [filter, setFilter] = useState<GraphFilter | null>(null);
  const [status, setStatus] = useState("connecting");
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(null);
  const [isolateLocalCircuit, setIsolateLocalCircuit] = useState(true);
  const clientRef = useRef<ViewerClient | null>(null);

  useEffect(() => {
    const client = new ViewerClient(DEFAULT_WS_URL, (message: ViewerMessage) => {
      if (message.type === "init") {
        setSlices(message.payload.slices);
        setGraph(message.payload.graph);
        setFilter(message.payload.default_filter);
        setStatus("connected");
        return;
      }

      if (message.type === "graph_update") {
        setGraph(message.payload);
        setSelectedNodeId(null);
        return;
      }

      if (message.type === "error") {
        setStatus(message.message);
      }
    });

    clientRef.current = client;
  }, []);

  useEffect(() => {
    if (!filter || !clientRef.current) {
      return;
    }
    clientRef.current.requestUpdate(filter);
  }, [filter]);

  const selectedNode = useMemo((): GraphNode | null => {
    if (!graph || selectedNodeId === null) {
      return null;
    }
    return graph.nodes.find((node) => node.id === selectedNodeId) ?? null;
  }, [graph, selectedNodeId]);

  const localNodeIds = useMemo(() => {
    const ids = new Set<number>();
    if (!graph || selectedNodeId === null) {
      return ids;
    }
    ids.add(selectedNodeId);
    for (const edge of graph.edges) {
      if (edge.src === selectedNodeId || edge.dst === selectedNodeId) {
        ids.add(edge.src);
        ids.add(edge.dst);
      }
    }
    return ids;
  }, [graph, selectedNodeId]);

  const localEdgeKeys = useMemo(() => {
    const keys = new Set<string>();
    if (!graph || selectedNodeId === null) {
      return keys;
    }
    for (const edge of graph.edges) {
      const srcInLocal = localNodeIds.has(edge.src);
      const dstInLocal = localNodeIds.has(edge.dst);
      if (srcInLocal && dstInLocal) {
        keys.add(`${edge.src}-${edge.dst}`);
      }
    }
    return keys;
  }, [graph, localNodeIds, selectedNodeId]);

  const nodeById = useMemo(() => {
    const map = new Map<number, GraphNode>();
    if (!graph) {
      return map;
    }
    for (const node of graph.nodes) {
      map.set(node.id, node);
    }
    return map;
  }, [graph]);

  const topConnections = useMemo(() => {
    const empty = {
      incoming: [] as Array<{ edge: GraphEdge; node: GraphNode | null }>,
      outgoing: [] as Array<{ edge: GraphEdge; node: GraphNode | null }>,
    };
    if (!graph || selectedNodeId === null) {
      return empty;
    }

    const incoming = graph.edges
      .filter((edge) => edge.dst === selectedNodeId)
      .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
      .slice(0, 5)
      .map((edge) => ({ edge, node: nodeById.get(edge.src) ?? null }));

    const outgoing = graph.edges
      .filter((edge) => edge.src === selectedNodeId)
      .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
      .slice(0, 5)
      .map((edge) => ({ edge, node: nodeById.get(edge.dst) ?? null }));

    return { incoming, outgoing };
  }, [graph, nodeById, selectedNodeId]);

  if (!filter) {
    return (
      <main className="layout">
        <div className="badge">Waiting for init ({status})…</div>
      </main>
    );
  }

  return (
    <main className="layout">
      <ControlsPanel
        slices={slices}
        filter={filter}
        graph={graph}
        onFilterChange={setFilter}
        selectedNode={selectedNode}
        topIncoming={topConnections.incoming}
        topOutgoing={topConnections.outgoing}
        isolateLocalCircuit={isolateLocalCircuit}
        onToggleIsolate={setIsolateLocalCircuit}
        onClearSelection={() => setSelectedNodeId(null)}
      />
      <div className="viewport">
        <div className="badge">ws: {status}</div>
        <div className="badge controls-hint">
          Rotate: left click · Pan: right/middle click or arrow keys · Zoom: wheel
        </div>
        <GraphScene
          graph={graph}
          selectedNodeId={selectedNodeId}
          localNodeIds={localNodeIds}
          localEdgeKeys={localEdgeKeys}
          isolateLocalCircuit={isolateLocalCircuit}
          onSelectNode={setSelectedNodeId}
        />
      </div>
    </main>
  );
}
