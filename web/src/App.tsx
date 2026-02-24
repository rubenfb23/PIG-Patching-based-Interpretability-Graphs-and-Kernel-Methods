import { useEffect, useMemo, useRef, useState } from "react";
import { ControlsPanel } from "./components/ControlsPanel";
import { GraphScene } from "./scene/GraphScene";
import {
  CausalEdgeData,
  CausalOverlayMeta,
  GraphEdge,
  GraphFilter,
  GraphNode,
  GraphPayload,
  ViewerMessage,
} from "./types";
import { ViewerClient } from "./ws/client";

// Parses "L2T9.att[4]" → { layer:2, token:9, type:"att", head:4 }
function parseNodeLabel(label: string): { layer: number; token: number; type: string; head: number | null } | null {
  const m = label.match(/^L(\d+)T(\d+)\.(\w+)(?:\[(\d+)\])?$/);
  if (!m) return null;
  return {
    layer: parseInt(m[1], 10),
    token: parseInt(m[2], 10),
    type: m[3],
    head: m[4] !== undefined ? parseInt(m[4], 10) : null,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function parseCausalEval(json: any): { edges: CausalEdgeData[]; meta: CausalOverlayMeta } {
  const raw = json.edges ?? [];
  const edges: CausalEdgeData[] = [];
  for (const e of raw) {
    const [srcLabel, dstLabel] = (e.edge_id as string).split("->");
    const src = parseNodeLabel(srcLabel);
    const dst = parseNodeLabel(dstLabel);
    if (!src || !dst) continue;
    const la = e.level_a?.I ?? {};
    const lc = e.level_c?.necessity ?? {};
    const lb = e.level_b?.M ?? {};
    const I_mean: number = la.mean ?? 0;
    const I_p: number = la.p_value ?? 1;
    const sig = I_p < 0.05 && I_mean > 0;
    edges.push({
      edge_id: e.edge_id,
      src,
      dst,
      I_mean,
      I_ci_low: la.ci_low ?? 0,
      I_ci_high: la.ci_high ?? 0,
      I_p_value: I_p,
      necessity_mean: lc.mean ?? 0,
      M_mean: lb.mean ?? 0,
      classification: e.diagnostics?.classification ?? "parallel_or_synergy",
      significant: sig,
    });
  }
  const sig = edges.filter((e) => e.significant);
  const meta: CausalOverlayMeta = {
    model_name: json.model_name ?? "",
    num_edges: edges.length,
    num_significant: sig.length,
    num_mediated: sig.filter((e) => e.classification === "mediated").length,
    num_parallel: sig.filter((e) => e.classification === "parallel_or_synergy").length,
  };
  return { edges, meta };
}

const unique = (items: Array<string | undefined | null>): string[] => {
  const out: string[] = [];
  for (const item of items) {
    const value = item?.trim();
    if (!value || out.includes(value)) {
      continue;
    }
    out.push(value);
  }
  return out;
};

const wsScheme = window.location.protocol === "https:" ? "wss" : "ws";
const wsHost = window.location.host;
const normalizeWsUrl = (value: string): string => value.replace(/\/+$/, "");

const wsFromCurrentPath = (relativePath: string): string => {
  const url = new URL(relativePath, window.location.href);
  url.protocol = wsScheme === "wss" ? "wss:" : "ws:";
  return normalizeWsUrl(url.toString());
};

const deriveForwardedWsRoots = (targetPort: number): string[] => {
  const page = new URL(window.location.href);
  const wsProtocol = wsScheme === "wss" ? "wss:" : "ws:";
  const roots: string[] = [];

  if (page.port) {
    const byPort = new URL(page.href);
    byPort.protocol = wsProtocol;
    byPort.port = String(targetPort);
    byPort.pathname = "/";
    byPort.search = "";
    byPort.hash = "";
    roots.push(normalizeWsUrl(byPort.toString()));
  }

  const proxyPathMatch = page.pathname.match(/^(.*\/proxy\/)(\d+)(\/.*)?$/);
  if (proxyPathMatch) {
    const byProxyPath = new URL(page.href);
    byProxyPath.protocol = wsProtocol;
    byProxyPath.pathname = `${proxyPathMatch[1]}${targetPort}${proxyPathMatch[3] ?? "/"}`;
    byProxyPath.search = "";
    byProxyPath.hash = "";
    roots.push(normalizeWsUrl(byProxyPath.toString()));
  }

  const byHost = new URL(page.href);
  const replacedHost = byHost.hostname.replace(/(^|-)5173(?=[.-]|$)/, `$1${targetPort}`);
  if (replacedHost !== byHost.hostname) {
    byHost.protocol = wsProtocol;
    byHost.hostname = replacedHost;
    byHost.pathname = "/";
    byHost.search = "";
    byHost.hash = "";
    roots.push(normalizeWsUrl(byHost.toString()));
  }

  return unique(roots);
};

const BASE_DIRECT_WS = `ws://127.0.0.1:8765`;
const DIFF_DIRECT_WS = `ws://127.0.0.1:8766`;
const BASE_PATH_WS = wsFromCurrentPath("./ws-base");
const DIFF_PATH_WS = wsFromCurrentPath("./ws-diff");
const BASE_PROXY_WS = normalizeWsUrl(`${wsScheme}://${wsHost}/ws-base`);
const DIFF_PROXY_WS = normalizeWsUrl(`${wsScheme}://${wsHost}/ws-diff`);
const BASE_FORWARD_ROOTS = deriveForwardedWsRoots(8765);
const DIFF_FORWARD_ROOTS = deriveForwardedWsRoots(8766);
const ENV_BASE_WS = import.meta.env.VITE_WS_BASE_URL as string | undefined;
const ENV_DIFF_WS = import.meta.env.VITE_WS_DIFF_URL as string | undefined;

const MODELS = [
  {
    label: "GPT-2 Base",
    url: BASE_PATH_WS,
    fallbackUrls: unique([
      ENV_BASE_WS,
      BASE_PATH_WS,
      BASE_PROXY_WS,
      ...BASE_FORWARD_ROOTS,
      BASE_DIRECT_WS,
    ]),
  },
  {
    label: "GPT-2 lr2e5_acc4",
    url: DIFF_PATH_WS,
    fallbackUrls: unique([
      ENV_DIFF_WS,
      DIFF_PATH_WS,
      DIFF_PROXY_WS,
      ...DIFF_FORWARD_ROOTS,
      DIFF_DIRECT_WS,
    ]),
  },
];
const BUILD_MARKER = "viewer-ws-fallback-v3";

export default function App() {
  const [slices, setSlices] = useState<string[]>([]);
  const [graph, setGraph] = useState<GraphPayload | null>(null);
  const [filter, setFilter] = useState<GraphFilter | null>(null);
  const [status, setStatus] = useState("connecting");
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(null);
  const [isolateLocalCircuit, setIsolateLocalCircuit] = useState(true);
  const [causalEdges, setCausalEdges] = useState<CausalEdgeData[]>([]);
  const [causalMeta, setCausalMeta] = useState<CausalOverlayMeta | null>(null);
  const [showCausalOverlay, setShowCausalOverlay] = useState(true);
  const [causalSigOnly, setCausalSigOnly] = useState(false);
  const [causalOnlyMode, setCausalOnlyMode] = useState(false);
  const [causalClassFilter, setCausalClassFilter] = useState<"all" | "mediated" | "parallel_or_synergy">("all");
  const [necessityMin, setNecessityMin] = useState(0);
  const [selectedModelIdx, setSelectedModelIdx] = useState(0);
  const [connectedUrl, setConnectedUrl] = useState<string>("");
  const [connectionTrace, setConnectionTrace] = useState<string[]>([]);
  const clientRef = useRef<ViewerClient | null>(null);
  const selectedNodeIdRef = useRef<number | null>(null);
  const selectedNodeSignatureRef = useRef<{
    layer: number;
    token: number;
    type: string;
    head: number | null;
  } | null>(null);

  useEffect(() => {
    selectedNodeIdRef.current = selectedNodeId;
  }, [selectedNodeId]);

  useEffect(() => {
    if (!graph || selectedNodeId === null) {
      selectedNodeSignatureRef.current = null;
      return;
    }
    const selectedNode = graph.nodes.find((node) => node.id === selectedNodeId);
    if (!selectedNode) {
      selectedNodeSignatureRef.current = null;
      return;
    }
    selectedNodeSignatureRef.current = {
      layer: selectedNode.layer,
      token: selectedNode.token,
      type: selectedNode.type,
      head: selectedNode.head,
    };
  }, [graph, selectedNodeId]);

  useEffect(() => {
    const model = MODELS[selectedModelIdx];
    const candidates = model.fallbackUrls;
    console.log(
      `[viewer][app] selecting model=${model.label} candidates=${candidates.join(", ")}`,
    );
    setStatus("connecting");
    setConnectedUrl("");
    setConnectionTrace([]);
    setSlices([]);
    setGraph(null);
    setFilter(null);
    setSelectedNodeId(null);
    let timeoutId: number | null = null;
    let activeClient: ViewerClient | null = null;
    let isDisposed = false;
    let isConnected = false;

    const pushTrace = (line: string) => {
      setConnectionTrace((prev) => [...prev.slice(-7), line]);
    };

    const clearAttemptTimer = () => {
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
        timeoutId = null;
      }
    };

    const attempt = (candidateIndex: number) => {
      if (isDisposed || isConnected) {
        return;
      }
      if (candidateIndex >= candidates.length) {
        setStatus("all_ws_endpoints_failed");
        pushTrace("all endpoints failed");
        return;
      }

      const endpoint = candidates[candidateIndex];
      setStatus(`connecting ${candidateIndex + 1}/${candidates.length}`);
      pushTrace(`try ${endpoint}`);
      console.log(
        `[viewer][app] attempt ${candidateIndex + 1}/${candidates.length} endpoint=${endpoint}`,
      );

      clearAttemptTimer();
      const client = new ViewerClient(
        endpoint,
        (message: ViewerMessage) => {
          if (isDisposed || isConnected || client !== activeClient) {
            return;
          }
          if (message.type === "init") {
            isConnected = true;
            clearAttemptTimer();
            pushTrace(`init ok ${endpoint}`);
            setConnectedUrl(endpoint);
            console.log(
              `[viewer][app] init received endpoint=${endpoint} slices=${message.payload.slices.length} nodes=${message.payload.graph.nodes.length} edges=${message.payload.graph.edges.length}`,
            );
            setSlices(message.payload.slices);
            setGraph(message.payload.graph);
            setFilter(message.payload.default_filter);
            setStatus("connected");
            clientRef.current = client;
            return;
          }

          if (message.type === "graph_update") {
            console.log(
              `[viewer][app] graph_update endpoint=${endpoint} nodes=${message.payload.nodes.length} edges=${message.payload.edges.length}`,
            );
            const nextGraph = message.payload;
            setGraph(nextGraph);

            const currentSelectedId = selectedNodeIdRef.current;
            if (currentSelectedId === null) {
              return;
            }

            const hasSameId = nextGraph.nodes.some((node) => node.id === currentSelectedId);
            if (hasSameId) {
              setSelectedNodeId(currentSelectedId);
              return;
            }

            const signature = selectedNodeSignatureRef.current;
            if (!signature) {
              setSelectedNodeId(null);
              return;
            }

            const remappedNode = nextGraph.nodes.find(
              (node) =>
                node.layer === signature.layer &&
                node.token === signature.token &&
                node.type === signature.type &&
                node.head === signature.head,
            );
            setSelectedNodeId(remappedNode?.id ?? null);
            return;
          }

          if (message.type === "error") {
            pushTrace(`backend error ${endpoint}: ${message.message}`);
            console.error(`[viewer][app] backend error endpoint=${endpoint}: ${message.message}`);
            setStatus(message.message);
          }
        },
        (socketStatus) => {
          if (isDisposed || isConnected || client !== activeClient) {
            return;
          }
          pushTrace(`${endpoint} -> ${socketStatus}`);
          console.log(`[viewer][app] socket status endpoint=${endpoint}: ${socketStatus}`);

          if (socketStatus === "socket_error" || socketStatus.startsWith("socket_closed(")) {
            clearAttemptTimer();
            attempt(candidateIndex + 1);
            return;
          }

          if (socketStatus !== "socket_open") {
            setStatus(socketStatus);
          }
        },
      );

      activeClient = client;
      timeoutId = window.setTimeout(() => {
        if (isDisposed || isConnected || client !== activeClient) {
          return;
        }
        pushTrace(`timeout ${endpoint}`);
        console.warn(`[viewer][app] init timeout endpoint=${endpoint}`);
        setStatus(`init_timeout (${endpoint})`);
        client.close();
        attempt(candidateIndex + 1);
      }, 5000);
    };

    attempt(0);

    return () => {
      isDisposed = true;
      clearAttemptTimer();
      activeClient?.close();
      clientRef.current = null;
    };
  }, [selectedModelIdx]);

  useEffect(() => {
    if (!filter || !clientRef.current) {
      return;
    }
    console.log(
      `[viewer][app] requestUpdate slice=${filter.slice_id} max_edges=${filter.max_edges} min_abs_weight=${filter.min_abs_weight}`,
    );
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

  const handleCausalFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const json = JSON.parse(ev.target?.result as string);
        const { edges, meta } = parseCausalEval(json);
        setCausalEdges(edges);
        setCausalMeta(meta);
        setShowCausalOverlay(true);
      } catch {
        // silently ignore parse errors
      }
    };
    reader.readAsText(file);
  };

  if (!filter) {
    return (
      <main className="layout">
        <div className="panel">
          <h2>Model</h2>
          <select
            value={selectedModelIdx}
            onChange={(event) => setSelectedModelIdx(parseInt(event.target.value, 10))}
          >
            {MODELS.map((model, idx) => (
              <option key={model.label} value={idx}>
                {model.label}
              </option>
            ))}
          </select>
        </div>
        <div className="badge">Waiting for init ({status})…</div>
        <div className="badge">build: {BUILD_MARKER}</div>
        <div className="badge">candidate: {MODELS[selectedModelIdx].url}</div>
        {connectedUrl ? <div className="badge">connected: {connectedUrl}</div> : null}
        <div className="panel" style={{ maxWidth: 560 }}>
          <h2>Connection trace</h2>
          {connectionTrace.length === 0 ? (
            <div className="muted">no events yet</div>
          ) : (
            <ul>
              {connectionTrace.map((line, idx) => (
                <li key={`${idx}-${line}`}>{line}</li>
              ))}
            </ul>
          )}
        </div>
      </main>
    );
  }

  return (
    <main className="layout">
      <ControlsPanel
        models={MODELS}
        selectedModelIdx={selectedModelIdx}
        onSelectModel={setSelectedModelIdx}
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
        causalMeta={causalMeta}
        showCausalOverlay={showCausalOverlay}
        onToggleCausalOverlay={setShowCausalOverlay}
        causalSigOnly={causalSigOnly}
        onToggleCausalSigOnly={setCausalSigOnly}
        causalOnlyMode={causalOnlyMode}
        onToggleCausalOnlyMode={setCausalOnlyMode}
        causalClassFilter={causalClassFilter}
        onSetCausalClassFilter={setCausalClassFilter}
        necessityMin={necessityMin}
        onSetNecessityMin={setNecessityMin}
        onLoadCausalFile={handleCausalFile}
      />
      <div className="viewport">
        <div className="badge">ws: {status}{connectedUrl ? ` (${connectedUrl})` : ""}</div>
        <div className="badge">build: {BUILD_MARKER}</div>
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
          causalEdges={showCausalOverlay ? causalEdges.filter((e) =>
            (causalClassFilter === "all" || e.classification === causalClassFilter) &&
            e.necessity_mean >= necessityMin
          ) : []}
          causalSigOnly={causalSigOnly}
          causalOnlyMode={causalOnlyMode}
        />
      </div>
    </main>
  );
}
