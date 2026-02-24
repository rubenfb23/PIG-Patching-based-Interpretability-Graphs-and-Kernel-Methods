import { useRef } from "react";
import { CausalOverlayMeta, GraphEdge, GraphFilter, GraphNode, GraphPayload } from "../types";

type ModelOption = { label: string; url: string };

type Props = {
  models: ModelOption[];
  selectedModelIdx: number;
  onSelectModel: (idx: number) => void;
  slices: string[];
  filter: GraphFilter;
  graph: GraphPayload | null;
  onFilterChange: (next: GraphFilter) => void;
  selectedNode: GraphNode | null;
  topIncoming: Array<{ edge: GraphEdge; node: GraphNode | null }>;
  topOutgoing: Array<{ edge: GraphEdge; node: GraphNode | null }>;
  isolateLocalCircuit: boolean;
  onToggleIsolate: (enabled: boolean) => void;
  onClearSelection: () => void;
  causalMeta: CausalOverlayMeta | null;
  showCausalOverlay: boolean;
  onToggleCausalOverlay: (enabled: boolean) => void;
  causalSigOnly: boolean;
  onToggleCausalSigOnly: (enabled: boolean) => void;
  causalOnlyMode: boolean;
  onToggleCausalOnlyMode: (enabled: boolean) => void;
  causalClassFilter: "all" | "mediated" | "parallel_or_synergy";
  onSetCausalClassFilter: (v: "all" | "mediated" | "parallel_or_synergy") => void;
  necessityMin: number;
  onSetNecessityMin: (v: number) => void;
  onLoadCausalFile: (file: File) => void;
};

const toNumber = (value: string): number | undefined => {
  if (value.trim() === "") {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
};

export function ControlsPanel({
  models,
  selectedModelIdx,
  onSelectModel,
  slices,
  filter,
  graph,
  onFilterChange,
  selectedNode,
  topIncoming,
  topOutgoing,
  isolateLocalCircuit,
  onToggleIsolate,
  onClearSelection,
  causalMeta,
  showCausalOverlay,
  onToggleCausalOverlay,
  causalSigOnly,
  onToggleCausalSigOnly,
  causalOnlyMode,
  onToggleCausalOnlyMode,
  causalClassFilter,
  onSetCausalClassFilter,
  necessityMin,
  onSetNecessityMin,
  onLoadCausalFile,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  return (
    <aside className="panel">
      <h1>PIG 4D Viewer</h1>
      <p>Explore slice/layer/token/threshold in real time.</p>

      <div className="field">
        <label>Model</label>
        <select
          value={selectedModelIdx}
          onChange={(ev) => onSelectModel(Number(ev.target.value))}
        >
          {models.map((m, i) => (
            <option key={m.url} value={i}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      <div className="field">
        <label>Slice</label>
        <select
          value={filter.slice_id}
          onChange={(event) =>
            onFilterChange({ ...filter, slice_id: event.target.value })
          }
        >
          {slices.map((sliceId) => (
            <option key={sliceId} value={sliceId}>
              {sliceId}
            </option>
          ))}
        </select>
      </div>

      <div className="field">
        <label>Layer min</label>
        <input
          value={filter.layer_min ?? ""}
          onChange={(event) =>
            onFilterChange({
              ...filter,
              layer_min: toNumber(event.target.value),
            })
          }
        />
      </div>

      <div className="field">
        <label>Layer max</label>
        <input
          value={filter.layer_max ?? ""}
          onChange={(event) =>
            onFilterChange({
              ...filter,
              layer_max: toNumber(event.target.value),
            })
          }
        />
      </div>

      <div className="field">
        <label>Token min</label>
        <input
          value={filter.token_min ?? ""}
          onChange={(event) =>
            onFilterChange({
              ...filter,
              token_min: toNumber(event.target.value),
            })
          }
        />
      </div>

      <div className="field">
        <label>Token max</label>
        <input
          value={filter.token_max ?? ""}
          onChange={(event) =>
            onFilterChange({
              ...filter,
              token_max: toNumber(event.target.value),
            })
          }
        />
      </div>

      <div className="field">
        <label>Min |weight|</label>
        <input
          type="number"
          step="0.01"
          value={filter.min_abs_weight}
          onChange={(event) =>
            onFilterChange({
              ...filter,
              min_abs_weight: Number(event.target.value),
            })
          }
        />
      </div>

      <div className="field">
        <label>Max edges</label>
        <input
          type="number"
          value={filter.max_edges}
          onChange={(event) =>
            onFilterChange({
              ...filter,
              max_edges: Number(event.target.value),
            })
          }
        />
      </div>

      {graph ? (
        <div className="stats">
          <div>nodes: {graph.stats.selected_nodes}</div>
          <div>edges: {graph.stats.selected_edges}</div>
          <div>slice: {graph.slice}</div>
        </div>
      ) : null}

      <div className="selection-section">
        <h2>Causal overlay</h2>
        <div className="causal-load-row">
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
          >
            {causalMeta ? "Replace JSON" : "Load causal_eval.json"}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            style={{ display: "none" }}
            onChange={(ev) => {
              const f = ev.target.files?.[0];
              if (f) onLoadCausalFile(f);
              ev.target.value = "";
            }}
          />
        </div>

        {causalMeta ? (
          <>
            <div className="causal-model muted">{causalMeta.model_name.split("/").pop()}</div>
            <div className="causal-stats">
              <span>{causalMeta.num_edges} edges</span>
              <span className="causal-sig">{causalMeta.num_significant} significant</span>
            </div>
            <div className="causal-legend">
              <span className="causal-dot causal-mediated" />
              <span>mediated ({causalMeta.num_mediated})</span>
              <span className="causal-dot causal-parallel" />
              <span>parallel ({causalMeta.num_parallel})</span>
            </div>
            <div className="causal-toggles">
              <label>
                <input
                  type="checkbox"
                  checked={showCausalOverlay}
                  onChange={(ev) => onToggleCausalOverlay(ev.target.checked)}
                />
                Show overlay
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={causalSigOnly}
                  onChange={(ev) => onToggleCausalSigOnly(ev.target.checked)}
                />
                Significant only
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={causalOnlyMode}
                  onChange={(ev) => onToggleCausalOnlyMode(ev.target.checked)}
                />
                Causal only (fade graph)
              </label>
            </div>
            <div className="field" style={{ marginTop: 8, marginBottom: 6 }}>
              <label>Classification</label>
              <select
                value={causalClassFilter}
                onChange={(ev) =>
                  onSetCausalClassFilter(ev.target.value as "all" | "mediated" | "parallel_or_synergy")
                }
              >
                <option value="all">All</option>
                <option value="parallel_or_synergy">Parallel only (backbone)</option>
                <option value="mediated">Mediated only</option>
              </select>
            </div>
            <div className="field" style={{ marginBottom: 0 }}>
              <label>Min necessity</label>
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                value={necessityMin}
                onChange={(ev) => onSetNecessityMin(Number(ev.target.value))}
              />
            </div>
          </>
        ) : (
          <div className="muted">Load a causal_eval.json to overlay causal edges.</div>
        )}
      </div>

      <div className="selection-section">
        <h2>Selected node</h2>
        {selectedNode ? (
          <>
            <div className="selection-row">
              id {selectedNode.id} · L{selectedNode.layer} · T{selectedNode.token}
            </div>
            <div className="selection-row">
              {selectedNode.type}
              {selectedNode.head !== null ? ` (head ${selectedNode.head})` : ""}
            </div>
            <div className="selection-actions">
              <button type="button" onClick={onClearSelection}>
                Clear
              </button>
              <label>
                <input
                  type="checkbox"
                  checked={isolateLocalCircuit}
                  onChange={(event) => onToggleIsolate(event.target.checked)}
                />
                Isolate local circuit
              </label>
            </div>

            <div className="connection-block">
              <h3>Top incoming</h3>
              {topIncoming.length === 0 ? (
                <div className="muted">No incoming edges</div>
              ) : (
                topIncoming.map((item) => (
                  <div key={`in-${item.edge.src}-${item.edge.dst}`} className="conn-row">
                    n{item.edge.src}
                    {item.node
                      ? ` (L${item.node.layer},T${item.node.token},${item.node.type})`
                      : ""}
                    <span>{item.edge.weight.toFixed(3)}</span>
                  </div>
                ))
              )}
            </div>

            <div className="connection-block">
              <h3>Top outgoing</h3>
              {topOutgoing.length === 0 ? (
                <div className="muted">No outgoing edges</div>
              ) : (
                topOutgoing.map((item) => (
                  <div key={`out-${item.edge.src}-${item.edge.dst}`} className="conn-row">
                    n{item.edge.dst}
                    {item.node
                      ? ` (L${item.node.layer},T${item.node.token},${item.node.type})`
                      : ""}
                    <span>{item.edge.weight.toFixed(3)}</span>
                  </div>
                ))
              )}
            </div>
          </>
        ) : (
          <div className="muted">Click a node to inspect its local circuit.</div>
        )}
      </div>
    </aside>
  );
}
