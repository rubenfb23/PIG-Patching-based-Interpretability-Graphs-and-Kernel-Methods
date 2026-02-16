import { GraphEdge, GraphFilter, GraphNode, GraphPayload } from "../types";

type Props = {
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
};

const toNumber = (value: string): number | undefined => {
  if (value.trim() === "") {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
};

export function ControlsPanel({
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
}: Props) {
  return (
    <aside className="panel">
      <h1>PIG 4D Viewer</h1>
      <p>Explore slice/layer/token/threshold in real time.</p>

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
