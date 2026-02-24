export type GraphNode = {
  id: number;
  layer: number;
  token: number;
  type: string;
  head: number | null;
};

export type GraphEdge = {
  src: number;
  dst: number;
  weight: number;
};

export type GraphPayload = {
  slice: string;
  num_layers: number;
  num_tokens: number;
  token_labels: string[];
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: {
    selected_nodes: number;
    selected_edges: number;
    graph_nodes: number;
    graph_edges: number;
  };
};

export type ViewerInitPayload = {
  slices: string[];
  default_filter: {
    slice_id: string;
    min_abs_weight: number;
    max_edges: number;
  };
  graph: GraphPayload;
};

export type ViewerMessage =
  | { type: "init"; payload: ViewerInitPayload }
  | { type: "graph_update"; payload: GraphPayload }
  | { type: "error"; message: string }
  | { type: "pong" };

export type GraphFilter = {
  slice_id: string;
  layer_min?: number;
  layer_max?: number;
  token_min?: number;
  token_max?: number;
  min_abs_weight: number;
  max_edges: number;
};

export type CausalNodeRef = {
  layer: number;
  token: number;
  type: string;
  head: number | null;
};

export type CausalEdgeData = {
  edge_id: string;
  src: CausalNodeRef;
  dst: CausalNodeRef;
  I_mean: number;
  I_ci_low: number;
  I_ci_high: number;
  I_p_value: number;
  necessity_mean: number;
  M_mean: number;
  classification: "mediated" | "parallel_or_synergy";
  significant: boolean;
};

export type CausalOverlayMeta = {
  model_name: string;
  num_edges: number;
  num_significant: number;
  num_mediated: number;
  num_parallel: number;
};
