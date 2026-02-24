# Full Top-k Graph Calculation (Toy 6x4) — Exactly Like PIG Code

This walkthrough reproduces the graph-building math from:

- `src/pig/patching.py` (`get_effect_matrix` flattening)
- `src/pig/graph.py` (`_compute_correlation_matrix`, `_apply_direction_constraint`, `_apply_topk_sparsification`)

It is a fully worked numeric example in English.

---

## 1) Toy activation/effect setup

We use **6 examples** and **4 nodes**.

So the effect matrix is `E` with shape `6 x 4`.

| Example | Node 0 | Node 1 | Node 2 | Node 3 |
|---|---:|---:|---:|---:|
| 1 | 0.10 | 0.20 | -0.10 | 0.00 |
| 2 | 0.20 | 0.10 | 0.00 | 0.10 |
| 3 | 0.00 | 0.30 | -0.20 | -0.10 |
| 4 | 0.40 | 0.00 | 0.20 | 0.30 |
| 5 | 0.30 | -0.10 | 0.10 | 0.20 |
| 6 | 0.50 | -0.20 | 0.30 | 0.40 |

Interpretation:

- Each row = one prompt example in one slice.
- Each column = one flattened patch node.

In the real pipeline, each example starts as `effects[layer, token, component]`, then is truncated to common token length and flattened with `reshape(-1)`.

---

## 2) How 3D activations become 4 nodes (toy mapping)

A simple way to get 4 nodes is:

- `num_layers = 1`
- `num_tokens = 2`
- `num_components = 2`

Then `num_nodes = 1 * 2 * 2 = 4`.

Flatten order is row-major (`reshape(-1)`), so node indices are:

- node 0: `(layer=0, token=0, component=0)`
- node 1: `(layer=0, token=0, component=1)`
- node 2: `(layer=0, token=1, component=0)`
- node 3: `(layer=0, token=1, component=1)`

---

## 3) Centering and standardization (exact code behavior)

The code computes `E_c = E - mu`, where `mu` is the column mean.

Column means:

```text
mu = [0.25, 0.05, 0.05, 0.15]
```

Centered matrix `E_c`:

```text
[-0.15,  0.15, -0.15, -0.15]
[-0.05,  0.05, -0.05, -0.05]
[-0.25,  0.25, -0.25, -0.25]
[ 0.15, -0.05,  0.15,  0.15]
[ 0.05, -0.15,  0.05,  0.05]
[ 0.25, -0.25,  0.25,  0.25]
```

Then standard deviation per column:

```text
sigma = [0.170783, 0.170783, 0.170783, 0.170783]
```

(If any `sigma < 1e-8`, code replaces it with `1.0`.)

Normalized matrix `Z = E_c / sigma`:

```text
[-0.878310,  0.878310, -0.878310, -0.878310]
[-0.292770,  0.292770, -0.292770, -0.292770]
[-1.463850,  1.463850, -1.463850, -1.463850]
[ 0.878310, -0.292770,  0.878310,  0.878310]
[ 0.292770, -0.878310,  0.292770,  0.292770]
[ 1.463850, -1.463850,  1.463850,  1.463850]
```

---

## 4) Correlation matrix

The code computes `C = (Z^T @ Z) / N`, with `N = 6`.

Result (`C`):

```text
[ 1.000000, -0.942857,  1.000000,  1.000000]
[-0.942857,  1.000000, -0.942857, -0.942857]
[ 1.000000, -0.942857,  1.000000,  1.000000]
[ 1.000000, -0.942857,  1.000000,  1.000000]
```

One explicit entry (example):

```text
C[0,1] = (1/6) * sum_t Z[t,0] * Z[t,1] = -0.942857
```

---

## 5) Direction constraint (`enforce_direction=True`)

To mimic `src < dst`, assume node order `0 < 1 < 2 < 3`.
Allowed edges are only upper triangle (strict), mask `M`:

```text
[0, 1, 1, 1]
[0, 0, 1, 1]
[0, 0, 0, 1]
[0, 0, 0, 0]
```

Apply mask elementwise: `C_dir = C * M`.

```text
[0,  -0.942857,  1.000000,  1.000000]
[0,   0,        -0.942857, -0.942857]
[0,   0,         0,         1.000000]
[0,   0,         0,         0]
```

---

## 6) Top-k sparsification per source row

Set `k=2`.
For each row `i`, code does:

1. `abs_row = abs(C_dir[i,:])`
2. `top_indices = argsort(abs_row)[-k:]`
3. Keep original signed weights only at those `top_indices`

Important:

- Ranking uses **absolute value**.
- Stored edge keeps **original sign**.
- This is **row-wise** top-k, not global top-k.

### Row-by-row

Row 0:

- values: `[0, -0.942857, 1.000000, 1.000000]`
- abs: `[0, 0.942857, 1.000000, 1.000000]`
- keep top-2: indices `{2,3}`
- kept: `(0->2)=1.000000`, `(0->3)=1.000000`

Row 1:

- values: `[0, 0, -0.942857, -0.942857]`
- abs: `[0, 0, 0.942857, 0.942857]`
- keep top-2: indices `{2,3}`
- kept: `(1->2)=-0.942857`, `(1->3)=-0.942857`

Row 2:

- values: `[0, 0, 0, 1.000000]`
- abs: `[0, 0, 0, 1.000000]`
- top-2 indices are `{2,3}`, but index 2 has weight 0
- kept nonzero: `(2->3)=1.000000`

Row 3:

- values all zero
- contributes no nonzero edges

Sparse matrix after top-k (`W_sparse`):

```text
[0, 0,  1.000000,  1.000000]
[0, 0, -0.942857, -0.942857]
[0, 0,  0,         1.000000]
[0, 0,  0,         0]
```

---

## 7) Final edge list (`min_weight = 0`, `i != j`)

Code emits edge if `abs(weight) > min_weight`.
So final edges are:

1. `(src=0, dst=2, weight=+1.000000)`
2. `(src=0, dst=3, weight=+1.000000)`
3. `(src=1, dst=2, weight=-0.942857)`
4. `(src=1, dst=3, weight=-0.942857)`
5. `(src=2, dst=3, weight=+1.000000)`

---

## 8) Notes that match implementation details

- If `k >= n_nodes`, code keeps all columns in each row.
- Ties are not specially resolved; `argsort` order determines which tied entries survive.
- With `abs_correlation_topk`, the builder first replaces `C` by `|C|` before direction/top-k.
- If `enforce_direction=False`, no mask is applied before top-k.

---

## 9) Minimal pseudocode (same algorithm)

```text
E = get_effect_matrix(slice)  # shape [N, D]
centered = E - mean(E, axis=0)
std = std(E, axis=0)
std = where(std < 1e-8, 1.0, std)
Z = centered / std
C = (Z^T @ Z) / N

if enforce_direction:
  C = C * direction_mask(nodes)  # src < dst

W_sparse = zeros_like(C)
for i in rows(C):
  idx = argsort(abs(C[i,:]))[-k:]
  W_sparse[i, idx] = C[i, idx]

edges = [(i,j,W_sparse[i,j]) for i,j if i!=j and abs(W_sparse[i,j]) > min_weight]
```

---

## 10) `S` in `math.md`: set of patched nodes (for causal levels)

In `math.md`, `S` is **not** the sparse matrix. It is the set of nodes you patch in the forward pass:

- `S = {u}` means patch only node `u`
- `S = {v}` means patch only node `v`
- `S = {u, v}` means patch both nodes together

Fractional restoration is:

```text
R(S) = (O_patch(S) - O_base) / (O_cln - O_base + eps)
```

with `eps = 1e-6`.

---

## 11) Full numeric example for Levels A, B, C

Use one candidate edge from the graph, e.g. `u = 1 -> v = 3`.

### 11.1 Scores for R(S)

Assume these observed scores for one example:

```text
O_cln  = -1.20
O_base = -2.00
O_patch({u})   = -1.70
O_patch({v})   = -1.50
O_patch({u,v}) = -1.48
O_patch({u} + clamp(v->base)) = -1.86
eps = 1e-6
```

Denominator shared by all restoration values:

```text
den = O_cln - O_base + eps
    = (-1.20) - (-2.00) + 1e-6
    = 0.800001
```

Now compute each `R(S)`:

```text
R_u = R({u})
    = (O_patch({u}) - O_base) / den
    = [(-1.70) - (-2.00)] / 0.800001
    = 0.30 / 0.800001
    = 0.3749995

R_v = R({v})
    = [(-1.50) - (-2.00)] / 0.800001
    = 0.50 / 0.800001
    = 0.6249992

R_uv = R({u,v})
     = [(-1.48) - (-2.00)] / 0.800001
     = 0.52 / 0.800001
     = 0.6499992

R_u_clamp_v = R({u} + clamp(v->base))
            = [(-1.86) - (-2.00)] / 0.800001
            = 0.14 / 0.800001
            = 0.1749998
```

### 11.2 Level A — activation influence `I(u -> v)`

Assume destination-node activations at `v` are:

```text
a_v_base     = [0.10, -0.20, 0.05, 0.00]
a_v_cln      = [0.50,  0.10, 0.25, 0.20]
a_v_patch(u) = [0.30,  0.00, 0.15, 0.12]
```

Numerator norm:

```text
a_v_patch(u) - a_v_base = [0.20, 0.20, 0.10, 0.12]
||...||_2 = sqrt(0.20^2 + 0.20^2 + 0.10^2 + 0.12^2)
        = sqrt(0.1044)
        = 0.323110
```

Denominator norm:

```text
a_v_cln - a_v_base = [0.40, 0.30, 0.20, 0.20]
||...||_2 = sqrt(0.40^2 + 0.30^2 + 0.20^2 + 0.20^2)
        = sqrt(0.33)
        = 0.574456
```

Final Level A value:

```text
I(u -> v) = 0.323110 / (0.574456 + 1e-6)
          = 0.562469
```

### 11.3 Level B — mediation `M(u -> v)`

```text
M(u -> v) = R_uv - R_v
          = 0.6499992 - 0.6249992
          = 0.0250000
```

Interpretation with threshold `|M| > 0.05`:

- `|0.025| < 0.05` -> close to `mediated` behavior (u adds little on top of v).

### 11.4 Level C — necessity `nec(u -> v)`

```text
nec(u -> v) = R_u - R_u_clamp_v
            = 0.3749995 - 0.1749998
            = 0.1999997
```

Interpretation:

- positive and fairly large -> blocking `v` removes much of `u`'s benefit, so `v` is necessary for this path.

---

## 12) Compact summary table (this one example)

| Metric | Value |
|---|---:|
| `R_u` | 0.3749995 |
| `R_v` | 0.6249992 |
| `R_uv` | 0.6499992 |
| `R_u_clamp_v` | 0.1749998 |
| `I(u -> v)` | 0.562469 |
| `M(u -> v)` | 0.0250000 |
| `nec(u -> v)` | 0.1999997 |
