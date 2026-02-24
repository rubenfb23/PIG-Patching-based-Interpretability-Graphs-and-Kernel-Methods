# David Conversation Notes

## What is `O` in this project?

`O` is a single scalar score taken from the model output:

- It is the **target-token logit** at the **last sequence position**.
- In code (`src/pig/model.py`), this is computed from `logits[0, -1, :]` and then selecting the target token id.

So in this repo:

- `O_cln`: target-token last-position logit on the clean prompt.
- `O_base`: target-token last-position logit on the corrupted prompt.
- `O_patch(S)`: same logit after patching a set of internal nodes `S`.

Important: `O` is **not** an internal node activation. It is a final readout scalar derived from model logits.

---

## What is the effect matrix?

The effect matrix is a table of patching effects:

- Rows = examples (prompt pairs).
- Columns = patch nodes (flattened layer/token/component positions).
- Entry `(t, i)` measures the causal effect of patching node `i` for example `t`.

Typical form:

```text
effect[t, i] = O_patch(node i, example t) - O_base(example t)
```

Interpretation:

- Positive value: patching node `i` helps recover correct behavior.
- Negative value: patching node `i` hurts.

---

## Relation to a real LLM run

For each example (prompt pair), the pipeline does:

1. Run corrupted prompt -> compute `O_base`.
2. Run clean prompt and cache internal activations.
3. For each node `i`, rerun corrupted prompt while replacing that node activation with the clean cached activation -> compute `O_patch(node i)`.
4. Subtract `O_base` to get effect for that node.

That produces one row of the effect matrix.

So the toy `6 x 4` matrix is just a small version of what happens in a real LLM (in practice, many more examples and many more nodes).

---

## Extra notation clarification (`S`)

`S` can mean two different things depending on context:

- In causal formulas (`R(S)`, Levels A/B/C): `S` = set of patched nodes.
- In graph-building sections, avoid using `S` for sparse matrix to prevent confusion (use `W_sparse` instead).

---

## Level formulas used in `math.md`

```text
R(S) = (O_patch(S) - O_base) / (O_cln - O_base + eps)
I(u->v) = ||a_v_patch(u) - a_v_base||_2 / (||a_v_cln - a_v_base||_2 + eps)
M(u->v) = R({u,v}) - R({v})
nec(u->v) = R({u}) - R({u} + clamp(v->base))
```

These are the main quantities used to evaluate candidate edges after top-k proposal.
