## distilgpt2 (n=100)

| Representation | Accuracy | Std | Dim |
|---|---|---|---|
| Fixed layout (signed) | 1.0000 | 0.0000 | 3,540 |
| Hashed fixed (signed) | 1.0000 | 0.0000 | 1,024 |
| Patch-effect (raw) | 1.0000 | 0.0000 | 46,080 |
| Fixed layout (weighted) | 0.9948 | 0.0074 | 3,540 |
| Hashed fixed (weighted) | 0.9948 | 0.0074 | 1,024 |
| Fixed layout (binary) | 0.9844 | 0.0128 | 3,540 |
| Top-k node | 0.9750 | 0.0354 | 10 |
| Coarse count | 0.9115 | 0.0531 | 105 |
| WL bootstrap | 0.6875 | 0.1295 | --- |
| Graphlet shape | 0.6719 | 0.1326 | 38 |
| GNN encoder | 0.6094 | 0.1090 | --- |
| Spectral shape | 0.5521 | 0.0147 | 96 |
| Null: weight-shuffle | 0.5234 | 0.0084 | --- |
| Null: edge-shuffle | 0.4953 | 0.0038 | --- |

## gpt2 (n=100)

| Representation | Accuracy | Std | Dim |
|---|---|---|---|
| Fixed layout (signed) | 0.9896 | 0.0147 | 14,280 |
| Top-k node | 0.9667 | 0.0118 | 10 |
| Patch-effect (raw) | 0.9583 | 0.0118 | 92,160 |
| Hashed fixed (signed) | 0.9375 | 0.0338 | 1,024 |
| Fixed layout (binary) | 0.9323 | 0.0737 | 14,280 |
| Coarse count | 0.9271 | 0.0074 | 105 |
| Hashed fixed (weighted) | 0.9115 | 0.0321 | 1,024 |
| Fixed layout (weighted) | 0.8698 | 0.1522 | 14,280 |
| WL bootstrap | 0.7656 | 0.1673 | --- |
| Graphlet shape | 0.6354 | 0.0849 | 38 |
| Spectral shape | 0.6042 | 0.0516 | 96 |
| GNN encoder | 0.5938 | 0.0338 | --- |
| Null: weight-shuffle | 0.5021 | 0.0029 | --- |
| Null: edge-shuffle | 0.5000 | 0.0000 | --- |

## gpt2 (n=500)

| Representation | Accuracy | Std | Dim |
|---|---|---|---|
| Fixed layout (binary) | 1.0000 | 0.0000 | 14,280 |
| Fixed layout (signed) | 1.0000 | 0.0000 | 14,280 |
| Fixed layout (weighted) | 1.0000 | 0.0000 | 14,280 |
| Hashed fixed (signed) | 1.0000 | 0.0000 | 1,024 |
| WL bootstrap | 1.0000 | 0.0000 | --- |
| Graphlet shape | 0.9948 | 0.0074 | 38 |
| Coarse count | 0.9740 | 0.0368 | 105 |
| Hashed fixed (weighted) | 0.9583 | 0.0368 | 1,024 |
| GNN encoder | 0.8333 | 0.1213 | --- |
| Null: edge-shuffle | nan | nan | --- |
| Null: weight-shuffle | nan | nan | --- |
| Spectral shape | 0.9115 | 0.0368 | 96 |
