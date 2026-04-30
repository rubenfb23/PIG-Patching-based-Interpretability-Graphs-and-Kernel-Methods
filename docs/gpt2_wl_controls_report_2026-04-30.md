# Informe GPT-2: WL, fixed-layout, GNN y decision para paper

Fecha: 2026-04-30
Modelo: `gpt2`
Configuracion base: `node_types=res`, `k=5`, corrupciones `name_swap` y `abba`, seeds `7,42,123`.

Nota sobre `n`: en este informe, `n` significa numero de prompt pairs por corrupcion y por seed. Como usamos dos corrupciones (`name_swap` y `abba`), `n=20` son `20 + 20 = 40` prompt pairs por seed, y `n=100` son `100 + 100 = 200` prompt pairs por seed.

## 1. Resumen

Se reviso la parte de representacion de grafos de la pipeline PIG. Antes se usaba principalmente `WL + SVM` sobre grafos auxiliares por ejemplo. Ahora se compararon cuatro familias:

- `WL per-example`: baseline historico.
- `WL bootstrap-slice`: WL sobre grafos tipo slice, mas cercano al objeto canonico PIG.
- `Fixed-layout`: vectorizacion directa de slots `u -> v` del transformer.
- `GNN`: message-passing pequeno sobre los mismos grafos bootstrap.

Conclusion para paper:

- El `1.000` inicial de fixed-layout con `n=20` era demasiado optimista como claim principal.
- Con `n=100` y split disjunto de ejemplos base, fixed-layout sigue siendo fuerte.
- La mejor variante es `fixed-layout sign linear`: `0.9896` accuracy media.
- `edge_shuffle` y `weight_shuffle` caen a azar: `0.5000` y `0.5021`.
- La GNN no mejora WL/fixed-layout y no conviene como resultado principal.

Recomendacion:

- Incluir en paper `WL bootstrap linear` como baseline clasico.
- Incluir `fixed-layout sign linear` como baseline fuerte.
- Incluir `fixed-layout binary linear` como ablation topologica.
- No vender los pesos continuos como perfectos; la senal mas robusta esta en topologia/signo.

## 2. Que cambia en la pipeline

Antes:

```text
Patch effects
    |
    v
Per-example graphs
    |
    v
WL features
    |
    v
SVM
```

Ahora:

```text
Patch effects
    |
    +--> Canonical slice graph ---------------------> causal eval
    |
    +--> Per-example graphs ------------------------> WL legacy
    |
    +--> Bootstrap slice graphs ----+---------------> WL
                                    |
                                    +---------------> fixed-layout
                                    |
                                    +---------------> GNN
                                    |
                                    +---------------> null controls
```

El objetivo no es cambiar el causal eval. El objetivo es decidir que representacion auxiliar de grafos merece aparecer en el paper.

## 3. Paso a paso con formulas

### 3.1 Patch effects

Para cada ejemplo `i` y nodo interno `u`, se parchea la activacion corrupta con la activacion limpia y se mide el cambio en el observable objetivo:

```math
E_u^{(i)}
=
O(\mathrm{patch}(x_{\mathrm{crp}}^{(i)}; u \leftarrow x_{\mathrm{cln}}^{(i)}))
-
O(x_{\mathrm{crp}}^{(i)})
```

Interpretacion:

- `E_u^{(i)} > 0`: parchear `u` ayuda a recuperar el comportamiento limpio.
- `E_u^{(i)} = 0`: parchear `u` casi no cambia el objetivo.
- `E_u^{(i)} < 0`: parchear `u` empeora el objetivo.

### 3.2 Grafo por slice

Para un slice `s`, se apilan efectos:

```math
X_s[i,u] = E_u^{(i)}
```

La arista entre nodos se calcula por correlacion entre perfiles de efectos:

```math
w_{uv}^{(s)}
=
\mathrm{corr}
\left(
    (E_u^{(i)})_{i \in s},
    (E_v^{(i)})_{i \in s}
\right)
```

Despues se impone direccion forward:

```math
u \rightarrow v
\quad \mathrm{solo\ si} \quad
u < v
```

Y se aplica top-k:

```math
E_s
=
\left\{
    (u,v) :
    v \in \mathrm{TopK}_k(|w_{uv}^{(s)}|)
\right\}
```

### 3.3 Bootstrap slice graphs

Como hay pocos slices, se generan varios grafos tipo slice re-muestreando ejemplos dentro del slice:

```math
B_{s,b}
\sim
\mathrm{sample}(D_s)
```

Luego se construye:

```math
G_{s,b}
=
\mathrm{GraphBuilder}(B_{s,b})
```

Esto da mas muestras para clasificadores, pero los bootstraps no son datos totalmente independientes.

## 4. Representaciones probadas

### 4.1 WL

WL convierte un grafo en histogramas de patrones locales.

Etiqueta inicial:

```math
\ell_v^{(0)}
=
(\mathrm{layer}(v), \mathrm{token}(v), \mathrm{type}(v), \mathrm{head}(v))
```

Refinamiento:

```math
\ell_v^{(h)}
=
\mathrm{hash}
\left(
    \ell_v^{(h-1)},
    \mathrm{sort}
    \left\{
        (\ell_z^{(h-1)}, b(w_{vz})) : (v,z) \in E
    \right\}
\right)
```

Vector final:

```math
\phi_a(G)
=
\sum_{h=0}^{H}
\sum_{v \in V}
\mathbf{1}[\ell_v^{(h)} = a]
```

Pros:

- Clasico, simple y defendible.
- No entrena una red neuronal.
- Adecuado como baseline.

Contras:

- Puede perder informacion posicional fina.
- Discretiza pesos.
- Si el layout de nodos es fijo, puede ser menos directo que usar slots `u -> v`.

### 4.2 Fixed-layout

Como todos los grafos tienen el mismo layout de nodos, cada posible arista `u -> v` puede ser una feature fija.

Version weighted:

```math
\phi_{uv}(G) = w_{uv}
```

Version binaria:

```math
\phi_{uv}(G)
=
\mathbf{1}[(u,v) \in E]
```

Version signo:

```math
\phi_{uv}(G)
=
\mathrm{sign}(w_{uv})
```

Pros:

- Usa directamente que `layer/token/component` significa lo mismo en todos los grafos.
- Muy interpretable por edge slot.
- Permite separar topologia, signo y peso continuo.

Contras:

- Alta dimensionalidad.
- Solo aplica bien si el layout de nodos es fijo.
- Puede aprender patron topologico de la tarea, no necesariamente causalidad fina.

### 4.3 GNN

Se implemento una GNN pequena sin PyTorch Geometric.

Feature inicial de nodo:

```math
x_v
=
\left[
\frac{\mathrm{layer}(v)}{L-1},
\frac{\mathrm{token}(v)}{T-1},
\mathrm{head}(v),
1,
\mathrm{onehot}(\mathrm{type}(v))
\right]
```

Mensaje:

```math
m_v^{(t)}
=
\sum_u
\widetilde{A}_{v,u}
h_u^{(t)}
```

Actualizacion:

```math
z_v^{(t)}
=
W_s^{(t)} \cdot h_v^{(t)}
+
W_m^{(t)} \cdot m_v^{(t)}
+
b^{(t)}
```

```math
h_v^{(t+1)}
=
\mathrm{ReLU}(z_v^{(t)})
```

Donde `W_s` aplica la transformacion del propio nodo y `W_m` aplica la transformacion del mensaje recibido.

Pooling:

```math
g(G)
=
\left[
\frac{1}{|V|}\sum_{v \in V} h_v^{(T)}
;
\max_{v \in V} h_v^{(T)}
\right]
```

Clasificador:

```math
p(y \mid G)
=
\mathrm{softmax}
\left(
\mathrm{MLP}(g(G))
\right)
```

Pros:

- Puede aprender representaciones no diseñadas a mano.
- Usa pesos continuos y atributos de nodo.
- Es flexible si aumentamos datos/slices.

Contras:

- Dataset pequeno para entrenar parametros.
- Mas estocastico y menos interpretable.
- En esta prueba no supera a fixed-layout ni a WL bootstrap.

## 5. Controles nulos

Se usaron tres controles:

Label permutation:

```math
y_i \leftarrow y_{\pi(i)}
```

Edge shuffle:

```math
(u,v,w_{uv})
\leftarrow
(u',v',w_{uv})
```

Weight shuffle:

```math
(u,v,w_{uv})
\leftarrow
(u,v,w_{\pi(uv)})
```

Lectura:

- Si label permutation sigue alto, hay leakage o sobreajuste.
- Si edge shuffle sigue alto, la topologia no importa.
- Si weight shuffle sigue alto, los pesos exactos no importan o la topologia domina.

## 6. Experimento A: `n=20`, bootstrap-CV inicial

Este experimento queda solo como preliminar. Sirve para explicar por que se sospecho del `1.000` de fixed-layout y por que se diseno el test `n=100` con split disjunto.

Configuracion: `20` ejemplos por corrupcion, `12` grafos bootstrap por slice y CV estratificado sobre grafos bootstrap.

Resultados medios:

| Representacion | Linear | Lectura |
|---|---:|---|
| WL per-example | 0.6667 | Baseline historico |
| WL bootstrap-slice | 0.8000 | Mejor baseline WL |
| Fixed-layout weighted | 1.0000 | Demasiado optimista |

Lectura: no se usa para decidir el paper. El CV sobre bootstraps puede mezclar train/test derivados de los mismos ejemplos base. La decision final se toma con `n=100` y split disjunto.

## 7. Experimento B: GNN sobre bootstrap graphs

Configuracion:

- Mismos grafos bootstrap `n=20`.
- GNN: `hidden_dim=32`, `num_layers=2`, `epochs=150`.
- 5-fold CV estratificado.
- Null controls con 5 repeticiones.

Resultados por seed:

| Seed | Accuracy | CV std | Label delta | Edge delta | Weight delta |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.6500 | 0.2793 | 0.1800 | 0.1220 | 0.0460 |
| 42 | 0.9100 | 0.1114 | 0.3760 | 0.1240 | 0.1740 |
| 123 | 0.7900 | 0.1281 | 0.2200 | 0.1280 | 0.0820 |

Media:

| Metodo | Accuracy | Label delta | Edge delta | Weight delta |
|---|---:|---:|---:|---:|
| GNN bootstrap-slice | 0.7833 | 0.2587 | 0.1247 | 0.1007 |
| WL bootstrap-slice | 0.8000 | 0.3913 | 0.3767 | 0.3667 |
| Fixed-layout weighted | 1.0000 | 0.5240 | 0.5133 | 0.0653 |

Interpretacion:

- La GNN aprende algo, pero no mejora WL bootstrap.
- Sus deltas contra controles son debiles.
- No conviene incluirla como resultado principal del paper.
- Puede quedar como baseline negativo o ablation en apendice.

## 8. Experimento C: `n=100`, split disjunto para decision de paper

Este es el experimento mas importante.

Problema corregido:

- Antes: bootstrap-CV podia mezclar train/test derivados de los mismos prompts.
- Ahora: se separan ejemplos base antes de construir bootstraps.

Split:

```math
D_s
=
D_s^{\mathrm{train}}
\cup
D_s^{\mathrm{test}}
```

```math
D_s^{\mathrm{train}}
\cap
D_s^{\mathrm{test}}
=
\emptyset
```

Configuracion:

- `100` ejemplos por corrupcion.
- `80` train y `20` test por corrupcion.
- `32` grafos bootstrap por slice en train.
- `32` grafos bootstrap por slice en test.
- `64` grafos train y `64` grafos test por seed.
- Seeds `7,42,123`.

Flujo:

```text
100 ejemplos por corrupcion
    |
    v
split disjunto por ejemplos base
    |
    +--> train examples --> bootstrap train graphs --> fit SVM
    |
    +--> test examples  --> bootstrap test graphs  --> evaluate
```

Resultados por seed:

| Seed | WL linear | WL RBF | Weighted linear | Binary linear | Sign linear | Edge-shuffle test | Weight-shuffle test |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.7969 | 0.8281 | 0.9531 | 0.9844 | 1.0000 | 0.5000 | 0.5000 |
| 42 | 0.5469 | 0.5156 | 0.6562 | 0.8281 | 0.9688 | 0.5000 | 0.5063 |
| 123 | 0.9531 | 0.5000 | 1.0000 | 0.9844 | 1.0000 | 0.5000 | 0.5000 |

Medias:

| Representacion | Accuracy | Std | Lectura |
|---|---:|---:|---|
| WL bootstrap linear | 0.7656 | 0.1673 | Baseline clasico razonable |
| WL bootstrap RBF | 0.6146 | 0.1511 | Inestable |
| Fixed weighted linear | 0.8698 | 0.1522 | Fuerte, pero variable |
| Fixed weighted RBF | 0.5000 | 0.0000 | No util |
| Fixed binary topology linear | 0.9323 | 0.0737 | Topologia separa bien |
| Fixed sign topology linear | 0.9896 | 0.0147 | Mejor y mas estable |
| Edge-shuffle test | 0.5000 | 0.0000 | Destruir topologia lleva a azar |
| Weight-shuffle test | 0.5021 | 0.0029 | Destruir asignacion de pesos lleva a azar |

Visual:

```text
Example-disjoint n=100, linear
WL bootstrap        ########-- 0.766
Fixed weighted      #########- 0.870
Fixed binary        #########- 0.932
Fixed sign          ########## 0.990

Test-time controls sobre fixed weighted
Edge shuffle        #####----- 0.500
Weight shuffle      #####----- 0.502
```

Interpretacion:

- El resultado fuerte sobrevive al split disjunto.
- La senal mas estable no es el peso continuo, sino el signo del edge slot.
- La mascara topologica binaria tambien es muy fuerte.
- RBF no aporta en fixed-layout.
- Los controles de test-time caen a azar, asi que la senal no sobrevive a destruir topologia o asignacion de pesos.

## 9. Pros y contras finales

| Metodo | Pros | Contras | Decision |
|---|---|---|---|
| WL per-example | Historico, barato, reproduce baseline | No es el objeto canonico PIG | Solo referencia |
| WL bootstrap-slice | Cercano al grafo por slice, defendible | Variable, RBF debil | Incluir como baseline |
| Fixed weighted | Directo, fuerte, interpretable por edge slot | Menos estable, high-dimensional | Incluir si hay espacio |
| Fixed binary | Aisla topologia, fuerte | No usa pesos | Incluir como ablation |
| Fixed sign | Mejor accuracy y estabilidad | Depende de layout fijo | Incluir como resultado fuerte |
| GNN | Flexible, aprende features | No mejora, menos interpretable | No incluir como main |

## 10. Recomendacion para paper

Tabla principal sugerida:

| Metodo | Accuracy media | Rol |
|---|---:|---|
| WL bootstrap linear | 0.7656 | Baseline clasico |
| Fixed weighted linear | 0.8698 | Edge weights continuos |
| Fixed binary topology linear | 0.9323 | Ablation topologica |
| Fixed sign topology linear | 0.9896 | Mejor resultado |
| Edge-shuffle test | 0.5000 | Control negativo |
| Weight-shuffle test | 0.5021 | Control negativo |

Claim defendible:

```text
Under an example-disjoint bootstrap evaluation with 100 prompts per corruption,
fixed-layout signed edge features achieve 0.9896 mean accuracy across seeds,
while edge and weight shuffling reduce performance to chance.
```

Claim que no conviene usar:

```text
Fixed-layout weighted gets 1.000 accuracy, therefore weights perfectly encode the mechanism.
```

Por que no:

- Ese `1.000` venia de `n=20` y bootstrap-CV.
- La version `n=100` muestra que topologia/signo son mas estables que pesos continuos.
- La conclusion correcta es sobre patron topologico/signado del grafo, no sobre perfeccion de pesos.

## 11. Comandos clave

GNN:

```bash
HF_HOME=/home/ruben/PIG/.cache/huggingface \
CUDA_VISIBLE_DEVICES=2 \
uv run python scripts/run_gnn_representation_study.py \
  --model-name gpt2 \
  --device cuda \
  --gnn-device cuda \
  --seeds 7,42,123 \
  --k-grid 5 \
  --num-examples-grid 20 \
  --node-types-grid res \
  --output-dir outputs/gnn_representation/gpt2_gnn_controls_20260430_res_k5_n20 \
  --cache-root .cache/classical_publication/gpt2_wl_controls_20260430_res_k5_n20 \
  --gnn-epochs 150 \
  --null-repeats 5
```

Decision para paper `n=100`:

```bash
HF_HOME=/home/ruben/PIG/.cache/huggingface \
CUDA_VISIBLE_DEVICES=2 \
uv run python scripts/run_paper_decision_study.py \
  --model-name gpt2 \
  --device cuda \
  --seeds 7,42,123 \
  --k-grid 5 \
  --num-examples-grid 100 \
  --node-types-grid res \
  --graphs-per-slice 32 \
  --null-repeats 10 \
  --output-dir outputs/paper_decision/gpt2_n100_res_k5_20260430 \
  --cache-root .cache/paper_decision/gpt2_n100_res_k5_20260430
```
