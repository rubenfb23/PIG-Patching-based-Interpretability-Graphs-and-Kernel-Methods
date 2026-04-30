# Informe: WL, bootstrap slice graphs y fixed-layout en GPT-2

Fecha: 2026-04-30  
Configuracion principal: `gpt2`, `node_types=res`, `k=5`, `num_examples_per_corruption=20`, `seeds={7,42,123}`.

## Resumen ejecutivo

La etapa clasica anterior usaba `WL + SVM` sobre grafos por ejemplo como evidencia auxiliar. Eso se mantiene para compatibilidad, pero ahora la pipeline compara esa senal contra dos alternativas:

- `WL bootstrap-slice`: grafos de correlacion por slice reconstruidos desde re-muestreos bootstrap.
- `fixed-layout bootstrap`: vectorizacion directa de pesos `src -> dst` sobre el layout fijo del transformer.

Resultado principal: en GPT-2, el baseline legacy `WL per-example` reproduce exactamente los resultados anteriores, pero `fixed-layout bootstrap` clasifica mucho mejor. Esto sugiere que, para esta tarea, la identidad fija de los nodos `(layer, token, component)` contiene una senal fuerte que WL no explota tan directamente.

## Que cambio en la pipeline

### Antes

```text
Patch effects
    |
    v
Per-example similarity graphs
    |
    v
WL features
    |
    v
Linear/RBF SVM
```

Problema: esos grafos por ejemplo eran utiles para tener suficientes muestras, pero no son el objeto canonico del metodo. El objeto canonico de PIG es el grafo de correlacion por slice.

### Ahora

```text
Patch effects
    |
    +--> Canonical slice graph ---------------------> causal eval
    |
    +--> Per-example graphs ------------------------> WL + SVM legacy
    |
    +--> Bootstrap slice graphs ----+---------------> WL + SVM
                                    |
                                    +---------------> fixed-layout + SVM
                                    |
                                    +---------------> null controls
```

La evaluacion causal no se cambio. Solo se ampliaron las comparaciones auxiliares de representacion.

## Explicacion desde cero

### 0. Que mide PIG antes de construir grafos

PIG empieza con pares de prompts:

- `x_cln`: prompt limpio, donde el modelo deberia comportarse bien.
- `x_crp`: prompt corrupto, donde cambiamos algo para romper o alterar ese comportamiento.
- `y_star`: token/objetivo que queremos medir.

Para cada nodo interno del transformer `u` se hace activation patching: se ejecuta el prompt corrupto, pero en el nodo `u` se reemplaza la activacion corrupta por la activacion limpia. Luego se mide cuanto mejora el score del comportamiento objetivo.

La formula es:

```math
E_u^{(i)}
=
O(\operatorname{patch}(x_{\mathrm{crp}}^{(i)}; u \leftarrow x_{\mathrm{cln}}^{(i)}))
-
O(x_{\mathrm{crp}}^{(i)})
```

Donde:

- `i` es el ejemplo.
- `u` es un nodo interno, por ejemplo `(layer, token, component)`.
- `O(x)` es el observable que mide el comportamiento objetivo.
- `E_u^{(i)}` es el efecto causal individual de parchear el nodo `u` en el ejemplo `i`.

Intuicion:

```text
E_u alto y positivo  -> ese nodo ayuda a recuperar el comportamiento limpio
E_u cercano a cero   -> parchear ese nodo casi no cambia nada
E_u negativo         -> parchear ese nodo empeora el objetivo
```

Esto es la materia prima. Todo lo demas intenta responder: como convertimos muchos efectos `E_u` en una representacion comparable entre corrupciones?

### 1. Antes: WL sobre grafos por ejemplo

Antes teniamos esta ruta auxiliar:

```text
Efectos de patching por ejemplo
    |
    v
Grafo por ejemplo
    |
    v
WL features
    |
    v
SVM linear/RBF
```

El grafo por ejemplo no es el objeto principal de PIG. Se construye para tener muchas muestras para clasificadores. Para un ejemplo `i`, cada nodo tiene un efecto `E_u^{(i)}`. La arista entre dos nodos se define por similitud de magnitud:

```math
w_{uv}^{(i)}
=
\frac{1}{1 + |E_u^{(i)} - E_v^{(i)}|}
```

Si dos nodos tienen efectos parecidos, el peso se acerca a `1`. Si tienen efectos muy distintos, el peso baja.

Despues se aplica top-k: para cada nodo `u`, se conservan solo las `k` aristas salientes mas fuertes.

Que aporta:

- Da suficientes muestras para entrenar SVM.
- Permite comprobar si los grafos contienen senal para distinguir slices.
- Sirve como baseline historico.

Pros:

- Es simple y barato.
- Ya estaba implementado y validado.
- Reproduce resultados anteriores exactamente.

Contras:

- No es el grafo canonico del metodo.
- La formula mide similitud de efectos dentro de un ejemplo, no co-variacion entre ejemplos.
- Puede capturar artefactos de magnitud mas que estructura causal estable.

Resultado GPT-2:

```text
WL per-example linear = 0.6667
WL per-example RBF    = 0.6250
```

Lectura: hay senal real, pero moderada.

### 2. WL: que hace matematicamente

WL significa Weisfeiler-Lehman. Es una forma clasica de convertir un grafo en un vector de features. La idea es etiquetar cada nodo por su identidad inicial y luego refinar esa etiqueta mirando sus vecinos.

Etiqueta inicial:

```math
\ell_v^{(0)}
=
(\operatorname{layer}(v), \operatorname{token}(v), \operatorname{type}(v), \operatorname{head}(v))
```

Refinamiento WL:

```math
\ell_v^{(h)}
=
\operatorname{hash}
\left(
    \ell_v^{(h-1)},
    \operatorname{sort}
    \left\{
        (\ell_z^{(h-1)}, b(w_{vz})) : (v,z) \in E
    \right\}
\right)
```

Donde:

- `h` es la profundidad WL.
- `z` son vecinos salientes de `v`.
- `w_vz` es el peso de la arista.
- `b(w_vz)` es una discretizacion del peso.
- `hash` produce una nueva etiqueta compacta.

El vector final cuenta cuantas veces aparece cada etiqueta:

```math
\phi_a(G)
=
\sum_{h=0}^{H}
\sum_{v \in V}
\mathbf{1}[\ell_v^{(h)} = a]
```

Intuicion:

```text
WL no guarda la matriz completa del grafo.
WL cuenta patrones locales de vecindario.
Dos grafos son parecidos si tienen histogramas parecidos de patrones locales.
```

Que aporta:

- Da un embedding fijo para grafos con estructura variable.
- Captura patrones locales de conectividad.
- Es interpretable como conteo de subestructuras.

Pros:

- Baseline clasico y defendible.
- Funciona con SVM sin entrenar una red neuronal.
- Menos propenso a sobreajuste que una GNN en datasets pequenos.

Contras:

- Puede perder informacion posicional fina.
- Depende de como se discreticen los pesos.
- Si todos los grafos tienen el mismo layout fijo, WL puede ser menos directo que usar los edge slots reales.

### 3. Nuevo: bootstrap slice graphs

El objeto canonico de PIG es el grafo por slice, no el grafo por ejemplo. Un slice es una familia de ejemplos, por ejemplo `name_swap` o `abba`.

Para un slice `s`, se construye una matriz de efectos:

```math
X_s[i,u] = E_u^{(i)}
```

El peso entre nodos se define por correlacion entre perfiles de efectos:

```math
w_{uv}^{(s)}
=
\operatorname{corr}
\left(
    (E_u^{(i)})_{i \in s},
    (E_v^{(i)})_{i \in s}
\right)
```

Luego se impone direccion forward:

```math
u \rightarrow v
\quad \text{solo si} \quad
u < v
```

Y se aplica top-k:

```math
E_s
=
\left\{
    (u,v) :
    v \in \operatorname{TopK}_v(|w_{uv}^{(s)}|)
\right\}
```

Problema: con un grafo por slice solo tenemos muy pocas muestras para clasificar.

Solucion nueva: bootstrap. Para cada slice, re-muestreamos ejemplos con reemplazo y construimos muchos grafos slice-like:

```math
B_{s,b}
\sim
\operatorname{Bootstrap}(\{i : i \in s\})
```

```math
G_{s,b}
=
\operatorname{GraphBuilder}(B_{s,b})
```

Donde `b` es el indice del bootstrap.

Visualmente:

```text
Slice name_swap: 20 ejemplos
    |
    +--> bootstrap 1 -> grafo name_swap #1
    +--> bootstrap 2 -> grafo name_swap #2
    +--> ...

Slice abba: 20 ejemplos
    |
    +--> bootstrap 1 -> grafo abba #1
    +--> bootstrap 2 -> grafo abba #2
    +--> ...
```

Que aporta:

- Mantiene la semantica del grafo canonico por slice.
- Da mas muestras para entrenar clasificadores.
- Permite comparar WL sobre grafos mas cercanos al metodo principal.

Pros:

- Mas metodologicamente correcto que per-example graphs.
- Mejor delta contra null controls.
- Linear SVM mejora claramente en GPT-2.

Contras:

- Las muestras bootstrap no son completamente independientes.
- RBF no funciono bien en esta configuracion.
- Hay que reportarlo como evidencia auxiliar, no como prueba causal principal.

Resultado GPT-2:

```text
WL bootstrap-slice linear = 0.8000
WL bootstrap-slice RBF    = 0.4867
```

Lectura: el linear mejora; RBF cae a azar, asi que la geometria no lineal no parece fiable aqui.

### 4. Nuevo: fixed-layout edge features

Los grafos PIG tienen una propiedad importante: todos comparten el mismo significado de nodo. El nodo `(layer=3, token=5, component=res)` significa lo mismo en todos los grafos.

Por eso probamos una representacion mas directa que WL: en vez de contar patrones, vectorizamos cada posible arista `u -> v`.

Formula:

```math
\psi_{uv}(G)
=
\begin{cases}
w_{uv}, & \text{si } (u,v) \in E(G) \\
0,      & \text{si } (u,v) \notin E(G)
\end{cases}
```

El vector completo es:

```math
\psi(G)
=
(\psi_{u_1v_1}(G), \psi_{u_1v_2}(G), \ldots, \psi_{u_nv_n}(G))
```

Intuicion:

```text
WL pregunta:
  "Que patrones locales aparecen en el grafo?"

Fixed-layout pregunta:
  "Cuanto pesa exactamente cada conexion layer/token -> layer/token?"
```

Que aporta:

- Aprovecha que el layout del transformer es fijo.
- No pierde identidad posicional.
- Sirve como baseline fuerte y muy simple.

Pros:

- Accuracy mas alta en GPT-2.
- Muy interpretable: cada feature es una arista concreta.
- No requiere GNN ni entrenamiento profundo.

Contras:

- Puede capturar distribuciones de pesos por posicion, no solo topologia.
- Si cambia el layout del modelo o tokenizacion, hay que alinear dimensiones.
- Puede sobreajustar si hay pocos grafos bootstrap.

Resultado GPT-2:

```text
Fixed-layout bootstrap linear = 1.0000
Fixed-layout bootstrap RBF    = 0.8433
```

Lectura: es la representacion con mas accuracy. Pero hay que mirar controles nulos antes de interpretarla como estructura causal.

### 5. Nuevo: null controls

Los null controls responden una pregunta critica:

```text
La accuracy viene de estructura real o de un artefacto facil?
```

Definimos:

```math
\Delta_{\mathrm{null}}
=
\operatorname{Acc}_{\mathrm{observada}}
-
\mathbb{E}[\operatorname{Acc}_{\mathrm{null}}]
```

Si `Delta` es alto, la senal observada sobrevive al control nulo.

#### 5.1 Label permutation

Permuta las etiquetas de clase:

```math
y_i'
=
y_{\pi(i)}
```

Las features no cambian. Si la accuracy sigue alta, el clasificador esta explotando azar o leakage.

Pros:

- Control basico imprescindible.
- Detecta leakage obvio.

Contras:

- No dice si la senal es topologica o de pesos.

#### 5.2 Edge shuffle

Conserva los pesos pero cambia los endpoints:

```math
E'
\sim
\operatorname{ShuffleEndpoints}(E)
```

```math
\{w_e' : e \in E'\}
=
\{w_e : e \in E\}
```

Si la accuracy cae, la posicion/topologia de las aristas importaba.

Pros:

- Testea si importa donde estan las conexiones.
- Mantiene la distribucion global de pesos.

Contras:

- Puede crear grafos artificiales poco realistas.
- No preserva necesariamente grados exactos.

#### 5.3 Weight shuffle

Conserva la topologia pero permuta pesos:

```math
E' = E
```

```math
w_e'
=
w_{\pi(e)}
```

Si la accuracy cae, los pesos correctos en las aristas correctas importaban.

Pros:

- Separa topologia de asignacion de pesos.
- Muy util para interpretar fixed-layout.

Contras:

- Si las distribuciones de pesos ya separan clases, puede no destruir toda la senal.
- No prueba causalidad por si solo.

## Resumen de pros y contras

| Representacion | Que prueba | Pros | Contras |
|---|---|---|---|
| WL per-example | Si grafos auxiliares por ejemplo separan slices | Simple, historico, reproducible | No es el objeto canonico; senal moderada |
| WL bootstrap-slice | Si grafos tipo slice separan clases usando WL | Mas cercano al metodo principal; buenos null deltas | RBF debil; muestras bootstrap dependientes |
| Fixed-layout bootstrap | Si los edge slots reales separan clases | Mejor accuracy; interpretable por arista | Weight-shuffle delta bajo; puede capturar patrones de pesos |
| GNN, no implementada | Si una red aprende mejores embeddings | Flexible; puede usar atributos ricos | Mas riesgo de overfit; menos interpretable; requiere mas datos |

## Cambios de codigo

### `src/pig/graph.py`

Se anadio `GraphBuilder.build_from_tensors()`. Es la misma construccion que el grafo canonico por slice, pero acepta un subconjunto explicito de tensores. Esto permite construir grafos bootstrap sin cambiar `build_from_slice()`.

Se anadio `build_bootstrap_slice_graphs()`. Para cada slice:

1. Toma los tensores de ese slice.
2. Re-muestrea ejemplos con bootstrap.
3. Construye un grafo de correlacion con `build_from_tensors()`.
4. Marca el grafo como `auxiliary_bootstrap_slice_baseline`.

### `src/pig/graph_features.py`

Modulo nuevo con dos funciones principales:

- `compute_fixed_layout_features_from_list()`: convierte cada grafo en un vector denso de edge slots `src -> dst`.
- `apply_null_control_to_graphs()`: genera controles nulos por `edge_shuffle` y `weight_shuffle`.

Controles implementados:

- `label_permutation`: conserva features, permuta labels.
- `edge_shuffle`: conserva pesos, cambia endpoints respetando direccion si el grafo original era dirigido forward.
- `weight_shuffle`: conserva topologia, permuta pesos entre aristas.

### `scripts/run_classical_publication_study.py`

El runner ahora reporta tres familias auxiliares:

- `wl_per_example`: baseline heredado.
- `wl_bootstrap_slice`: WL sobre grafos bootstrap por slice.
- `fixed_layout_bootstrap_slice`: pesos directos `src -> dst` sobre grafos bootstrap.

Tambien escribe deltas contra null controls:

```text
delta = accuracy_observada - accuracy_null_media
```

Un delta alto indica que la senal no se explica facilmente por el control nulo.

## Resultados GPT-2

Artefactos:

- `outputs/classical_publication/gpt2_wl_controls_20260430_res_k5_n20/runs/*.json`
- `outputs/classical_publication/gpt2_wl_controls_20260430_res_k5_n20/summary_all_seeds.csv`

### Accuracy media

| Representacion | Linear SVM | RBF SVM | Lectura |
|---|---:|---:|---|
| WL per-example, anterior | 0.6667 | 0.6250 | Baseline original |
| WL per-example, nuevo | 0.6667 | 0.6250 | Reproduce exactamente el baseline |
| WL bootstrap-slice | 0.8000 | 0.4867 | Mejora en linear, RBF cae a azar |
| Fixed-layout bootstrap | 1.0000 | 0.8433 | Mejor accuracy global |

### Por seed

| Seed | WL per-example linear | WL bootstrap linear | Fixed-layout linear | Causal survival |
|---:|---:|---:|---:|---:|
| 7 | 0.675 | 0.760 | 1.000 | 1.000 |
| 42 | 0.600 | 0.800 | 1.000 | 1.000 |
| 123 | 0.725 | 0.840 | 1.000 | 1.000 |

### Null-control deltas, linear SVM

| Representacion | Label delta | Edge delta | Weight delta | Lectura |
|---|---:|---:|---:|---|
| WL per-example | 0.1617 | 0.1783 | 0.1283 | Senal real pero moderada |
| WL bootstrap-slice | 0.3913 | 0.3767 | 0.3667 | Senal mas robusta que per-example |
| Fixed-layout bootstrap | 0.5240 | 0.5133 | 0.0653 | Muy fuerte, pero sensible al control de pesos |

Visualmente:

```text
Accuracy linear
WL per-example      #######--- 0.667
WL bootstrap        ########-- 0.800
Fixed-layout        ########## 1.000

Robustez vs weight shuffle
WL per-example      #--------- 0.128
WL bootstrap        ####------ 0.367
Fixed-layout        #--------- 0.065
```

## Interpretacion

`WL per-example` no empeoro; reproduce el resultado anterior. Eso es importante porque valida que la refactorizacion no rompio la metrica legacy.

`WL bootstrap-slice` parece metodologicamente mas sano que `WL per-example`: usa grafos con la misma semantica que el objeto canonico por slice. Su linear accuracy sube a `0.8000` y sus deltas contra controles son bastante mas fuertes. El problema es que RBF cae a `0.4867`, lo que sugiere que la geometria WL resultante no es estable bajo RBF o que el tamano de muestra bootstrap aun es pequeno.

`Fixed-layout bootstrap` gana claramente en accuracy. Esto tiene sentido porque el grafo PIG tiene un layout fijo: nodo `layer=3, token=5` significa lo mismo en todos los grafos. WL, en cambio, transforma estructura en histogramas de vecindarios y puede perder informacion posicional fina. Para esta pipeline, una representacion directa de edge slots es un baseline muy competitivo.

La advertencia principal es el `weight_shuffle delta` bajo de fixed-layout (`0.0653`). Eso indica que parte de su poder puede venir de patrones de magnitud/distribucion de pesos en posiciones fijas, no necesariamente de la topologia causal fina. No invalida el resultado, pero obliga a no venderlo como prueba topologica fuerte sin mas controles.

## Conclusion sobre GNN

No recomiendo saltar todavia a una GNN como sustituto principal de WL.

Razones:

- Fixed-layout ya captura una senal mas fuerte que WL sin aprendizaje profundo.
- El dataset sigue siendo pequeno para entrenar una GNN sin sobreajuste.
- Una GNN message-passing estandar no necesariamente sera mas expresiva que WL para estructura; su ventaja seria aprender pesos/atributos, pero eso requiere mas datos y controles.

La siguiente mejora razonable es optimizar y formalizar `fixed-layout` y `WL bootstrap-slice`, no meter una GNN todavia.

## Recomendacion siguiente

1. Mantener `WL per-example` solo como baseline historico.
2. Promover `WL bootstrap-slice` como baseline WL mas metodologicamente correcto.
3. Incluir `fixed-layout bootstrap` como baseline clasico fuerte.
4. Reportar siempre null controls, especialmente `weight_shuffle`.
5. Antes de GNN, anadir variantes fixed-layout:
   - pesos crudos,
   - signo separado de magnitud,
   - ranking/top-k binario,
   - features de grado por layer/token,
   - train/test split por seed para evitar dependencia bootstrap excesiva.

## Nota operacional

La corrida GPT-2 se relanzo con GPU libre:

```bash
HF_HOME=/home/ruben/PIG/.cache/huggingface \
CUDA_VISIBLE_DEVICES=2 \
uv run python scripts/run_classical_publication_study.py \
  --model-name gpt2 \
  --device cuda \
  --seeds 123 \
  --k-grid 5 \
  --num-examples-grid 20 \
  --node-types-grid 'res' \
  --output-dir outputs/classical_publication/gpt2_wl_controls_20260430_res_k5_n20 \
  --cache-root .cache/classical_publication/gpt2_wl_controls_20260430_res_k5_n20
```

La GPU ayuda, pero no tanto como deberia porque la pipeline actual hace muchos forwards pequenos con hooks y poca agregacion/batching. El siguiente cuello tecnico real es batchear patching/causal eval.
