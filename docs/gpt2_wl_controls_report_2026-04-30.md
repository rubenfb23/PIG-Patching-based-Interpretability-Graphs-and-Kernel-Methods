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
