# Informe GPT-2: representaciones escalables para sustituir fixed-layout

Fecha: 2026-04-30
Modelo: `gpt2`
Setup: `n=100`, corrupciones `name_swap` y `abba`, seeds `7,42,123`, `node_types=res`, `k=5`, `32` bootstrap graphs por slice, split disjunto por ejemplos base.

## 1. Pregunta

El resultado mas fuerte hasta ahora es `fixed-layout sign linear`, pero no escala bien.

Si el grafo tiene `N` nodos:

```math
N \approx L \cdot T \cdot C
```

Fixed-layout completo crea una feature por posible arista:

```math
D_{\mathrm{fixed}}
=
N(N-1)
```

En GPT-2, para esta configuracion:

```text
D_fixed = 14280 features
```

Esto todavia es manejable. En modelos mas grandes, `L`, `T` y posiblemente `C` crecen, asi que `D_fixed` crece cuadraticamente. La pregunta es si podemos comprimir sin perder demasiada accuracy.

## 2. Opciones probadas

### 2.1 Hashed fixed-layout

Antes:

```math
\phi_{uv}(G)
=
\mathrm{sign}(w_{uv})
```

Problema: hay una coordenada por cada edge slot exacto `u -> v`.

Ahora: mandamos cada edge slot a una dimension fija con hashing estable:

```math
j
=
h(u,v)
\bmod d
```

```math
\phi_j(G)
\mathrel{+}=
\mathrm{sign}(w_{uv})
```

Tambien probamos version weighted:

```math
\phi_j(G)
\mathrel{+}=
w_{uv}
```

Configuracion:

```text
d = 1024
```

Pros:

- Dimension fija aunque el modelo crezca.
- Mantiene identidad aproximada de edge slot.
- Es compatible con SVM lineal.
- Tiempo de construccion proporcional a aristas reales, no a `N^2`.

Contras:

- Hay colisiones: dos edge slots pueden caer en la misma coordenada.
- Menos interpretable que fixed-layout exacto.
- Hay que elegir `d`.

Escalado:

```math
D_{\mathrm{hashed}}
=
d
```

```math
\mathrm{coste}
\approx
O(|E|)
```

Con top-k:

```math
|E|
\approx
kN
```

### 2.2 Coarse fixed-layout

Antes: cada edge exacto es distinto.

Ahora: agrupamos edges por tipo grueso:

```math
\phi_{a,b,\Delta l,\Delta t,s}(G)
=
\sum_{(u,v) \in E}
\mathbf{1}[
c(u)=a,
c(v)=b,
B_l(l_v-l_u)=\Delta l,
B_t(t_v-t_u)=\Delta t,
\mathrm{sign}(w_{uv})=s
]
```

Donde:

- `c(u)` es el tipo de nodo: por ejemplo `res`, `att`, `mlp`.
- `B_l` agrupa diferencias de layer en buckets.
- `B_t` agrupa diferencias de token en buckets.
- `s` es el signo del peso.

Pros:

- Muy compacto.
- Mucho mas interpretable que hashing.
- Dimension casi constante si mantenemos el numero de buckets fijo.
- Captura patron mecanistico grueso: tipo de componente, distancia en layer/token y signo.

Contras:

- Pierde identidad exacta de layer/token.
- Puede mezclar mecanismos distintos dentro del mismo bucket.
- Menos expresivo que fixed-layout exacto.

Escalado aproximado:

```math
D_{\mathrm{coarse}}
\le
|C|^2
\cdot
|B_l|
\cdot
|B_t|
\cdot
3
```

En este experimento, la dimension real fue:

```text
105 a 112 features segun seed
```

### 2.3 Neural graph encoder + SVM

Antes habiamos probado una GNN como clasificador final.

Ahora probamos otra cosa: usar la GNN solo como compresor y mantener el SVM como clasificador final.

La GNN calcula embeddings de nodo con message passing:

```math
m_v^{(t)}
=
\sum_u
\widetilde{A}_{v,u}
h_u^{(t)}
```

```math
h_v^{(t+1)}
=
\mathrm{ReLU}
\left(
W_s^{(t)}h_v^{(t)}
+
W_m^{(t)}m_v^{(t)}
+
b^{(t)}
\right)
```

Pooling del grafo:

```math
z_G
=
\left[
\frac{1}{|V|}
\sum_v h_v^{(T)}
;
\max_v h_v^{(T)}
\right]
```

SVM final:

```math
\hat{y}
=
\mathrm{SVM}(z_G)
```

Importante: la GNN se entrena solo con los grafos de train. Despues se extraen embeddings para train/test y se entrena el SVM encima. El encoder no ve los grafos de test durante entrenamiento.

Configuracion:

```text
hidden_dim = 32
num_layers = 2
epochs = 150
embedding_dim = 64
device = cuda
```

Pros:

- Dimension pequena: `64`.
- Conceptualmente es la opcion mas flexible.
- Escala sobre aristas reales: `O(|E|h)` por capa.

Contras:

- Necesita entrenar parametros.
- Con pocos grafos train puede sobreajustar o aprender poco.
- Menos interpretable que hashing/coarse.

## 3. Resultados

Resultados por seed:

| Seed | Fixed sign | Hashed sign | Hashed weighted | Coarse count | GNN encoder SVM | WL linear |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 1.0000 | 0.9688 | 0.8750 | 0.9219 | 0.6094 | 0.7969 |
| 42 | 0.9688 | 0.9531 | 0.9531 | 0.9375 | 0.5156 | 0.5469 |
| 123 | 1.0000 | 0.8906 | 0.9062 | 0.9219 | 0.8281 | 0.9531 |

Medias:

| Metodo | Dim | Mean accuracy | Std | Lectura |
|---|---:|---:|---:|---|
| Fixed sign linear | 14280 | 0.9896 | 0.0180 | Mejor, pero no escalable |
| Hashed sign linear | 1024 | 0.9375 | 0.0413 | Mejor sustituto escalable |
| Coarse count linear | 105-112 | 0.9271 | 0.0090 | Muy compacto y estable |
| Hashed weighted linear | 1024 | 0.9115 | 0.0393 | Bueno, peor que sign |
| WL bootstrap linear | variable | 0.7656 | 0.2049 | Baseline clasico |
| GNN encoder + SVM linear | 64 | 0.6510 | 0.1604 | No competitivo |
| Spectral linear | 96 | 0.6198 | 0.0451 | Forma global insuficiente |
| Graphlet linear | 38 | 0.6354 | 0.1040 | Motifs locales insuficientes |

Visual:

```text
Fixed sign          ########## 0.990  dim=14280
Hashed sign         #########- 0.938  dim=1024
Coarse count        #########- 0.927  dim=105-112
Hashed weighted     #########- 0.911  dim=1024
WL bootstrap        ########-- 0.766
GNN encoder SVM     ######---- 0.651  dim=64
Spectral            ######---- 0.620
Graphlets           ######---- 0.635
```

## 4. Experimento adicional: `n=500`

Se repitio el mismo test con `n=500`, es decir:

```text
500 prompt pairs por corrupcion y seed
1000 prompt pairs totales por seed
```

Resultados medios:

| Metodo | n=100 mean | n=500 mean | Cambio |
|---|---:|---:|---:|
| Fixed sign linear | 0.9896 | 1.0000 | +0.0104 |
| Hashed sign linear | 0.9375 | 1.0000 | +0.0625 |
| Hashed weighted linear | 0.9115 | 0.9583 | +0.0468 |
| Coarse count linear | 0.9271 | 0.9740 | +0.0469 |
| GNN encoder + SVM linear | 0.6510 | 0.8333 | +0.1823 |
| WL bootstrap linear | 0.7656 | 1.0000 | +0.2344 |
| Spectral linear | 0.6198 | 0.9115 | +0.2917 |
| Graphlet linear | 0.6354 | 0.9948 | +0.3594 |

Por seed en `n=500`:

| Seed | Fixed sign | Hashed sign | Hashed weighted | Coarse count | GNN encoder SVM | WL linear | Spectral | Graphlet |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 1.0000 | 1.0000 | 0.9844 | 1.0000 | 0.8281 | 1.0000 | 0.9375 | 1.0000 |
| 42 | 1.0000 | 1.0000 | 0.9844 | 0.9219 | 0.6875 | 1.0000 | 0.8594 | 0.9844 |
| 123 | 1.0000 | 1.0000 | 0.9062 | 1.0000 | 0.9844 | 1.0000 | 0.9375 | 1.0000 |

Lectura:

- `Hashed sign` llega a `1.0000` en los tres seeds: es el mejor sustituto escalable de fixed-layout.
- `Coarse count` sube a `0.9740`: muy buena opcion si queremos interpretabilidad y pocas features.
- `GNN encoder + SVM` mejora mucho (`0.6510 -> 0.8333`), asi que si recibe mas datos aprende mejor, pero todavia no alcanza a hashed/coarse.
- `Spectral` y `graphlets` tambien mejoran mucho. Esto sugiere que con `n=100` los grafos de slice eran ruidosos; con `n=500`, incluso la forma global/local del grafo ya separa bastante bien.
- El coste real de `n=500` lo domina patching: cada seed calcula `1000` patch effects.

Conclusion del test `n=500`:

```text
Mas datos ayudan a casi todas las representaciones, pero la mejor decision
para escalar sigue siendo hashed fixed-layout sign.
```

## 5. Interpretacion

La conclusion importante no es solo que fixed-layout funciona. La conclusion es:

```text
La senal vive en la identidad signada del edge slot.
```

Spectral y graphlets fallan porque comprimen demasiado:

```text
conservan forma global/local,
pero borran layer/token/edge identity.
```

GNN encoder tambien comprime mucho y aprende con pocos grafos:

```text
64 dimensiones,
pero no suficiente supervision para superar metodos simples.
```

Hashed sign y coarse count son el punto medio correcto:

```text
mantienen identidad mecanistica aproximada
y evitan el crecimiento cuadratico de fixed-layout.
```

## 6. Recomendacion

Para modelos grandes, no usaria fixed-layout completo como representacion principal.

Recomendacion practica:

1. Usar `hashed fixed-layout sign` como sustituto principal escalable.
2. Usar `coarse fixed-layout count` como version mas interpretable y muy compacta.
3. Mantener `fixed-layout sign` en GPT-2 como upper bound no escalable.
4. No usar `GNN encoder + SVM` como main por ahora.

Claim defendible para paper:

```text
The strongest signal is signed edge-slot identity. A 1024-dimensional hashed
edge-slot representation preserves most of the fixed-layout performance while
removing the quadratic feature growth, and a coarse edge-bucket representation
achieves similar performance with roughly 100 features.
```

## 7. Limitacion

Esto escala la representacion del grafo, no elimina todo el coste de patching.

El coste de construir efectos sigue dependiendo de cuantos nodos internos parcheamos:

```math
\mathrm{coste\ patching}
\approx
O(\#\mathrm{examples} \cdot N)
```

Entonces, para modelos grandes, el siguiente cuello de botella sera reducir nodos candidatos o hacer patching mas selectivo. Pero para la parte `graph -> features -> SVM`, la mejor direccion ahora es:

```text
fixed-layout exacto -> hashed sign / coarse count
```

## 8. Comando

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
  --null-repeats 0 \
  --spectral-values 16 \
  --hashed-dim 1024 \
  --gnn-device cuda \
  --gnn-epochs 150 \
  --output-dir outputs/paper_decision/gpt2_n100_res_k5_scalable_representations_20260430 \
  --cache-root .cache/paper_decision/gpt2_n100_res_k5_20260430
```

`n=500`:

```bash
HF_HOME=/home/ruben/PIG/.cache/huggingface \
CUDA_VISIBLE_DEVICES=2 \
uv run python scripts/run_paper_decision_study.py \
  --model-name gpt2 \
  --device cuda \
  --seeds 7,42,123 \
  --k-grid 5 \
  --num-examples-grid 500 \
  --node-types-grid res \
  --graphs-per-slice 32 \
  --null-repeats 0 \
  --spectral-values 16 \
  --hashed-dim 1024 \
  --gnn-device cuda \
  --gnn-epochs 150 \
  --output-dir outputs/paper_decision/gpt2_n500_res_k5_scalable_representations_20260430 \
  --cache-root .cache/paper_decision/gpt2_n500_res_k5_20260430
```
