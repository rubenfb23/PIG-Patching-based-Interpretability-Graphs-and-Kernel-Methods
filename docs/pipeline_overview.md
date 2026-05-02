# PIG — Visión general del pipeline

## 1. Tipos de patches comprobados

### Tipos de nodos (componentes del transformer)

Definidos en `src/pig/model.py`:

| Tipo | Constante | Descripción |
|------|-----------|-------------|
| **Residual stream** | `NODE_TYPE_RES = "res"` | Activación del residual stream en cada capa y token |
| **MLP** | `NODE_TYPE_MLP = "mlp"` | Salida del bloque MLP de cada capa |
| **Attention** | `NODE_TYPE_ATT = "att"` | Salida de cada cabeza de atención (`head` específica) |

Cada nodo queda indexado por `(layer, token, node_type[, head])`.  
El conjunto completo: `ALLOWED_NODE_TYPES = {"res", "mlp", "att"}`.

El valor por defecto al correr el pipeline es solo `"res"` (residual stream), pero se pueden combinar los tres.

---

### Tipos de corrupción del prompt

Definidos en `src/pig/prompts.py`, tarea: **IOI (Indirect Object Identification)**.

#### `name_swap` — `NameSwapCorruption`
Sustituye el nombre del sujeto S por el nombre del objeto indirecto IO.
El target `y_star` desaparece del prompt corrupto.

```
Limpio:   "John gave the book to Mary. Mary gave it back to"  →  predice "John"
Corrupto: "Mary gave the book to Mary. Mary gave it back to"  →  "John" ya no aparece
```

#### `abba` — `ABBACorruption`
Intercambia los roles de S e IO en el template (patrón ABBA).
Ambos nombres siguen presentes, pero en posiciones invertidas.

```
Limpio (ABAB): "John gave to Mary. Mary gave back to"  →  predice "John"
Corrupto (ABBA): "Mary gave to John. John gave back to"
```

---

### Fórmula del efecto de patch

Para cada nodo `u = (layer, token, component)`:

$$E_u = O(\text{patched\_forward}(x^{crp}; u)) - O(x^{crp})$$

Un $E_u$ positivo grande indica que restaurar la activación limpia en ese nodo recupera causalmente el comportamiento objetivo.

---

### Resumen de dimensiones

| Dimensión | Valores |
|-----------|---------|
| Tarea | `ioi` (Indirect Object Identification) |
| Corrupción | `name_swap`, `abba` |
| Nodo — tipo | `res`, `mlp`, `att` |
| Nodo — posición | `(layer, token[, head])` |

---

## 2. Origen de los ejemplos

Los ejemplos se generan **sintéticamente** dentro de `IOIGenerator`, a partir de dos listas hardcodeadas en `src/pig/prompts.py`.

### Lista de nombres (`NAMES`)
24 nombres ingleses comunes. En cada ejemplo se eligen **2 nombres distintos** al azar (`rng.sample`).

### Lista de templates (`TEMPLATES`)
6 plantillas con huecos `{S}` (sujeto) e `{IO}` (objeto indirecto). Se elige **1 template** al azar (`rng.choice`):

```
"{S} gave the book to {IO}. {IO} gave it back to"
"{S} lent the money to {IO}. {IO} returned it to"
"{S} passed the ball to {IO}. {IO} threw it to"
"{S} sent a letter to {IO}. {IO} replied to"
"{S} handed the keys to {IO}. {IO} returned them to"
"{S} showed the photo to {IO}. {IO} gave it back to"
```

### Proceso de generación

```
S, IO  ← sample(NAMES, 2)          # ej. "John", "Mary"
template ← choice(TEMPLATES)        # ej. "{S} gave the book to {IO}..."

x_cln  = template(S=S, IO=IO)       # "John gave the book to Mary. Mary gave it back to"
y_star = S                           # "John"

# name_swap:  x_crp = x_cln.replace(S, IO)   → "Mary gave the book to Mary..."
# abba:       x_crp = template(S=IO, IO=S)    → "Mary gave the book to John..."
```

La reproducibilidad está controlada por `seed=42` en `random.Random`. No hay datos externos — todo es generación procedural.

---

## 3. Qué es activation patching

La técnica se llama **activation patching** (también conocida como *causal tracing* o *interchange intervention*). Es una técnica de interpretabilidad mecanística de redes neuronales.

### La pregunta que responde

Dado un transformer que produce una respuesta correcta con `x_cln` y una incorrecta con `x_crp`:  
**¿Qué componente interno es causalmente responsable de esa diferencia?**

### El procedimiento

```
1. Ejecutar el modelo con x_cln  →  guardar activaciones limpias
2. Ejecutar el modelo con x_crp  →  obtener baseline score O(x_crp)
3. Volver a ejecutar con x_crp, pero en un solo nodo u=(layer, token, componente)
   sustituir la activación corrupta por la limpia (el "patch")
4. Medir el nuevo score O_patched
```

- $E_u \approx 0$ → ese nodo **no importa** para la diferencia de comportamiento  
- $E_u$ grande → ese nodo **sí es causal**: restaurar su activación limpia recupera el comportamiento correcto

### Lo que hace PIG específicamente

1. Calcula $E_u$ para **todos los nodos** de todas las capas y tokens
2. Construye un **grafo** donde los nodos son posiciones del transformer y los pesos de las aristas reflejan la co-variación de efectos entre nodos
3. Usa los grafos por `slice` como objeto canónico del método y deja la clasificación clásica como evidencia auxiliar

> Técnica popularizada por *"Locating and Editing Factual Associations in GPT"* (Meng et al., 2022) y *"Interpretability in the Wild"* (Wang et al., 2022) — este último precisamente en la tarea IOI que usa este repo.

---

## 4. Pipeline de comparación

### Paso 1 — Grafo canónico por slice

Para cada slice (`name_swap`, `abba`) se construye un `PatchInfluenceGraph` (`src/pig/graph.py`):
- Nodos = posiciones del transformer
- Aristas ponderadas por co-variación de efectos entre nodos
- Top-k sparsificación para estandarizar densidad

Este es el objeto principal de PIG para discovery y validación causal.

### Paso 1b — Grafo auxiliar por ejemplo

`build_per_example()` genera grafos adicionales para clasificación auxiliar.
No son el objeto canónico del método: se usan solo para tener suficientes muestras en `WL + SVM`.

### Paso 1c — Grafos bootstrap por slice

`build_bootstrap_slice_graphs()` re-muestrea ejemplos dentro de cada slice y
vuelve a construir grafos de correlación. Esto da más muestras para
clasificación auxiliar sin abandonar la semántica del grafo canónico por slice.

### Paso 2 — Embedding WL (Weisfeiler-Lehman)

`src/pig/embeddings.py` convierte cada grafo en un **vector de conteos de subgrafos**:

```
grafo → WL iterations → hash de vecindarios → histograma de features
```

Resultado: una matriz `X` de forma `[n_ejemplos, n_features_WL]`.

La pipeline clásica ahora compara WL contra un baseline `fixed-layout`, que
vectoriza directamente los pesos `src -> dst` sobre el layout fijo del
transformer. Si este baseline iguala o supera a WL, WL no está añadiendo mucha
información estructural.

### Paso 3a — Kernel clásico (evidencia auxiliar)

`src/pig/kernels.py` entrena un **SVM** con:

- `linear`: producto escalar sobre features WL
- `rbf`: similitud gaussiana $\exp(-\gamma \|x_i - x_j\|^2)$

Métricas reportadas: **accuracy** y **AUC-ROC** en clasificar a qué slice pertenece cada grafo auxiliar.
La evaluación clásica usa normalización dentro del CV para evitar leakage.
El runner también reporta controles nulos con permutación de etiquetas,
shuffle de topología y shuffle de pesos; el valor útil es el delta entre la
accuracy observada y la accuracy bajo control nulo.

### Paso 3b — Validación causal multinivel

`src/pig/causal.py` toma candidatos propuestos por correlación y los somete a:

1. Split disjunto `discovery/evaluation`
2. Métricas de influencia/restauración/mediación
3. Correcciones por múltiples tests
4. Resumen de qué parte del grafo correlacional sobrevive como backbone causal

### La pregunta de fondo

¿Qué fracción de los edges propuestos por correlación sobrevive validación causal estricta, con qué estabilidad y a qué coste?  
La clasificación clásica responde una pregunta secundaria: si los grafos auxiliares contienen señal suficiente para distinguir slices, entonces la representación estructural no es trivial.

---

## 5. Almacenamiento intermedio (caché)

La matriz WL no se persiste como artefacto final — existe en memoria durante el pipeline. Los pasos costosos se cachean:

| Artefacto | Directorio | Clase |
|-----------|-----------|-------|
| Efectos de patch $E_u$ | `.cache/patch_effects/<model_key>/` | `PatchEffectCache` — `src/pig/patching.py` |
| Embeddings WL (feature matrix) | `.cache/wl_embeddings/` | `WLEmbeddingCache` — `src/pig/embeddings.py` |

Detalles de seguridad/reproducibilidad de caché:

- Patch cache guarda metadata fuerte por tensor (`cache_schema_version`, `model_name`, `model_fingerprint`, `axis_fingerprint`, `node_types`, `created_at_utc`).
- En evaluación causal, el loader filtra estrictamente por `model_fingerprint` + `axis_fingerprint`.
- Entradas legacy sin metadata moderna se rechazan por defecto (`fail-closed`).
- El key de caché WL usa hash canónico del contenido del grafo (nodos + aristas + pesos + dirección + slice), no solo conteos agregados.

Ambas usan archivos **JSON** con nombre = SHA-256 (16 chars) del contenido.  
Las figuras de salida se escriben en `outputs/`.

### Trazabilidad de salidas causales

`pig causal-eval` escribe por defecto en un directorio versionado por modelo/fecha/seed:

- `outputs/causal_eval/<model>_<utc>_seed<seed>/`

Artefactos clave por corrida:

- `causal_eval.json`
- `causal_eval.npz`
- `quicklook_causal.png`
- `causal_eval.log`

El comando falla si `--output-dir` ya existe y no está vacío, salvo que se pase `--overwrite`.

---

## 6. El kernel trick en cada SVM

| Kernel | Trick real | Espacio implícito |
|--------|-----------|-------------------|
| `linear` | No | $\mathbb{R}^d$ (features WL) |
| `rbf` | **Sí** | $\mathbb{R}^\infty$ (funciones de Mercer) |
| `quantum fidelity` | Solo en hardware real | $\mathbb{C}^{2^n}$ (estados cuánticos) |

### RBF — único con kernel trick real

$$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$$

Corresponde implícitamente a un espacio de features de dimensión infinita. La SVM nunca calcula ese espacio — solo evalúa $K$ entre pares de puntos.

### Quantum fidelity — simulado

$$K(i,j) = |\langle \psi_i | \psi_j \rangle|^2$$

El feature map explícito es $\phi(x) = |\psi(x)\rangle \in \mathbb{C}^{2^n}$.  
En esta implementación los statevectors se computan explícitamente (`state_matrix.conj() @ state_matrix.T`), por lo que **no hay ventaja computacional** — es una simulación clásica de lo que haría hardware cuántico real.  
En hardware cuántico real mediría $|\langle \psi_i | \psi_j \rangle|^2$ directamente con un circuito de interferencia sin materializar el vector de $2^n$ dimensiones.
