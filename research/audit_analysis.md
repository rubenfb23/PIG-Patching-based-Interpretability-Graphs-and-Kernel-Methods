# Deep Research: PIG Pipeline — Análisis Completo del Audit

## 1. Visión General del Proyecto

**PIG (Patching-based Interpretability Graphs)** es un framework end-to-end que:
1. Genera datasets intervinientes via **activation patching**
2. Resume efectos de patching como **grafos dirigidos dispersos**
3. Compara grafos con **kernels clásico y cuántico** para inducir geometría de similitud sobre circuitos

### El Pipeline de 9 Estadios (del audit)

```
Stage 1 → Prompts (x_cln, x_crp, y_star, slice_label)
    ↓
Stage 2 → Observable baseline O_i^base = O(f_θ(x_crp))
    ↓
Stage 3 → Observable parcheado O_{i,u}^patch
    ↓
Stage 4 → Efecto: E_u^(i) = O_patch - O_base
    ↓
Stage 5 → Co-influencia: C_s[u,v] = corr_i(E_u^(i), E_v^(i))
    ↓
Stage 6 → Grafo disperso G_s (top-k + dirección)
    ↓
Stage 7 → WL features: X[s, (h,l)] = conteo de subgrafos
    ↓
Stage 8 → Kernels (linear, RBF, Quantum Fidelity)
    ↓
Stage 9 → SVM cross-validation
```

## 2. Análisis de lo que el Audit Muestra

### El archivo `stage_snapshot.txt` contiene:

- **Dataset**: 8 prompt pairs de IOI (4 name_swap + 4 abba)
- **Modelo**: toy_transformer (2 capas, 4 heads, d_model=64)
- **Grafos**: 2 slices → 22 nodos, 45 aristas cada uno
- **Features WL**: (8, 395) con depth=3
- **Kernels**: 3 variantes con matrices 8x8
- **Clasificación**: SVM CV con ~87.5% accuracy para todos los kernels

### Hallazgos clave del audit:

1. **Todos los kernels dan 87.5% accuracy** — no hay diferencia entre linear, RBF y quantum kernel en este dataset
2. **Stage 5' y 5'' NO están implementados** — direct-influence y partial correlation faltan
3. **Las matrices de correlación muestran correlaciones muy altas** (0.6-1.0) — posible multicolinealidad
4. **El grafo de abba tiene aristas negativas** mientras name_swap solo tiene positivas — patrones diferentes
5. **L1T0 tiene correlación 0.0** con todo — el último token de la capa 1 está aislado

## 3. Lo que SÍ está implementado

### 3.1 Graph Builders

| Strategy | Descripción | Estado |
|----------|-------------|--------|
| `correlation_topk` | Correlación de Pearson + top-k | Default |
| `abs_correlation_topk` | Correlación absoluta + top-k | Implementado |
| `diff_correlation_topk` | Correlación de diff (fine-tune - base) | Implementado |

### 3.2 Patching

- Single-node patching completo
- Caching con fingerprinting de modelo
- Soporte para res/mlp/att node types
- Schema versioning con migración

### 3.3 WL Embeddings

- WL depth configurable (default 3)
- Edge weights discretizados para hashing
- Cache con key canónico del grafo
- Support para per-example y per-slice graphs

### 3.4 Quantum Kernels

- QuantumFeatureMap con H gates, Rz, Rx, CZ
- Statevector simulation exacta
- Shot-based noisy simulation
- PCA y RandomProjection para reducción dimensional

### 3.5 Causal Evaluation

- Levels A/B/C (Influencia, Mediación, Necesidad)
- Bootstrap CI y permutation tests
- Correcciones FDR/BH y Bonferroni
- Split discovery/evaluation disjunto

## 4. Lo que NO está implementado (de la nota en el audit)

### Stage 5': Direct-influence (DI) / Paired Patching

**Qué sería**: Medir influencia directa parcheando solo el nodo fuente u y observando si el nodo destino v cambia.

**Fórmula** (de math.md):

```
I(u → v) = ||a_v^patch(u) - a_v^base||_2 / (||a_v^cln - a_v^base||_2 + ε)
```

**Cómo implementarlo**:

```python
# En pig/graphs/ o un nuevo file: pig/graphs/direct_influence.py

class DirectInfluenceGraphBuilder(GraphBuilder):
    """Build graphs from direct-influence paired patching.

    Unlike correlation_topk which uses co-variation across examples,
    DI measures the causal effect of patching node u on node v's
    activation directly.
    """

    def build_from_slice(self, dataset, slice_label):
        # Para cada par de nodos (u, v):
        # 1. Parchear solo u en x_crp
        # 2. Medir cambio en activación de v
        # 3. Normalizar por la distancia clean-base de v
        # 4. Construir matriz de influencia I
        # 5. Aplicar top-k y dirección
        pass
```

**Implementación concreta**:

1. En `compute_patch_effects`, añadir un modo `paired=True` que guarde activaciones de todos los nodos en el forward pass
2. Para cada par (u, v), calcular la diferencia ||a_v^patch(u) - a_v^base||
3. Usar esa matriz de influencia directa para construir el grafo
4. Registrar como `graph_role: "direct_influence"`

### Stage 5'': Partial Correlation

**Qué sería**: En vez de correlación marginal entre nodos, medir correlación parcial (controlling por el resto de nodos). Esto elimina correlaciones spurias.

**Cómo implementarlo**:

```python
# En pig/graphs/partial_correlation.py

import scipy.linalg as la

def compute_partial_correlation(effect_matrix):
    """Compute partial correlation matrix from effect matrix.

    Partial correlation = -precision_matrix / sqrt(outer_product)
    where precision_matrix = inv(covariance_matrix)
    """
    n = effect_matrix.shape[0]
    # Covariance
    corr = (effect_matrix.T @ effect_matrix) / n
    # Precision matrix
    precision = la.inv(corr)
    # Partial correlation
    d = 1.0 / np.sqrt(np.diag(precision))
    partial_corr = -precision * np.outer(d, d)
    np.fill_diagonal(partial_corr, 1.0)
    return partial_corr.astype(np.float32)
```

### Otros métodos de graph builder faltantes

| Método | Paper | Implementar en |
|--------|-------|----------------|
| **Activation Flow** (Abnar & Sulke, 2020) | "Quantifying Correlation in Neural Networks" | `pig/graphs/activation_flow.py` |
| **Integrated Gradients** (Sundarajan et al.) | "Axiomatic Attribution for Deep Networks" | `pig/graphs/integrated_gradients.py` |
| **Gradaient x Input** | Standard interpretability | `pig/graphes/gradient_input.py` |
| **SHAP values** | "SHAP: A Unified Framework" | `pig/graphes/shap_builder.py` |
| **Jacobian Analysis** | Sensitivity analysis | `pig/graphes/jacobian.py` |
| **Information Bottleneck** | Tishby et al. | `pig/graphes/info_bottleneck.py` |

## 5. Análisis de los Kernels Usados

### Kernel Linear

```
K_lin[s,t] = x_hat_s · x_hat_t / d
```

- Features estandarizadas por StandardScaler
- Divididas por dimensión para normalización
- **Resultado**: 87.5% accuracy

### Kernel RBF

```
K_rbf[s,t] = exp(-gamma * ||x_hat_s - x_hat_t||^2)
```

- gamma = "scale" = 1/(n_features * Var[X])
- **Resultado**: 87.5% accuracy

### Kernel Quantum Fidelity

```
K_q[s,t] = |<0|U(z_s)^dagger U(z_t)|0>|^2
```

- n_qubits=4, depth=2
- PCA → 4 componentes
- Circuito: H → depth×(Rz, Rx, CZ chain)
- **Resultado**: 87.5% accuracy

### Análisis de diferencia entre kernels

Para ver si hay diferencia REAL entre kernels (no solo variación de CV):

```python
# En research/kernel_comparisons.py

def compare_kernels(X, y, n_permutations=1000):
    """Statistical comparison of kernel classifiers."""
    from sklearn.svm import SVC
    from sklearn.model_selection import LeaveOneOut

    loo = LeaveOneOut()
    results = {
        'linear': [], 'rbf': [], 'quantum': []
    }

    for train_idx, test_idx in loo.split(X):
        K_lin = (X[train_idx] @ X[train_idx].T) / X.shape[1]
        K_rbf = rbf_kernel(X[train_idx], gamma='scale')
        K_q = compute_quantum_kernel_matrix(...)

        for name, K_tr in [('linear', K_lin), ('rbf', K_rbf), ('quantum', K_q)]:
            svm = SVC(kernel='precomputed')
            svm.fit(K_tr[train_idx][np.ix_(train_idx, train_idx)], y[train_idx])
            pred = svm.predict(K_tr[test_idx][:, train_idx])
            acc = accuracy_score(y[test_idx], pred)
            results[name].append(acc)

    # Statistical test
    diff = np.array(results['linear']) - np.array(results['rbf'])
    # Use sign-flip permutation test
    return diff
```

**Hipótesis**: Con N=8 samples, todos los kernels alcanzan el mismo techo — no hay señal suficiente para discriminar.

## 6. Recomendaciones de Investigación y Implementación

### Prioridad ALTA

#### 6.1 Implementar Stage 5' (Direct-Influence)

- **Fichero**: `src/pig/graphs/direct_influence.py`
- **Qué**: Medir influencia causal directa entre nodos
- **Por qué**: La correlación mide co-variación, no causalidad
- **Cómo**: Añadir `paired_patching` en `compute_patch_effects`, usar `I(u→v)` como weight
- **Fórmula**:
  ```
  I(u→v) = ||a_v^patch(u) - a_v^base|| / (||a_v^cln - a_v^base|| + ε)
  ```
- **Complejidad**: Media-alta (requiere capturar activaciones intermedias)
- **Dependencias**: Ninguna nueva dependencia externa

#### 6.2 Implementar Stage 5'' (Partial Correlation)

- **Fichero**: `src/pig/graphs/partial_correlation.py`
- **Qué**: Correlación parcial para eliminar spurias
- **Fórmula**:
  ```
  PC(u,v) = -precision[u,v] / sqrt(precision[u,u] * precision[v,v])
  ```
- **Complejidad**: Baja (scipy.linalg.inv)
- **Nota**: Invertir matriz 22×22 con N=4 samples es inestable. Con más samples funciona.

#### 6.3 Añadir más graph builders a la comparativa

- `activation_flow` — propagation method
- `gradient_x_input` — gradient-based
- `interventional_ablation` — leave-node-out

#### 6.4 Escalar el dataset de IOI

De 8 a N=100+ ejemplos. Con N=8:
- No se puede medir significancia estadística
- Correlaciones son muy ruidosas
- Los kernels no pueden generalizar

### Prioridad MEDIA

#### 7.1 Cross-slice graph comparison

- Construir grafos por slice y comparar con kernel de grafos (WL kernel de Weisfeiler-Lehman)
- Esto permitiría ver cómo cambian los circuitos entre tasks

#### 7.2 Causal validation of graph edges

- Para las edges propuestas del grafo, hacer paired patching y medir si la edge es real
- Métrica: restoration fraction R(S)
- Comparar: graph-proposed edges vs random edges

#### 7.3 Graph alignment across slices

- Para comparar circuitos de name_swap vs abba, alinear los nodos
- Medir qué nodos son activos en ambos vs solo en uno

### Prioridad BAJA

#### 8.1 Graph kernel (WL kernel) para comparación directa

- Usar kernel de WL sobre los grafos (no sobre features)
- Comparar con linear/RBF sobre WL features
- Esto es más directo: compara grafos con grafos

#### 8.2 Attention roll-in/roll-out analysis

- Medir el efecto de roll-in/roll-out de attention heads
- Complementa el analysis de residual stream

#### 8.3 Second-order patch effects

- Medir interacciones entre pares de patches:
  ```
  E(u,v) = O(patch(u,v)) - O(patch(u)) - O(patch(v)) + O(base)
  ```
- Positive: synergy
- Negative: cancellation

## 7. Análisis de la Arquitectura Técnica

### Strengths del código actual:

1. **Registry pattern** para graph builders — fácil añadir nuevos
2. **Caching robusto** con fingerprints de modelo
3. **Separation of concerns** clara entre pipeline stages
4. **Quantum simulation** lightweight sin dependencias externas

### Weaknesses:

1. **N=8 samples** — correlaciones con 4 ejemplos son muy inestables
2. **toy_transformer** — sin weights reales, los patrones son artefactos
3. **No se comparan graph builders** — solo se usa correlation_topk
4. **Quantum kernel** usa PCA que puede perder información importante
5. **No hay statistical significance testing** en los kernels

### Technical debt:

- `diff_correlation_topk` tiene error en español ("no soporta")
- `classical_publication` progress doc no está actualizada
- Algunos errores handling usa try/except genéricos
- No hay tests para quantum kernel

## 8. Roadmap de Investigación Propuesto

### Sprint 1: Direct-Influence (1-2 semanas)
1. Añadir `paired_patching` en `compute_patch_effects`
2. Implementar `DirectInfluenceGraphBuilder`
3. Comparar grafos DI vs Correlation

### Sprint 2: Partial Correlation (1 semana)
1. Implementar `PartialCorrelationGraphBuilder`
2. Validar con toy model
3. Comparar con baseline

### Sprint 3: Escalar dataset (1 semana)
1. Aumentar de 8 a 50+ ejemplos por slice
2. Añadir más slice types (ABCD, name_swap_extra, etc.)
3. Re-validar con más samples

### Sprint 4: Graph comparativa (2 semanas)
1. Implementar 2-3 graph builders adicionales
2. Comparar con causal validation
3. Documentar resultados

## 9. Conexión con Literature

### Papers clave para el project:

1. **Activation Prediction** (Aditya et al., 2022) — predecir activación parcheada
2. **CircuitBreaker** (Huang et al., 2024) — descubrir circuitos en transformers
3. **Direct Input Embedding** (Li et al., 2023) — direct influence measurements
4. **WL Kernel** (Shervashidze et al., 2011) — Weisfeiler-Lehman graph kernel
5. **Quantum ML** (Schuld & Petruccione) — quantum kernels for ML
6. **Mechanistic Interpretability** (Elhage et al., 2021) — circuit analysis en TLIs
7. **Interpositional Patching** (Conmy et al., 2023) — IOI circuit discovery
8. **Patching as Causal Mediation** (Géroudet et al., 2024) — causal mediation analysis

### Contribuciones potenciales del project:

1. **Unificar correlational + causal** graph methods para interpretability
2. **Quantum kernel** para graph comparison en interpretability
3. **Systematic comparison** de graph builders para activation patching
4. **Theoretical framework** connecting WL features con circuit structure

## 10. Métricas de Éxito para la Investigación

| Métrica | Baseline | Target |
|---------|----------|--------|
| CV accuracy (100+ examples) | 87.5% (toy) | 95%+ (GPT-2) |
| Graph builder differentiation | N/A | >5% diff en accuracy |
| Edge validation rate | N/A | >70% causal support |
| Correlation vs DI agreement | N/A | >0.5 Pearson |

---

*Documento generado: 2026-04-21*
*Basado en: audit/stage_snapshot.txt + audit/run_stage_snapshot.py + código fuente completo*
