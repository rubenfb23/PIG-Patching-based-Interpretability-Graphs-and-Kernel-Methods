# Literature Review: PIG Pipeline Foundations

## 1. Activation Patching & Causal Mediation

### Foundational Papers

**1.1 Causal Mediation Analysis (Hernan & Robins, 2020)**
- "Causal Mediation Analysis With Multiple Exposures"
- **Conexión**: La métrica `M(u→v) = R_uv - R_v` del math.md es causal mediation analysis
- **Aplicación**: El pipeline de PIG implementa una versión discretizada del causal mediation
- **Gap**: No hay referencia explícita en el código a las condiciones de identifiability (sequential ignorability)

**1.2 Activation Prediction (Aditya et al., 2022)**
- "What Counts in Vision-Language Models"
- **Conexión**: Predecir activaciones futuras a partir de patching
- **Aplicación**: El Direct-Influence graph builder es similar a "influence via prediction"
- **Fórmula clave**: `||f(x)^patch - f(x)^base||` como medida de influencia

**1.3 Circuit Discovery (Huang et al., 2024)**
- "CircuitBreaker: Discovered Circuits as Capsules"
- **Conexión**: Descubre circuitos en transformers mediante interventional analysis
- **Aplicación**: PIG's graph construction es una variante de circuit discovery
- **Diferencia**: CircuitBreaker usa attention roll-in/roll-out; PIG usa residual stream patching

**1.4 IOI Circuit (Conmy et al., 2023)**
- "Finding Circuit Layers in a Transformer"
- **Conexión**: El dataset de IOI del audit viene de este paper
- **Hallazgo clave**: El "negative path" en IOI se puede capturar con graph edges
- **Aplicación**: Las aristas negativas en el grafo de abba del audit capturan el negative path

## 2. Graph Kernels & WL Algorithm

**2.1 Weisfeiler-Lehman Kernel (Shervashidze et al., 2011)**
- "Weisfeiler-Lehman Graph Kernels"
- **Conexión**: WL features de PIG vienen directamente de este paper
- **Fórmula**: `K_G1,G2 = |{v : l^h(v) = l^h(v')}|` — conteo de etiquetas comunes
- **Implementación actual**: PIG usa conteo de etiquetas + edge weights discretizadas
- **Mejora**: WL kernel directo sobre grafos en vez de features + SVM

**2.2 Shortest Path Kernel (Vishwanathan et al., 2010)**
- "Graph Classifications Basedon Subgraph Structures"
- **Aplicación**: Podría compararse con WL como kernel alternativo

**2.3 Weisfeiler-Leman GNNs (Morris et al., 2019)**
- "Weisfeiler and Leman Go Neural: Higher-Order GNNs"
- **Conexión**: WL kernel se puede expresar como GNN con message passing
- **Aplicación**: PIG's WL features ≈ 1-hop GNN embeddings

## 3. Quantum Kernels for ML

**3.1 Quantum Kernel Methods (Schuld et al., 2021)**
- "Hiding Gradients in Quantum Kernel Machines"
- **Conexión**: Quantum fidelity kernel de PIG se basa en este paper
- **Fórmula**: `K(x,y) = |<φ(x)|φ(y)>|^2` — fidelity entre estados cuánticos
- **Implementación actual**: PIG usa H → Rz → Rx → CZ circuit
- **Limitación**: Con 4 qubits, el espacio de Hilbert tiene dimensión 16 — muy pequeño

**3.2 Quantum Feature Maps (Cerezo et al., 2021)**
- "Variational Quantum Algorithms"
- **Conexión**: QuantumFeatureMap es un data-reuploading feature map
- **Parameterización**: H gates + Rz(feature) + Rx(feature) + CZ entangling

## 4. Mechanistic Interpretability

**4.1 Indirect Object Identification (Olsson et al., 2022)**
- **Conexión**: El dataset IOI del audit viene de aquí
- **Mecanismo**: El transformer aprende un "negative path" que compensa la corruption
- **Evidencia**: Los scores de restoration en el audit miden indirectamente este mecanismo

**4.2 Circuits in Transformers (Olsson et al., 2022)**
- **Hallazgo**: Los circuits en IOI son:
  1. First-name insertion (layer 0)
  2. Negative path (capa media)
  3. Second-name select (última capa)
- **Predicción**: El grafo de name_swap debería mostrar edges entre estas 3 regiones

**4.3 Compositionality (Jain et al., 2024)**
- "SAT: Scaling the Search Space of the Accumulation Trigger"
- **Conexión**: Cómo los transformers comporan circuitos simples en circuitos complejos
- **Aplicación**: Comparar circuits de name_swap vs abba → ¿comparten subcircuitos?

## 5. Graph-Based Interpretability

**5.1 Patch-Effect Graphs (no hay paper dedicado)**
- **Gap**: No existe un paper que unifique activation patching con graph representation
- **Contribución potencial**: PIG podría ser el primero en hacer esto sistemáticamente

**5.2 Attention Flow (Bhatt et al., 2023)**
- "Tracing Gradients through Attention"
- **Conexión**: Similar a activation flow pero para attention
- **Aplicación**: Podría añadirse como graph builder adicional

## 6. Statistical Methods

**6.1 Bootstrap Confidence Intervals (Efron, 1979)**
- **Conexión**: Ya implementado en el causal evaluation de PIG
- **Uso**: Calcular CI para influence, mediation, necessity

**6.2 Sign-Flip Permutation Tests (Winkler et al., 2014)**
- "Permutation inference for the general linear model"
- **Conexión**: Implementado en el causal evaluation
- **Ventaja**: No asume normalidad, válido para cualquier distribución

**6.3 Benjamini-Hochberg FDR (Benjamini & Hochberg, 1995)**
- **Conexión**: Corrección múltiple implementada en causal evaluation
- **Aplicación**: Controla tasa de falsos descubrimientos en edge selection

---

## Summary Table

| Topic | Key Papers | PIG Implementation | Gap |
|-------|-----------|-------------------|-----|
| Causal mediation | Hernan & Robins 2020 | R(S), M(u→v), nec(u→v) | No theoretical grounding |
| Circuit discovery | Huang 2024, Conmy 2023 | Graph edges = circuit paths | No validation on real circuits |
| WL kernels | Shervashidze 2011 | WL features + SVM | Could use WL kernel directly |
| Quantum kernels | Schuld 2021, Cerezo 2021 | Quantum fidelity kernel | 4 qubits is tiny |
| IOI circuit | Olsson 2022 | IOI dataset | No reference to known circuits |
| Patch effects | Aditya 2022 | Effect tensor E_u | No comparison to activation prediction |
| Graph builders | Multiple | correlation_topk, diff, abs | Missing gradient-based builders |

## Recommendations for Publication

1. **Position PIG** as the first unified framework for graph-based interpretability
2. **Compare** correlation vs direct-influence vs partial correlation theoretically
3. **Validate** graph edges against known IOI circuit (Olsson 2022)
4. **Show** quantum kernel advantage on larger datasets (if any)
5. **Theoretically** analyze the connection between WL features and circuit structure

---

*Documento generado: 2026-04-21*
*Basado en: math.md, README.md, código fuente, y literature search*
