## Notación base

  Para cada **prompt pair** $(x^\text{cln},\ x^\text{crp},\ y^*)$:

  | Símbolo | Descripción |
  |---|---|
  | $O_\text{cln}$ | `score(x_cln, y*)` — log-prob del modelo en el prompt limpio |
  | $O_\text{base}$ | `score(x_crp, y*)` — log-prob en el prompt corrompido |
  | $O_\text{patch}(S)$ | log-prob cuando se parchean los nodos del conjunto $S$ (activaciones de $x^\text{cln}$ inyectadas en el forward
  pass de $x^\text{crp}$) |
  | $\varepsilon$ | constante de estabilidad numérica ($10^{-6}$) |

  ---

## Propuesta de candidatos (antes de evaluar)

  Se construye la matriz de efectos de patch $\mathbf{X} \in \mathbb{R}^{N \times D}$ donde $N$ = ejemplos y $D$ = nodos totales:

  $$\mathbf{X}_{t,i} = \frac{\text{effects}_{t,i}}{\max(O^\text{cln}_t - O^\text{base}_t,\ \varepsilon)}$$

  Se centra y estandariza por columnas, y se calcula la matriz de correlación:

  $$\mathbf{C} = \frac{\tilde{\mathbf{X}}^\top \tilde{\mathbf{X}}}{N}, \quad C_{ij} = \text{correlación de efectos de patch entre nodo } i
  \text{ y nodo } j$$

  Los **top-k pares dirigidos** $(u \to v)$ con $u < v$ (orden causal forward) y mayor $|C_{uv}|$ son los candidatos.

  ---

  ## Restauración fraccionaria $R(S)$

  Para cualquier conjunto de nodos parchados $S$:

  $$\boxed{R(S) = \frac{O_\text{patch}(S) - O_\text{base}}{O_\text{cln} - O_\text{base} + \varepsilon}}$$

  $R(S) \approx 0$: parchar $S$ no ayuda. $R(S) \approx 1$: se recupera completamente el comportamiento limpio.

  ---

  ## Nivel A — Influencia de activación $I(u \to v)$

  Mide si parchar el nodo fuente $u$ desplaza la activación del nodo destino $v$.

  Sea $\mathbf{a}_v^{\text{base}}$, $\mathbf{a}_v^{\text{cln}}$, $\mathbf{a}_v^{\text{patch}(u)}$ los vectores de activación en $v$:

  $$\boxed{I(u \to v) = \frac{\|\mathbf{a}_v^{\text{patch}(u)} - \mathbf{a}_v^{\text{base}}\|_2}{\|\mathbf{a}_v^{\text{cln}} -
  \mathbf{a}_v^{\text{base}}\|_2 + \varepsilon}}$$

  $I=0$: parchar $u$ no mueve $v$. $I=1$: $v$ se desplaza tanto como en el forward limpio.

  ---

  ## Nivel B — Mediación $M(u \to v)$

  Primero se computan tres fracciones de restauración:

  $$R_u = R(\{u\}), \quad R_v = R(\{v\}), \quad R_{uv} = R(\{u, v\})$$

  El **score de mediación**:

  $$\boxed{M(u \to v) = R_{uv} - R_v}$$

  - $M \approx 0$: parchar $u$ no añade nada a lo que ya hace $v$ → la influencia de $u$ **pasa a través de** $v$ (clasificación:
  `mediated`)
  - $|M| > 0.05$: $u$ y $v$ contribuyen por caminos independientes → `parallel_or_synergy`

  ---

  ## Nivel C — Necesidad $\text{nec}(u \to v)$

  Se mide cuánto depende la efectividad de $u$ de que $v$ pueda transmitir la señal. Se computa el score parchando $u$ pero **clampeando**
  $v$ de vuelta a su activación base:

  $$R_{u,\ \text{clamp}(v)} = R\bigl(\{u\} \text{ patch} + v \leftarrow \mathbf{a}_v^{\text{base}}\bigr)$$

  $$\boxed{\text{nec}(u \to v) = R_u - R_{u,\ \text{clamp}(v)}}$$

  - $\text{nec} > 0$: bloquear $v$ reduce la efectividad de $u$ → $v$ es un eslabón **necesario** en el camino causal
  - $\text{nec} \approx 0$: $u$ puede ejercer su efecto sin que $v$ propague nada

  ---

  ## Tests estadísticos (aplicados a cada métrica)

  Para cada métrica $X$, dadas $N$ observaciones $\{x_1, \ldots, x_N\}$:

  **Bootstrap CI** ($B$ muestras, $\alpha = 0.05$):

  $$\hat{\mu} = \frac{1}{N}\sum_i x_i, \quad \text{CI} = \left[\hat{q}_{B,\,\alpha/2},\ \hat{q}_{B,\,1-\alpha/2}\right]$$

  donde $\hat{q}_{B,p}$ es el cuantil $p$ de $\{\bar{x}^{(b)}\}_{b=1}^B$ con remuestreo con reemplazo.

  **Sign-flip permutation test** ($H_0: \mu = 0$, $P$ permutaciones):

  $$T_\text{obs} = \left|\hat{\mu}\right|, \quad T^{(p)} = \left|\frac{1}{N}\sum_i s_i^{(p)} x_i\right| \quad \text{con } s_i^{(p)} \sim
  \text{Rademacher}$$

  $$\text{p-value} = \frac{\#\{p : T^{(p)} \geq T_\text{obs}\} + 1}{P + 1}$$

  ---

  ## Correcciones por comparaciones múltiples

  Se aplican sobre los p-values de las **7 métricas** $\times$ **$m$ aristas** simultáneamente:

  **Benjamini-Hochberg (FDR)** — ordena los $m$ p-values $p_{(1)} \leq \cdots \leq p_{(m)}$:

  $$\tilde{p}_{(i)}^{\text{BH}} = \min_{j \geq i}\left(\frac{m}{j} \cdot p_{(j)}\right)$$

  **Bonferroni** (más conservador):

  $$\tilde{p}_i^{\text{Bonf}} = \min(m \cdot p_i,\ 1)$$

  ---

  ## Significancia final de una arista

  Una arista $u \to v$ se declara **significativa** si (usando el p-value raw de $I$):

  $$I_\text{mean} > 0 \quad \text{AND} \quad p_I < 0.05$$

  El visualizador aplica además filtros adicionales sobre `necessity_mean` y `classification`.
