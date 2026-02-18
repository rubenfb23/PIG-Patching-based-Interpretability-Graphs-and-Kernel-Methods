# CLMI (src/clmi): ¿qué experimentos son y cómo se hacen?

Este módulo (CLMI) es una **suite de experimentos** sobre **aprendizaje continuo (continual learning) en GPT‑2** + **interpretabilidad mecánica**, usando tareas sintéticas controladas.

La idea es construir un escenario donde:
- el modelo aprende una tarea **A**,
- luego aprende una tarea **B** que **interfiere** con A,
- y luego vuelve a entrenar en **A** (fase A2),

para medir **olvido (forgetting)**, **histeresis**, y relacionarlo con medidas de **no conmutatividad** (NC) en “subespacios causales” del modelo.

---

## 1) Tarea sintética A/B (diccionario con solapamiento controlado)
Archivo clave: `src/clmi/data/synth_tasks.py`

Se construyen dos tareas de tipo “diccionario”:

- Cada ejemplo es un prompt tipo:
  
  ```
  Q: <clave>
  A:
  ```
  
  y el objetivo es que el modelo prediga un **valor** asociado a esa clave.

- La función `generate_task_pair(...)` crea:
  - una tarea **A** con `n_keys` claves y valores,
  - una tarea **B** con `n_keys` claves y valores,
  - un parámetro `overlap ∈ [0,1]` que decide cuántas claves se comparten.

**Punto importante:**
- Para las claves compartidas, se fuerza un **conflicto**: la misma clave tiene **valor distinto** en A y en B.
- Así, aprender B tiende a **romper** el desempeño en A (interferencia / catastrophic forgetting).

---

## 2) Protocolo de aprendizaje continuo (A → B → A2)
Script principal: `scripts/run_pair.py`

Para cada par A/B:

1. **M0 (inicio):** se carga el modelo base (por defecto GPT‑2).
2. **Fase A:** fine‑tuning en tarea A → checkpoint **MA**.
3. **Fase B:** fine‑tuning en tarea B (mientras se monitorea caída en A) → checkpoint **MAB**.
4. **Fase A2:** se re‑entrena en A para medir recuperación (histeresis) → checkpoint **MABA**.

Dos variantes del orden:
- `--protocol ABA` (por defecto)
- `--protocol BAB` (invirtiendo el orden para tests de simetría)

Dos modos de fine‑tuning:
- `--ft-mode full`: entrena parámetros normales del modelo.
- `--ft-mode lora`: entrena adaptadores LoRA (requiere `peft`).

---

## 3) Qué métricas se reportan (lo “medible” del experimento)

### (a) Accuracy por fase
Se mide la exactitud (exact match) de A y B en distintos checkpoints:
- `AccA_MA`, `AccA_MAB`, `AccA_MABA`
- `AccB_MAB`, `AccB_MABA`

### (b) Forgetting
Archivo: `src/clmi/metrics/forgetting.py`

Se define como:
- `forgettingA = AccA_MA − AccA_MAB`

(es decir: cuánto cae A después de aprender B).

### (c) Histeresis (remanencia, coercitividad, área)
También en `src/clmi/metrics/forgetting.py`.

Se mira la curva de recuperación cuando se vuelve a entrenar A (fase A2):
- **remanence:** qué tan bien queda A justo después de B,
- **coercivity_steps:** cuántos pasos tarda en recuperar (p. ej. 95% del nivel de MA),
- **hysteresis_area:** área de “déficit” durante la recuperación.

### (d) Robustez a ruido
Archivo: `src/clmi/metrics/robustness.py`

Evalúa el desempeño en:
- entradas limpias,
- entradas con ruido (typos / reemplazo de clave),
- ruido agregado en embeddings (hook sobre la capa de embeddings).

---

## 4) Subespacios causales por capa (proyectores)
Archivo: `src/clmi/causal/subspaces.py`

Aquí se construye una representación del “subespacio” relevante para cada tarea.

Resumen:
- Se ejecuta el modelo y se extrae el **hidden state residual** en la última posición del prompt.
- Se define un objetivo `y` basado en **logit_diff** (logit de token correcto − token negativo).
- Para cada capa se ajusta un subespacio de dimensión `k`:
  - se entrenan muchos regresores Ridge con bootstrap,
  - se apilan sus vectores y se ortonormaliza (QR),
  - eso produce una base `U` y un proyector `P = U U^T`.

Estos proyectores se cachean en `results/cache/projectors/`.

---

## 5) No conmutatividad (NC): la hipótesis central
Archivo: `src/clmi/metrics/noncommutativity.py`

Con proyectores `P_A` y `P_B` por capa se mide:

- **NC por capa:** tamaño del conmutador
  
  \[ [P_A, P_B] = P_A P_B - P_B P_A \]
  
  usando la norma de Frobenius `||·||_F`.

- **NC global:** promedio ponderado entre capas.

**Intuición para explicarlo:**
- Si los “mecanismos” que necesita A y los que necesita B se superponen de manera conflictiva, sus proyectores pueden “pelearse”.
- La no conmutatividad intenta cuantificar esa incompatibilidad.

Además hay un concepto de **NC funcional**:
- se compara intervenir primero con A y luego con B (**AB**) vs al revés (**BA**)
- y se mide divergencia entre distribuciones de salida.

---

## 6) Intervenciones funcionales (AB vs BA)
Archivo: `src/clmi/causal/patching.py` + `src/clmi/model/hooks.py`

Se define una intervención “suave” sobre hidden states:

- aplicar una proyección `P` y sumar/restar un término:
  - `reinforce`: `h ← h + β · (hP)`
  - `suppress`: `h ← h − β · (hP)`

Luego se compara:
- aplicar proyectores de A y B en orden **AB**
- vs en orden **BA**

y se mide `KL(p_AB || p_BA)` sobre un set de prompts.

---

## 7) Kernels: convertir tareas en “similitudes”
Archivo: `src/clmi/kernels/kernels.py`

Se definen 3 kernels entre tareas:

1. **k_proj (solapamiento de subespacios)**
   - suma `Tr(P_A P_B)` por capa.

2. **k_NC (kernel basado en no conmutatividad)**
   - `exp( −γ · ||[P_A,P_B]||^2 )`.

3. **k_func (kernel funcional)**
   - compara embeddings de efecto de intervención `phi(T)` (lineal o RBF).

En `scripts/run_experiments.py` estos kernels se usan para:
- construir matrices kernel,
- correlacionarlas con métricas de forgetting/interferencia,
- ajustar modelos simples tipo kernel ridge para “predecir” forgetting.

---

## 8) Mitigaciones probadas (para reducir interferencia)
Script: `scripts/run_pair.py`

Durante la fase B se prueban opciones:

- `--mitigation none`: baseline.

- `--mitigation freeze_nc`:
  - calcula NC pre‑B,
  - elige las capas con NC más alta,
  - las congela para que B no las modifique.

- `--mitigation anchor_reg`:
  - calcula “targets” de activaciones (anchors) en capas problemáticas (top‑NC),
  - durante entrenamiento en B agrega un término que empuja a mantener esas activaciones.

---

## 9) Cómo se corre (comandos típicos)

### Correr 1 experimento A/B (un par)

```bash
uv run python scripts/run_pair.py \
  --model gpt2 \
  --device auto \
  --seed 0 \
  --pair-id 0 \
  --overlap 0.5 \
  --ft-mode full \
  --mitigation none
```

### Correr la batería completa (muchos overlaps/seeds)

```bash
uv run python scripts/run_experiments.py --model gpt2 --device auto
```

(Para corridas grandes hay un script tipo “serio” en `scripts/run_serious.sh`.)

---

## 10) Qué sale al final (artefactos)
Los resultados típicos quedan en:
- `results/runs/<run_id>/` (cada corrida individual)
- `results/tables/` (resúmenes agregados)
- `outputs/figures/` o `results/figures/` (plots)

Ejemplos de archivos por corrida:
- `pair_metrics.json` (métricas finales)
- `nc_layers.csv` (NC por capa)
- `hysteresis_curve.csv` (curva A→B→A2)
- `robustness_A.csv`, `robustness_B.csv`
- `phi_embeddings.npz` (para análisis kernel offline)

---

## 11) Explicación “en una frase” (para decirlo en voz alta)
Entrenamos GPT‑2 en dos tareas sintéticas A y B que compiten por las mismas claves, medimos cuánto olvida A tras aprender B, extraemos subespacios por capa que explican el comportamiento, cuantificamos su incompatibilidad con no conmutatividad (incluyendo intervención AB vs BA), y probamos mitigaciones (congelar capas conflictivas o regularización por anclas) para reducir el olvido.
