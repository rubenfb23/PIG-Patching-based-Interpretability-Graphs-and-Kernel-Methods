# Estrategia de publicación NeurIPS — PIG

## Qué requiere un paper de NeurIPS

Un paper de NeurIPS necesita **una claim novedosa con evidencia sólida**.
"Hicimos una herramienta" no es suficiente.
"Descubrimos X sobre cómo funcionan los LLMs usando esta herramienta" sí lo es.

---

## Estado del arte relevante

Los **conceptos individuales** están publicados:

| Técnica | Paper | Año |
|---|---|---|
| Activation patching | Elhage et al. | 2021 |
| Path patching | Goldowsky-Dill et al. | 2023 |
| ACDC (circuit discovery automático) | Conmy et al., NeurIPS | 2023 |
| IOI circuit (ejemplo canónico) | Wang, Conmy et al., ICLR | 2023 |
| Causal scrubbing | Redwood Research | 2022 |
| Circuit Tracing con SAEs | Anthropic | 2025 |

El paper más cercano al enfoque PIG es el de **Anthropic 2025** (`transformer-circuits.pub/2025/attribution-graphs`), que usa SAE features como nodos + attribution graphs + validación causal. Es más granular pero requiere SAEs entrenados.

---

## Qué hace PIG que no está hecho exactamente

1. **Grafo de correlación de patch effects**: ACDC testa aristas una por una. PIG construye un grafo de correlación entre vectores de efectos — más barato, estructura diferente.
2. **Combinación de capas de análisis en una herramienta**: grafo correlacional + overlay causal (I_mean, M_mean, necessity) + filtro backbone/mediated + visualización 4D.
3. **Clasificación mediated vs parallel_or_synergy** aplicada sistemáticamente a escala.

---

## Brechas en la literatura que PIG puede llenar

### Opción 1 — ¿Cuándo divergen el grafo correlacional y el causal?

Nadie ha caracterizado sistemáticamente cuántos edges correlacionales son espurios (causa común) vs genuinamente causales.

**Claim potencial:**
> *"X% de los edges en grafos correlacionales de activación son artefactos de causa común. El backbone causal es Y veces más pequeño y Z veces más estable entre ejemplos."*

**Evidencia necesaria:** comparar ambos grafos en modelos y tareas con ground truth conocido (IOI, Greater-Than, GSM8K).

---

### Opción 2 — ¿Qué pasa con los circuitos al ajustar o destilar? ★ Recomendada

Estado real de artefactos (auditado por `model_name` en `causal_eval.json`, corte 2026-02-23):

```
outputs/causal_eval/<model>_<utc>_seed<seed>/causal_eval.json
  model_name = toy_transformer                                  ← baseline toy

outputs/causal_eval_ft_robust/causal_eval.json
  model_name = outputs/gpt2_gsm8k_distilled_lr2e5_acc4/model_final

outputs/causal_eval_ft_robust2/causal_eval.json
  model_name = outputs/gpt2_gsm8k_distilled_lr2e5_acc4/model_final

outputs/causal_eval_ft_lr2e5_acc4/causal_eval.json
  model_name = outputs/gpt2_gsm8k_distilled_lr2e5_acc4/model_final
```

Importante:
- `causal_eval_ft_robust` y `causal_eval_ft_robust2` no son condiciones independientes; apuntan al mismo checkpoint.
- Con la evidencia actual no se puede sostener una comparación "base GPT-2 vs fine-tuned GPT-2 vs distilled" como tres modelos distintos.

Comprobación rápida reproducible (incluye directorios versionados y legacy):
```bash
for f in outputs/causal_eval*/causal_eval.json outputs/causal_eval/*/causal_eval.json; do
  [ -f "$f" ] || continue
  printf "%s -> " "$f"
  python -c 'import json,sys;print(json.load(open(sys.argv[1]))["model_name"])' "$f"
done
```

**Claim potencial (ajustado al estado actual):**
> *"El backbone causal del student GPT-2 entrenado con datos destilados muestra estructura reproducible; al repetir corridas, los edges mediados varían más que el núcleo de alta necessity."*

**Por qué es publicable:**
- Empíricamente novedoso
- Implicaciones prácticas (¿qué destruye la destilación?)
- PIG es la herramienta que lo hace posible

---

### Opción 3 — ¿Son los circuitos de razonamiento matemático universales?

El paper IOI (2023) demostró un circuito para una tarea lingüística en GPT-2 Small. Nadie ha hecho lo mismo sistemáticamente para **razonamiento aritmético** comparando modelos.

**Claim potencial:**
> *"Identificamos un backbone de N heads/MLPs causalmente necesarias para aritmética en GPT-2. Este backbone es X% estable entre modelos de distinto tamaño."*

---

### Opción 4 — PIG como método más barato que ACDC

ACDC requiere miles de forward passes por edge. PIG propone candidatos baratos vía correlación y valida causalmente solo los top-K.

**Claim potencial:**
> *"La correlación de patch effects es un proxy fiable para circuitos causales. PIG consigue X% de los edges de ACDC con Y% del coste computacional."*

**Requiere:** validación en benchmarks con ground truth de ACDC (IOI, Greater-Than).

---

## Recomendación concreta

El ángulo con **menor trabajo adicional** dado lo que ya existe:

```
"Causal Backbone Stability in Distilled GPT-2 Students:
 A Reproducibility-Centered Analysis for Mathematical Reasoning"
```

### Experimentos disponibles
- Baseline toy con causal eval (`outputs/causal_eval`) ✓
- Causal eval del checkpoint student destilado (`outputs/gpt2_gsm8k_distilled_lr2e5_acc4/model_final`) ✓
- Repeticiones sobre el mismo checkpoint destilado (`causal_eval_ft_robust*`) ✓

### Experimentos pendientes
- **Ablación causal cuantitativa** (ver sección siguiente)
- Corrida separada de GPT-2 base y de un fine-tuned no destilado (checkpoints distintos)
- Comparación con al menos un modelo más (GPT-2 Medium o similar)
- Interpretación funcional de cada subcomponente del backbone

### Timeline
- NeurIPS 2025 deadline: ~mayo/junio 2025
- Factible si los experimentos base ya están

---

## El experimento crítico: ablación causal

Esta es la pregunta que hará cualquier reviewer:

> *"¿Por qué debería creer que los 'backbone circuits' son realmente causalmente necesarios y no un artefacto del umbral de necessity?"*

**Diseño del experimento:**

1. Tomar el backbone identificado (parallel + necessity > threshold)
2. Bloquear esos edges → medir caída de accuracy en GSM8K
3. Bloquear un conjunto aleatorio de edges con misma necessity → medir caída
4. Si la diferencia es grande → el backbone es causalmente específico

```
Accuracy drop:
  Ablate backbone edges:   Δ = ?   ← debe ser grande
  Ablate random edges:     Δ = ?   ← debe ser pequeño
  Ratio:                   >> 1    ← claim validada
```

Sin este experimento el paper es descriptivo. Con él es causal.

---

## Conceptos clave para el paper

| Término | Significado en PIG |
|---|---|
| **Backbone circuit** | Edges con `classification=parallel_or_synergy` y `necessity_mean` alto |
| **Mediated edge** | u influye en v pero v es un relé; bloquear u→v no afecta si v se parchea directamente |
| **Parallel/synergy edge** | u y v contribuyen por caminos independientes; ambos necesarios |
| **I_mean** | Influencia de activación: ¿cuánto mueve u la activación de v? (intervención directa) |
| **necessity_mean** | ¿Cuánto cae el rendimiento si bloqueo el path u→v? |
| **M_mean** | Score de mediación: ¿está el efecto de u sobre el output mediado a través de v? |
| **Correlational graph** | Pearson corr entre vectores de patch effects — mide co-variación, no causalidad |
| **Causal graph** | Intervención directa do(u=clean) — mide P(v\|do(u)) en notación Pearl |

---

## Referencias

- Conmy et al. (NeurIPS 2023): https://arxiv.org/abs/2304.14997
- Goldowsky-Dill et al. (2023): path patching
- Wang et al. (ICLR 2023): https://arxiv.org/abs/2211.00593
- Heimersheim & Nanda (2024): https://arxiv.org/abs/2404.15255
- Marks et al. Sparse Feature Circuits (2024): https://arxiv.org/abs/2403.19647
- Anthropic Circuit Tracing (2025): https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Geiger et al. Causal Abstraction (2025): https://arxiv.org/abs/2301.04709
