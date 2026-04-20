# Role-based steering for reducing sycophancy: Gemma 2 27B and Qwen 3 32B

Aggregated results and paper artefacts for a sycophancy-steering study that
asks three questions on two instruction-tuned LLMs:

1. Do general **persona directions** (e.g. "skeptic", "devil's advocate")
   reduce sycophancy as effectively as a **targeted CAA vector** trained
   specifically on sycophancy data (Rimsky et al. 2024)?
2. Does that reduction live in the component **orthogonal to CAA**, the
   component **aligned with CAA**, or both?
3. When the CAA-orthogonal residual of a role is evaluated as a
   **standalone steering direction with its own tune-locked coefficient**,
   does it still reduce sycophancy? Does the pattern "high-|cos(role, CAA)|
   residual doesn't reduce" from the matched-coefficient decomposition
   analysis hold when residuals are steered with their own optimal
   coefficients?

Evaluated on held-out `sycophancy_on_philpapers2020` A/B preferences
(Perez et al. 2023), 300 base questions × 2 orderings per seed, with
direction-aware best-coefficient selection on the tune half and
Wilcoxon+Holm–Bonferroni on the test half across all 24 conditions.

## Top-level findings

**(1) General persona directions match targeted CAA.** On both models,
steering with a "skeptic" persona direction reduces sycophancy to within
~10% of what the targeted CAA vector achieves, even though the CAA vector
is trained on syc-specific A/B contrasts and the persona direction is
not.

| | Gemma 2 27B | Qwen 3 32B |
|---|---|---|
| CAA Δ logit | −0.87 | −1.97 |
| Skeptic Δ logit | −0.71 | −1.82 |
| Skeptic / CAA (fraction of CAA's effect) | 0.82 | 0.93 |

**(2) Role reduction lives in the CAA-orthogonal component.** In the
matched-coefficient decomposition analysis — steering at each role's
tune-locked coefficient with either (a) only the unit-normalised
CAA-aligned component of the role, or (b) only the unit-normalised
residual — the **residual alone** recovers essentially the full role
effect on both models. Every critical role's residual reduces sycophancy
at the parent role's best coefficient. See
`gemma/results/decomposition_eval_test.json` and
`qwen/results/decomposition_eval_test.json`.

**(3) Standalone residual steering works on both models.** Giving each
residual its own tune-locked coefficient (not reusing the parent role's)
and evaluating on held-out test with Holm across all 24 conditions:

| Residual | Gemma Δ logit [95% CI], Holm p | Qwen mean Δ logit (3 seeds), Holm |
|---|---|---|
| Skeptic ⊥ CAA | −0.75 [−0.96, −0.57], p=3.6e-11 **✱** | −1.80 ± 0.14, **3/3 ✱** |
| Contrarian ⊥ CAA | −0.32 [−0.55, −0.10], p=0.072 **ns (fails Holm)** | −1.65 ± 0.12, **3/3 ✱** |
| Collaborator ⊥ CAA (Gemma's high-\|cos\| role) | −0.93 [−1.13, −0.76], p<1e-15 **✱** | — |
| Devil's Advocate ⊥ CAA (Qwen's high-\|cos\| role) | — | −2.19 ± 0.08, **3/3 ✱** |

**On Gemma 2 of 3 standalone residuals survive Holm-Bonferroni across
24 conditions; Contrarian ⊥ CAA does not (raw p=0.020, Holm-adjusted
p=0.072 on n=150 test bases, single seed).** Its 95% CI excludes zero
but the adjusted p does not. Reporting this as a null result is
important; the Gemma write-up should not treat all three residuals as
confirmed reducers. On Qwen all three are Holm-significant in every
seed.

**(4) The high-|cos(role, CAA)| residual does *not* fail to reduce
sycophancy when given its own tune-locked coefficient.** This is the
aspect of the matched-coefficient decomposition claim that does **not**
replicate under standalone evaluation on either model. On Gemma,
Collaborator ⊥ CAA (|cos|=0.165, the highest in the decomposition) is
the largest reducer of the three (Δ=−0.93); Contrarian ⊥ CAA
(|cos|=0.033, the lowest) fails Holm significance. On Qwen, Devil's
Advocate ⊥ CAA (|cos|=0.108, the highest) is the largest reducer
(Δ=−2.19 mean across seeds).

## Key model differences

| | Gemma 2 27B | Qwen 3 32B |
|---|---|---|
| target layer | 22 / 46 | 32 / 64 (canonical from `assistant_axis.models.MODEL_CONFIGS`) |
| coefficient sweep | ±500, ±1000, ±2000, ±5000 | ±50, ±100, ±200, ±500 (10× smaller: Qwen's activation norm is smaller, so same absolute coefs saturate) |
| chat-template flag | (N/A) | `enable_thinking=False` — otherwise the A/B logit is measured at the `<think>` token rather than post-`</think>` |
| baseline syc logit (test) | +1.01 | +3.00 |
| baseline syc rate (test) | 59% | 84% |
| bidirectionality (conformist roles significantly raise syc) | 1 / 4 (Collaborator only) | 0 / 4 — likely ceiling effect (Qwen starts 84% syc) |
| sign of cos(role, CAA) on critical roles | mostly +near-zero | mostly −near-zero |
| role-vector source | `lu-christina/assistant-axis-vectors/gemma-2-27b/` | `lu-christina/assistant-axis-vectors/qwen-3-32b/` |

## Directory layout

```
role-based-steering/
├── README.md                                    (this file)
├── paper/
│   ├── RESULTS.md                               numbers + interpretation
│   ├── METHODS.md                               protocol, layers, sweeps, stats
│   ├── LIMITATIONS.md                           honest caveats
│   ├── figures/                                 publication figures (per model + cross)
│   ├── tables/
│   │   ├── main_results_both_models.csv        one row per (model, condition)
│   │   └── main_results_both_models.md         same, compact markdown
│   └── scripts/
│       ├── build_unified_table.py              rebuilds the table from per-model results
│       └── residual_standalone_sidebyside.py   rebuilds cross_model/*.md
├── gemma/
│   ├── results/                                 aggregated JSONs + summaries
│   ├── caa_decomposition.json
│   └── figures/                                 per-model figures incl. fig7
├── qwen/
│   ├── results/                                 same structure
│   ├── caa_decomposition.json, caa_metadata.json
│   └── figures/
└── cross_model/
    ├── residual_standalone_gemma_vs_qwen.md    side-by-side narrative
    └── residual_standalone_gemma_vs_qwen.csv   one row per (model, residual)
```

Per-row eval outputs (`all_results_*.json`), raw steering vectors, and
per-seed checkpoints are **not** included — they are rebuildable from the
upstream repos and are large. For those:

- Gemma pipeline: https://github.com/kelkarI/sycophancy-gemma
  (previously `kelkarI/sycophancy-final`; renamed for naming parity with
  `kelkarI/sycophancy-qwen`; GitHub auto-redirects the old URL)
  (directory `experiment-main/`)
- Qwen pipeline: https://github.com/kelkarI/sycophancy-qwen

## Reproduction

Each model's pipeline is fully reproducible from its own repo. This
"central" repo consumes those pipelines' outputs; the regeneration script
for the unified table is:

```bash
python paper/scripts/build_unified_table.py
python paper/scripts/residual_standalone_sidebyside.py
```

Both read from `gemma/` and `qwen/` and write to `paper/tables/` and
`cross_model/` respectively.

## Citations

- **CAA method**: Rimsky et al. 2024, *Steering Llama 2 via Contrastive
  Activation Addition*. arXiv:2312.06681.
- **Persona/assistant-axis vectors**: `lu-christina/assistant-axis-vectors`
  on HuggingFace (generation-plus-LLM-judge pipeline from
  `safety-research/assistant-axis`).
- **Sycophancy benchmark**: Perez et al. 2023, *Discovering Language Model
  Behaviors with Model-Written Evaluations*. arXiv:2212.09251.
- **Models**: `google/gemma-2-27b-it`, `Qwen/Qwen3-32B`.
