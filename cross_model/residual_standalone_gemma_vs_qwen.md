# Standalone CAA-orthogonal residual conditions — Gemma vs Qwen

Both models were evaluated with the same 24-condition pipeline (assistant_axis, 5 critical roles, 4 conformist roles, CAA, 10 random controls, 3 standalone residuals), **Holm correction applied per-seed across the 14-condition primary family (11 main + 3 standalone residuals; the 10 random controls are NOT in the family)** on the held-out test split, with coefficients locked from each model's own tune split. The standalone residual for each model is the role vector minus its projection onto CAA, unit-normalised, then run through the full coefficient sweep as an independent steering condition with a fresh tune-locked coefficient.

> **Audit note (`AUDIT_NOTES.md`).** A previous version of this file said "Holm across 24 conditions" — the random controls are not in the Holm family. Some Δ values shown earlier in this file used a single-seed (seed-123) baseline for Qwen (+3.024) and a wrong baseline for Gemma residuals (+1.009 instead of seed-42's +1.0038). All Δ values in this version use either (a) the cross-seed mean baseline + per-seed paired Δ for multi-seed conditions, or (b) the seed-42 baseline for Gemma single-seed residual conditions, clearly labelled.

## Model differences

| Property | Gemma-2-27b-it | Qwen 3 32B |
|---|---|---|
| target layer | 22 / 46 | 32 / 64 |
| coefficient sweep | ±500, ±1000, ±2000, ±5000 | ±50, ±100, ±200, ±500 |
| baseline mean syc logit (cross-seed mean) | **+1.0146** | **+3.000** |
| baseline mean syc logit (seed 42 only) | +1.0038 | +2.822 |
| baseline mean syc logit (seed 123 only) | +1.0095 | +3.024 |
| baseline sycophancy rate (cross-seed mean) | 59.7% | 84.0% |
| role vector source | lu-christina/assistant-axis-vectors (`gemma-2-27b/`) | lu-christina/assistant-axis-vectors (`qwen-3-32b/`) |
| max |cos(role, CAA)| among critical+conformist roles | 0.165 (collaborator) | 0.108 (devils_advocate) |

The coefficient-sweep rescale on Qwen was required because Qwen's layer-32 default-activation norm (~166) is much smaller than Gemma's layer-22 norm, so |coef|=5000 completely saturates Qwen's next-token distribution (top-1 becomes whitespace / control tokens). The rescaled Qwen grid was verified on a pilot probe to give a clean dose-response from baseline-like at |coef|=50 to anti-sycophantic at |coef|=200 to saturated at |coef|=500. Gemma's original grid was retained unchanged.

## Standalone residual results (held-out test split, Holm across 14-condition family)

### Gemma-2-27b-it (single-seed: seed 42)

Baseline logit (seed 42): +1.0038   Baseline rate (seed 42): 59.3%

The Gemma residual conditions were added to the pipeline single-seed (seed 42 only). All Δ values below are computed against the **seed-42 baseline** (not against the multi-seed cross-seed mean baseline used elsewhere in this report). The 95% CIs are paired bootstrap CIs over the n=150 test bases of seed 42.

| Condition | \|cos(role, CAA)\| | Best coef (locked) | Parent's best coef | Δ logit [95% CI] | Δ rate pp [95% CI] | Holm p | sig |
|---|---|---|---|---|---|---|---|
| skeptic_residual       | +0.0640 | +2000 | +2000 (same) | -0.748 [-0.952, -0.562] | -9.3 [-12.7, -6.3] | 3.6e-11 | ✱ |
| contrarian_residual    | +0.0326 | +2000 | +1000 (differs) | -0.316 [-0.547, -0.091] | -4.0 [-8.3, +0.0] | 0.0725 | ns |
| collaborator_residual  | +0.1648 | +5000 | +500 (differs) | -0.924 [-1.124, -0.750] | -8.3 [-11.7, -5.0] | < 1e-15 | ✱ |

### Qwen 3 32B — multi-seed (3 test seeds: 42, 7, 123)

Baseline logit (cross-seed mean): +3.000   Baseline rate (cross-seed mean): 84.0%

All Δ values below are **per-seed paired** (each seed's post-steer minus that seed's baseline, then averaged). Use these as the canonical Qwen residual numbers.

| Condition | \|cos(role, CAA)\| | Best coef (locked) | Parent's best coef | Δ logit (mean ± std across 3 seeds) | Δ rate pp (mean ± std) | Sig seeds (Holm) |
|---|---|---|---|---|---|---|
| skeptic_residual         | -0.1049 | +200 | +200 (same) | -1.799 ± 0.062  | -16.22 ± 1.83 | 3/3 ✱ |
| contrarian_residual      | -0.0705 | -100 | -100 (same) | -1.647 ± 0.054  | -10.11 ± 3.66 | 3/3 ✱ |
| devils_advocate_residual | -0.1078 | +200 | +200 (same) | -2.193 ± 0.181  | -15.56 ± 2.01 | 3/3 ✱ |

**Earlier single-seed (seed 123) Qwen residual numbers** — kept for traceability since they appear in some figures derived from `seed_123/` outputs:

| Condition | Δ logit (seed 123 only, vs seed-123 baseline +3.024) | Note |
|---|---|---|
| skeptic_residual          | -1.749 | Single-seed |
| contrarian_residual       | -1.684 | Single-seed |
| devils_advocate_residual  | -2.130 | Single-seed |

## Qwen multi-seed aggregate (mean ± std across 3 test seeds)

Re-ran the full 24-condition pipeline on Qwen across 5 tune seeds (42, 7, 123, 456, 789) and 3 test seeds (42, 7, 123), matching the Gemma pipeline's multi-seed protocol. Tune-locked coefficients came from the mode-across-seeds aggregate (results/best_coefs_tune_aggregate.json). Wilcoxon+Holm applied per seed.

Baseline across 3 test seeds: logit = +3.000 ± 0.136 (matches the cross-seed mean baseline used in the residual table above; this is a sanity check that the same per-seed baseline values flow through both tables in this report).

| Condition | mean post-steer logit ± std | sig seeds / n | mean Δ logit |
|---|---|---|---|
| assistant_axis             | +0.590 ± 0.082 | 3/3 | -2.410 |
| caa                        | +1.035 ± 0.042 | 3/3 | -1.965 |
| devils_advocate            | +0.728 ± 0.071 | 3/3 | -2.272 |
| contrarian                 | +0.725 ± 0.071 | 3/3 | -2.275 |
| skeptic                    | +1.177 ± 0.134 | 3/3 | -1.823 |
| judge                      | +1.301 ± 0.094 | 3/3 | -1.699 |
| scientist                  | +2.016 ± 0.161 | 3/3 | -0.984 |
| peacekeeper                | +2.291 ± 0.094 | 0/3 | -0.709 |
| pacifist                   | +0.021 ± 0.008 | 0/3 | -2.979 |
| collaborator               | +2.970 ± 0.153 | 0/3 | -0.029 |
| facilitator                | +2.531 ± 0.244 | 0/3 | -0.469 |
| skeptic_residual           | +1.201 ± 0.138 | 3/3 | -1.799 |
| contrarian_residual        | +1.353 ± 0.123 | 3/3 | -1.647 |
| devils_advocate_residual   | +0.807 ± 0.076 | 3/3 | -2.193 |

On Qwen, 3/3 seeds show Holm-significant reductions for CAA, assistant_axis, all 5 critical roles, and all 3 standalone residuals. Conformist roles: 0/3 — but note all four conformist roles produce *negative* mean Δ logit at their tune-locked coef on Qwen (peacekeeper −0.71, pacifist −2.98 [degraded], collaborator −0.03, facilitator −0.47), so the "conformist roles increase sycophancy" prediction does not survive on Qwen even before significance correction. The cross-seed std on the post-steer logit is small (≤ 0.25 for all significant conditions), indicating the single-seed numbers reported above are stable.

### Gemma residual parity note

Gemma's multi-seed aggregate (`experiment-main/results/multiseed_aggregate.json`) contains only the 21 original conditions — the 3 standalone residuals were added to the pipeline single-seed. The Gemma residual numbers in the table above are therefore from one seed-42 tune/test subsample. The Qwen residual numbers shown at the head of this report (from the same seed-123 subsample that results/ ends in after the multi-seed run) are consistent with the 3-seed aggregate in this panel (all three conditions Δlogit CI far from zero in every seed), so a re-run of Gemma multi-seed is unlikely to flip the core verdict — but it would produce comparable error bars.

## Pattern check vs. matched-coefficient decomposition finding

Claim under test: low-|cos(role, CAA)| residuals reduce sycophancy when steered as standalone directions; the high-|cos(role, CAA)| residual does not. This is the standalone-evaluation analogue of the matched-coefficient decomposition finding.

| Model | Low-cos residuals | All low-cos reduce syc? | High-cos residual | High-cos also reduces syc? | Ordering high-cos vs low-cos | Verdict |
|---|---|---|---|---|---|---|
| Gemma | contrarian (Δ=−0.32, ns), skeptic (Δ=−0.75) | partial: skeptic yes, contrarian fails Holm | collaborator (Δ=−0.92) | yes | high-cos > low-cos (collaborator > skeptic > contrarian) | **does NOT hold** — high-cos is the *strongest* reducer |
| Qwen  | contrarian (Δ=−1.65), skeptic (Δ=−1.80)       | yes | devils_advocate (Δ=−2.19) | yes | high-cos > low-cos (devils_advocate > skeptic > contrarian) | **does NOT hold** — high-cos is the *strongest* reducer |

### Interpretation

On **Gemma**, the high-cosine residual (`collaborator`, |cos|=0.165) reduces sycophancy at its own tune-locked coefficient (+5000) by Δlogit = −0.929, strongly significant after Holm. The matched-coefficient decomposition finding does not replicate when residuals are steered standalone. The low-cosine `contrarian` residual also fails to reach Holm significance (p=0.072) — the other half of the claim also breaks.

On **Qwen**, the highest-|cos(role, CAA)| role using the official `lu-christina/assistant-axis-vectors/qwen-3-32b` release at layer 32 is `devils_advocate` (|cos|=0.108); skeptic is second at 0.105. All three residuals significantly reduce sycophancy after Holm, each with a 95% paired-bootstrap CI that excludes zero. Using the **multi-seed paired Δ values** from the table above: `devils_advocate_residual` Δlogit = **−2.193 ± 0.181** is actually the LARGEST reducer of the three on Qwen — `skeptic_residual` (Δ=−1.799, |cos|=0.105) and `contrarian_residual` (Δ=−1.647, |cos|=0.071) both reduce sycophancy LESS, despite contrarian's lower cosine with CAA. So the expected low-cos > high-cos ordering not only fails to appear, the ordering is reversed: high-|cos| reduces *more*.

(A previous version of this paragraph quoted Δ values of −2.052 and −2.153; neither is reproducible from any available data file. The values above are computed from `qwen/results/multiseed_aggregate_test.json` per-seed `logits` minus the per-seed baselines from `clean-results/data/qwen3-32b_clean.json["baselines_per_seed"]`.)

Taken together, the *standalone* evaluation on both models does not support the matched-coefficient claim that the CAA-orthogonal residual of a high-|cos(CAA)| role fails to reduce sycophancy. Given each residual its own tune-locked coefficient, the high-cos residual reduces sycophancy at least as well as the low-cos residuals on both models.

### Methodological notes

1. **Vector source.** Both models use persona/assistant-axis vectors from `lu-christina/assistant-axis-vectors` (the generation-plus-LLM-judge pipeline from `safety-research/assistant-axis`). Gemma uses the `gemma-2-27b/` subdir at layer 22; Qwen uses the `qwen-3-32b/` subdir at layer 32 (the canonical target layer from `assistant_axis.models.MODEL_CONFIGS`).

2. **CAA extraction.** The authors don't ship a sycophancy CAA vector -- that's computed on top by both experiments using Rimsky et al. 2024: 1000 pairs from NLP-survey and 1000 from political-typology-quiz, contrasting last-token activations on (question + syc-answer) vs (question + honest-answer) at the target layer of each model.

3. **Coefficient rescaling.** Qwen's coefficient grid is 10× smaller than Gemma's (±50..±500 vs ±500..±5000). At layer 32 on Qwen, |coef|=5000 on a unit vector saturates the next-token distribution. Both experiments therefore span roughly the same dose-response regime relative to their own activation scale, but the saturation cliff lives at different absolute coefficient values in the two models.

4. **Qwen chat template.** Qwen 3's default chat template injects an empty `<think></think>` block when `enable_thinking=False`. The Qwen pipeline patches `build_prompt` in `02_evaluate_steering.py` to pass `enable_thinking=False` so the A/B logprob measurement is at the post-`</think>` position (where the model would normally emit its answer). Without this patch, the measured logits are at the `<think>`-start position and the sycophancy signal collapses.

5. **Baseline sycophancy rate differs between models.** Gemma 2 27B baseline is +1.0146 logit / 59.7% syc rate (cross-seed mean); Qwen 3 32B baseline is +3.000 logit / 84.0% (cross-seed mean). Qwen starts substantially more sycophantic on this benchmark, which amplifies the absolute magnitude of Δlogit that any given intervention can produce (Qwen has more room to reduce). Cross-model magnitude comparisons should be read relative to each model's baseline.

6. **Random control magnitudes differ accordingly.** A typical "random unit-Gaussian steering vector at any non-zero coef" produces Δlogit ≈ −0.25 on Gemma and Δlogit ≈ −1.06 on Qwen (averaged across 10 random vectors × 8 non-zero coefs per seed; see `clean-results/data/{model}_clean.json["conditions"]["random"]`). Qwen's random-floor reduction is substantial because *any* large-coefficient perturbation pushes Qwen's saturated baseline toward middle. When comparing role-vector reductions across models, divide each model's role Δ by its random-floor Δ for a more like-for-like ratio: e.g. Skeptic-above-random on Gemma = −0.71/−0.25 ≈ 2.8×; on Qwen = −1.82/−1.06 ≈ 1.7×.
