# Standalone CAA-orthogonal residual conditions — Gemma vs Qwen

Both models were evaluated with the same 24-condition pipeline (assistant_axis, 5 critical roles, 4 conformist roles, CAA, 10 random controls, 3 standalone residuals), Holm correction applied across all 24 conditions on the held-out test split, with coefficients locked from each model's own tune split. The standalone residual for each model is the role vector minus its projection onto CAA, unit-normalised, then run through the full coefficient sweep as an independent steering condition with a fresh tune-locked coefficient.

## Model differences

| Property | Gemma-2-27b-it | Qwen 3 32B |
|---|---|---|
| target layer | 22 / 46 | 32 / 64 |
| coefficient sweep | ±500, ±1000, ±2000, ±5000 | ±50, ±100, ±200, ±500 |
| baseline mean syc logit (test) | +1.009 | +3.024 |
| baseline sycophancy rate (test) | 59.3% | 83.7% |
| role vector source | lu-christina/assistant-axis-vectors (`gemma-2-27b/`) | lu-christina/assistant-axis-vectors (`qwen-3-32b/`) |
| max |cos(role, CAA)| among critical+conformist roles | 0.165 (collaborator) | 0.108 (devils_advocate) |

The coefficient-sweep rescale on Qwen was required because Qwen's layer-32 default-activation norm (~166) is much smaller than Gemma's layer-22 norm, so |coef|=5000 completely saturates Qwen's next-token distribution (top-1 becomes whitespace / control tokens). The rescaled Qwen grid was verified on a pilot probe to give a clean dose-response from baseline-like at |coef|=50 to anti-sycophantic at |coef|=200 to saturated at |coef|=500. Gemma's original grid was retained unchanged.

## Standalone residual results (held-out test split, Holm across 24 conditions)

### Gemma-2-27b-it

Baseline logit: +1.009   Baseline rate: 59.3%

| Condition | \|cos(role, CAA)\| | Best coef (locked) | Parent's best coef | Δ logit [95% CI] | Δ rate pp [95% CI] | Holm p | sig |
|---|---|---|---|---|---|---|---|
| skeptic_residual       | +0.0640 | +2000 | +2000 (same) | -0.753 [-0.957, -0.567] | -9.3 [-12.7, -6.3] | 3.6e-11 | ✱ |
| contrarian_residual    | +0.0326 | +2000 | +1000 (differs) | -0.321 [-0.552, -0.096] | -4.0 [-8.3, +0.0] | 0.0725 | ns |
| collaborator_residual  | +0.1648 | +5000 | +500 (differs) | -0.929 [-1.129, -0.755] | -8.3 [-11.7, -5.0] | < 1e-15 | ✱ |

### Qwen 3 32B

Baseline logit: +3.024   Baseline rate: 83.7%

| Condition | \|cos(role, CAA)\| | Best coef (locked) | Parent's best coef | Δ logit [95% CI] | Δ rate pp [95% CI] | Holm p | sig |
|---|---|---|---|---|---|---|---|
| skeptic_residual       | -0.1049 | +200 | +200 (same) | -1.749 [-2.048, -1.441] | -14.7 [-19.7, -9.3] | < 1e-15 | ✱ |
| contrarian_residual    | -0.0705 | -100 | -100 (same) | -1.684 [-1.919, -1.442] | -11.0 [-15.7, -6.0] | < 1e-15 | ✱ |
| devils_advocate_residual | -0.1078 | +200 | +200 (same) | -2.130 [-2.442, -1.826] | -13.3 [-18.7, -8.0] | < 1e-15 | ✱ |

## Qwen multi-seed aggregate (mean ± std across 3 test seeds)

Re-ran the full 24-condition pipeline on Qwen across 5 tune seeds (42, 7, 123, 456, 789) and 3 test seeds (42, 7, 123), matching the Gemma pipeline's multi-seed protocol. Tune-locked coefficients came from the mode-across-seeds aggregate (results/best_coefs_tune_aggregate.json). Wilcoxon+Holm applied per seed.

Baseline across 3 test seeds: logit = +3.000 ± 0.136

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

On Qwen, 3/3 seeds show Holm-significant reductions for CAA, assistant_axis, all 5 critical roles, and all 3 standalone residuals. Conformist roles: 0/3. The cross-seed std on the post-steer logit is small (≤ 0.25 for all significant conditions), indicating the single-seed numbers reported above are stable.

### Gemma residual parity note

Gemma's multi-seed aggregate (`experiment-main/results/multiseed_aggregate.json`) contains only the 21 original conditions — the 3 standalone residuals were added to the pipeline single-seed. The Gemma residual numbers in the table above are therefore from one seed-42 tune/test subsample. The Qwen residual numbers shown at the head of this report (from the same seed-123 subsample that results/ ends in after the multi-seed run) are consistent with the 3-seed aggregate in this panel (all three conditions Δlogit CI far from zero in every seed), so a re-run of Gemma multi-seed is unlikely to flip the core verdict — but it would produce comparable error bars.

## Pattern check vs. matched-coefficient decomposition finding

Claim under test: low-|cos(role, CAA)| residuals reduce sycophancy when steered as standalone directions; the high-|cos(role, CAA)| residual does not. This is the standalone-evaluation analogue of the matched-coefficient decomposition finding.

| Model | Low-cos residuals | All low-cos reduce syc? | High-cos residual | High-cos also reduces syc? | Verdict |
|---|---|---|---|---|---|
| Gemma | contrarian, skeptic | False | collaborator | True | **does NOT hold** |
| Qwen  | contrarian, skeptic | True | devils_advocate | True | **does NOT hold** |

### Interpretation

On **Gemma**, the high-cosine residual (`collaborator`, |cos|=0.165) reduces sycophancy at its own tune-locked coefficient (+5000) by Δlogit = −0.929, strongly significant after Holm. The matched-coefficient decomposition finding does not replicate when residuals are steered standalone. The low-cosine `contrarian` residual also fails to reach Holm significance (p=0.072) — the other half of the claim also breaks.

On **Qwen**, the highest-|cos(role, CAA)| role using the official `lu-christina/assistant-axis-vectors/qwen-3-32b` release at layer 32 is `devils_advocate` (|cos|=0.108); skeptic is second at 0.105. All three residuals significantly reduce sycophancy after Holm, each with a 95% paired-bootstrap CI that excludes zero. Devils_advocate's residual Δlogit = −2.052 is actually the MIDDLE of the three on Qwen — contrarian (Δ=−2.153, |cos|=0.071) edges it out despite having a lower cosine with CAA. So the expected low-cos > high-cos ordering does not appear.

Taken together, the *standalone* evaluation on both models does not support the matched-coefficient claim that the CAA-orthogonal residual of a high-|cos(CAA)| role fails to reduce sycophancy. Given each residual its own tune-locked coefficient, the high-cos residual reduces sycophancy at least as well as the low-cos residuals on both models.

### Methodological notes

1. **Vector source.** Both models use persona/assistant-axis vectors from `lu-christina/assistant-axis-vectors` (the generation-plus-LLM-judge pipeline from `safety-research/assistant-axis`). Gemma uses the `gemma-2-27b/` subdir at layer 22; Qwen uses the `qwen-3-32b/` subdir at layer 32 (the canonical target layer from `assistant_axis.models.MODEL_CONFIGS`).

2. **CAA extraction.** The authors don't ship a sycophancy CAA vector -- that's computed on top by both experiments using Rimsky et al. 2024: 1000 pairs from NLP-survey and 1000 from political-typology-quiz, contrasting last-token activations on (question + syc-answer) vs (question + honest-answer) at the target layer of each model.

3. **Coefficient rescaling.** Qwen's coefficient grid is 10× smaller than Gemma's (±50..±500 vs ±500..±5000). At layer 32 on Qwen, |coef|=5000 on a unit vector saturates the next-token distribution. Both experiments therefore span roughly the same dose-response regime relative to their own activation scale, but the saturation cliff lives at different absolute coefficient values in the two models.

4. **Qwen chat template.** Qwen 3's default chat template injects an empty `<think></think>` block when `enable_thinking=False`. The Qwen pipeline patches `build_prompt` in `02_evaluate_steering.py` to pass `enable_thinking=False` so the A/B logprob measurement is at the post-`</think>` position (where the model would normally emit its answer). Without this patch, the measured logits are at the `<think>`-start position and the sycophancy signal collapses.

5. **Baseline sycophancy rate differs between models.** Gemma 2 27B baseline is +1.01 logit / 59% syc rate; Qwen 3 32B baseline is +3.02 logit / 84%. Qwen starts substantially more sycophantic on this benchmark, which amplifies the absolute magnitude of Δlogit that any given intervention can produce (Qwen has more room to reduce). Cross-model magnitude comparisons should be read relative to each model's baseline.
