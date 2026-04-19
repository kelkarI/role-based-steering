# Results

All numbers on the held-out `sycophancy_on_philpapers2020` test split
(150 base questions × 2 orderings per seed), with coefficients locked
from the tune split. Gemma has 5 tune / 3 test seeds for the 21 main
conditions and 1 seed (42) for the 3 standalone residuals; Qwen has 5
tune / 3 test seeds for all 24 conditions. Holm–Bonferroni applied
across all 24 conditions per seed.

## 1. Baselines

| | Gemma 2 27B | Qwen 3 32B |
|---|---|---|
| Mean sycophancy logit (no steering) | +1.009 | +3.000 (mean across 3 test seeds; cross-seed σ=0.14) |
| Sycophancy rate | 59% | 84% |

Qwen starts much more sycophantic on this benchmark. Cross-model
comparisons of Δ logit should therefore be read as fractions of each
model's baseline, not as absolute amounts.

## 2. Main conditions (tune-locked best coef, test split)

Full per-model table in `paper/tables/main_results_both_models.{csv,md}`.
Selected rows:

| Model | Condition | Best coef | Δ logit | Δ rate pp | Significant seeds |
|---|---|---|---|---|---|
| Gemma | CAA | −2000 | −0.87 ± 0.01 | −9.0 | 3/3 |
| Gemma | Skeptic | +2000 | −0.71 ± 0.03 | −9.3 | 3/3 |
| Gemma | Assistant Axis | +2000 | −0.37 ± 0.03 | −4.3 | 3/3 |
| Gemma | Devil's Advocate | +2000 | −0.52 ± 0.03 | −8.3 | 3/3 |
| Gemma | Collaborator (conformist) | +500 | +0.05 ± 0.01 | +3.3 | 2/3 (sig raise) |
| Gemma | Pacifist (conformist) | +2000 | +0.11 ± 0.01 | +2.3 | 1/3 |
| Qwen  | CAA | −200 | −1.97 ± 0.04 | −19.0 | 3/3 |
| Qwen  | Skeptic | +200 | −1.82 ± 0.13 | −19.3 | 3/3 |
| Qwen  | Assistant Axis | +200 | −2.43 ± 0.08 | −21.3 | 3/3 |
| Qwen  | Devil's Advocate | +200 | −2.27 ± 0.07 | −16.0 | 3/3 |
| Qwen  | Collaborator (conformist) | −100 | −0.03 ± 0.15 | −1.0 | 0/3 |
| Qwen  | Pacifist (conformist) | +500 | −2.98 ± 0.01 | −34.0 (degraded: rate 50%) | 0/3 |

## 3. Finding 1 — "General persona matches targeted CAA"

| | Gemma | Qwen |
|---|---|---|
| CAA Δ logit | −0.87 | −1.97 |
| Skeptic Δ logit | −0.71 | −1.82 |
| Skeptic / CAA | 0.82 | 0.93 |
| As fraction of baseline | CAA 87%, Skeptic 71% | CAA 66%, Skeptic 61% |

The Skeptic persona direction — extracted from a contrastive
*generation* pipeline that has no access to A/B sycophancy labels —
matches a targeted CAA vector's reduction to within a factor of 0.93 on
Qwen and 0.82 on Gemma. This replicates the headline finding of the
Gemma study on a second model.

## 4. Finding 2 — "The reduction lives in the CAA-orthogonal residual" (matched-coefficient)

Matched-coefficient decomposition eval: for each role, steer at the
parent's best coef with only the residual, then only with only the
CAA-aligned component.

**Residual reduction at the parent's best coef (Δ logit vs baseline):**

| Role | Gemma residual Δ | Qwen residual Δ |
|---|---|---|
| skeptic | −0.75 | −1.75 |
| devils_advocate | −0.54 | −2.13 |
| contrarian | −0.33 | −2.32 |
| judge | −0.62 | −1.71 |
| scientist | −0.52 | −1.08 |
| assistant_axis | −0.27 | −2.46 |

Every critical role's residual alone recovers most of the parent role's
reduction on both models. See `{model}/results/decomposition_eval_test.json`.

**CAA-aligned component behaviour differs between models.** On Gemma,
most critical roles have `cos(role, CAA) > 0`, so the unit-normalised
CAA-component points *toward* sycophancy and *increases* logit at
positive coefficients (Δ ≈ +0.48 for devils_advocate, contrarian,
skeptic, judge, scientist). On Qwen, most have `cos(role, CAA) < 0`, so
the component points *away* from sycophancy and *decreases* logit
(Δ ≈ −2.0). The "aligned component opposes the role's reduction"
phrasing from the Gemma study does not replicate on Qwen — but it is
also not needed for the core finding about the residual.

## 5. Finding 3 — "Standalone residuals reduce sycophancy too"

Each residual evaluated as a fresh condition with its own tune-locked
coefficient, Holm across all 24 conditions:

| Residual | \|cos(role,CAA)\| | Gemma Δ logit [95% CI] | Holm p | Qwen Δ logit (3 seeds) | Sig seeds |
|---|---|---|---|---|---|
| Skeptic ⊥ CAA | G:0.064 / Q:0.105 | −0.75 [−0.96, −0.57] | 3.6e-11 ✱ | −1.80 ± 0.14 | 3/3 ✱ |
| Contrarian ⊥ CAA | G:0.033 / Q:0.071 | −0.32 [−0.55, −0.10] | 0.072 ns | −1.65 ± 0.12 | 3/3 ✱ |
| Collaborator ⊥ CAA (Gemma high-cos) | G:0.165 | −0.93 [−1.13, −0.76] | <1e-15 ✱ | — | — |
| Devil's Advocate ⊥ CAA (Qwen high-cos) | Q:0.108 | — | — | −2.19 ± 0.08 | 3/3 ✱ |

All residuals that reach Holm significance reduce sycophancy; on Qwen
the highest-\|cos\| residual (devils_advocate) produces the **largest**
reduction of the three, and on Gemma the highest-\|cos\| residual
(collaborator) also produces the largest reduction.

## 6. Finding 4 — "High-\|cos\| residual fails to reduce sycophancy" does NOT hold standalone

This is the claim the study set out to test. Under the matched-coef
decomposition, at the *role's* best coefficient, the residual of a
high-\|cos(role, CAA)\| role has a smaller marginal effect than a
low-\|cos\| role's residual (for Gemma collaborator 02c Δ ≈ −0.04 vs
skeptic 02c Δ ≈ −0.75; this is mostly because the locked coef was tuned
for the full role, not its residual).

When the residual gets its **own** tune-locked coefficient, this pattern
disappears on both models:

- **Gemma collaborator_residual** at locked +5000 → Δ = −0.93, larger
  than skeptic_residual's −0.75.
- **Qwen devils_advocate_residual** at locked +200 → Δ = −2.19, larger
  than skeptic_residual's −1.80 and contrarian_residual's −1.65.

Pattern verdict on both models: **does not hold**.

## 7. Bidirectionality (conformist roles increase sycophancy)

| | Gemma | Qwen |
|---|---|---|
| Conformist roles with Holm-sig INCREASE at increase-selector best coef | 1 / 4 (Collaborator, Δ=+0.05, p_adj=0.002) | 0 / 4 across 3 seeds |

On Qwen, the baseline sycophancy rate is 84% — there is little headroom
for conformist directions to push the model *more* sycophantic before
saturation. Pacifist at +500 on Qwen drives the rate to exactly 50%, a
degradation signature (logit ≈ 0, near-random between A and B). The
bidirectionality result should therefore be read as "holds weakly on
Gemma, not tested cleanly on Qwen due to ceiling effect."

## 8. Figure inventory

Per-model figures from each model's `03_analysis.py`:

- `fig1_steering_curves.png`: coefficient sweep curves per condition
  (rate and logit)
- `fig4_best_comparison.png`: bar chart of best Δ logit per condition
- `fig5_vector_similarities.png`: cosine heat-map between steering
  vectors
- `fig6_persona_space_pca.png`: PCA of the persona-space + CAA + random
- `fig7_residual_standalone_vs_parent.png`: coefficient-sweep overlay
  (residual, parent role, CAA) — the headline figure for the
  standalone-residual result

Cross-model:

- `cross_model/residual_standalone_gemma_vs_qwen.md` — narrative with
  side-by-side tables.

## 9. Full result tables

- `paper/tables/main_results_both_models.csv` — one row per (model,
  condition); includes best coef, baseline logit/rate, Δ logit, Δ rate,
  cross-seed std where multi-seed, single-seed Holm p, `cos(role, CAA)`.
- `{model}/results/summary_test.txt` — human-readable per-model summary
  from `03_analysis.py`.
- `{model}/results/statistical_tests_test.json` — primary Wilcoxon +
  Holm, McNemar, dose-response Spearman, axis-projection Pearson, all
  per-condition.
- `{model}/results/over_correction_eval_test.json` — 16 probes × 12
  conditions classified output.
- `{model}/results/decomposition_eval_test.json` — matched-coefficient
  decomposition eval, 20 (condition × component) pairs.
