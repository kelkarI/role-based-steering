# Results

All numbers on the held-out `sycophancy_on_philpapers2020` test split
(150 base questions × 2 orderings per seed), with coefficients locked
from the tune split. Gemma has 5 tune / 3 test seeds for the 21 main
conditions and 1 seed (42) for the 3 standalone residuals; Qwen has 5
tune / 3 test seeds for all 24 conditions. **Holm–Bonferroni is applied
per-seed across the 14-condition primary family (11 main + 3 standalone
residuals)** — see `gemma/results/_metadata.json` (`significance.holm_family_size`)
and `AUDIT_NOTES.md`. A previous version of this doc said "across all 24
conditions"; that was incorrect — the 10 random controls are not in the
Holm family.

**Baseline convention.** Every Δ logit and Δ rate below uses the
**cross-seed mean baseline** and **per-seed paired Δ** (each seed's
post-steer value minus *that seed's* baseline, then averaged). Gemma:
baseline = +1.0146 (cross-seed mean of [+1.0095, +1.0038, +1.0305]).
Qwen: baseline = +3.000 (cross-seed mean of [+3.024, +2.822, +3.153]).
A previous version of this doc and the headline CSV used a single-seed
baseline (+1.009 / +3.024); see `AUDIT_NOTES.md` §2.

## 1. Baselines

| | Gemma 2 27B | Qwen 3 32B |
|---|---|---|
| Mean sycophancy logit (no steering) | +1.0146 (cross-seed mean of [+1.0095, +1.0038, +1.0305]) | +3.000 (cross-seed mean of [+3.024, +2.822, +3.153]; σ=0.14) |
| Sycophancy rate | 59.7% | 84.0% |

Qwen starts much more sycophantic on this benchmark. Cross-model
comparisons of Δ logit should therefore be read as fractions of each
model's baseline, not as absolute amounts.

## 2. Main conditions (tune-locked best coef, test split)

Full per-model table in `paper/tables/main_results_both_models.{csv,md}`.
Selected rows:

All Δ values are **per-seed paired** (mean and std across 3 test seeds).
"± x" is the std of the per-seed Δ, NOT the std of the post-steer logit.

| Model | Condition | Best coef | Δ logit | Δ rate pp | Significant seeds |
|---|---|---|---|---|---|
| Gemma | CAA | −2000 | −0.879 ± 0.001 | −8.89 | 3/3 |
| Gemma | Skeptic | +2000 | −0.711 ± 0.013 | −9.56 | 3/3 |
| Gemma | Assistant Axis | +2000 | −0.375 ± 0.018 | −5.00 | 3/3 |
| Gemma | Devil's Advocate | +2000 | −0.521 ± 0.016 | −8.67 | 3/3 |
| Gemma | Collaborator (conformist) | +500 | +0.045 ± 0.006 | +1.67 | 2/3 (sig raise) |
| Gemma | Pacifist (conformist) | +2000 | +0.100 ± 0.001 | +0.33 | 1/3 |
| Gemma | Facilitator (conformist) | −5000 | **−0.727 ± 0.008** | −9.22 | 0/3 (Holm) |
| Qwen  | CAA | −200 | −1.965 ± 0.126 | −20.89 | 3/3 |
| Qwen  | Skeptic | +200 | −1.823 ± 0.058 | −18.11 | 3/3 |
| Qwen  | Assistant Axis | +200 | −2.410 ± 0.092 | −18.56 | 3/3 |
| Qwen  | Devil's Advocate | +200 | −2.272 ± 0.195 | −16.56 | 3/3 |
| Qwen  | Collaborator (conformist) | −100 | −0.029 ± 0.016 | −1.67 | 0/3 |
| Qwen  | Facilitator (conformist) | −200 | **−0.469 ± 0.099** | −0.56 | 0/3 (Holm) |
| Qwen  | Pacifist (conformist) | +500 | −2.979 ± 0.174 | −34.00 ‡ (degraded: rate locked at 0.5) | 0/3 |

**Note on `facilitator`.** Although grouped with the conformist family,
its tune-locked best coefficient is *negative* (−5000 on Gemma, −200 on
Qwen) and the resulting Δ logit *reduces* sycophancy on both models.
This is a genuine counter-example to the simple "conformist roles
increase sycophancy" framing in §7. Neither facilitator effect reaches
Holm significance per-seed (raw Wilcoxon does not survive the 14-test
correction; see §7 for the family-direction discussion). The bottom-line
verdict in §7 has been updated to reflect this.

## 3. Finding 1 — "General persona matches targeted CAA"

| | Gemma | Qwen |
|---|---|---|
| CAA Δ logit | −0.879 | −1.965 |
| Skeptic Δ logit | −0.711 | −1.823 |
| Skeptic / CAA | 0.81 | 0.93 |
| As fraction of baseline | CAA 87%, Skeptic 70% | CAA 66%, Skeptic 61% |

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

2 of 3 Gemma residuals reach Holm significance. **Contrarian ⊥ CAA on
Gemma is explicitly a null: raw Wilcoxon p=0.020, Holm-adjusted
p=0.072** over 24 conditions on n=150 test bases (single seed). Its
95% CI excludes zero at −0.32 [−0.55, −0.10] but the adjusted p does
not pass α=0.05. All 3 Qwen residuals reach Holm significance in all 3
test seeds.

Where residuals *are* significant: on Qwen the highest-\|cos\| residual
(devils_advocate) produces the largest reduction of the three; on
Gemma the highest-\|cos\| residual (collaborator) also produces the
largest reduction. See the high-cos-range caveat in LIMITATIONS §6b —
on Qwen the three roles are barely separated in \|cos\| (top gap 0.003)
so this ordering is a nominal one, not a geometric outlier contrast.

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
| Conformist roles whose locked Δ logit is +ve (i.e. the role increases syc) | 2 / 4 (Pacifist Δ=+0.100, Collaborator Δ=+0.045) | 0 / 4 |
| Conformist roles with Holm-sig INCREASE at locked coef | 1 / 4 (Collaborator, Δ=+0.045, p_adj=0.002) | 0 / 4 across 3 seeds |
| Conformist roles whose locked Δ logit is −ve (i.e. the role REDUCES syc) | 2 / 4 (Peacekeeper Δ=−0.052, Facilitator Δ=−0.727) | 4 / 4 (all four) |

On Qwen, the baseline sycophancy rate is 84% — there is little headroom
for conformist directions to push the model *more* sycophantic before
saturation. Pacifist at +500 on Qwen drives the rate to exactly 50%, a
degradation signature (logit ≈ 0, near-random between A and B). The
bidirectionality result should therefore be read as **"holds in 1/4
of Gemma conformist roles (Collaborator only); the other 3/4 either
stay flat (Peacekeeper, Pacifist within noise) or actively *reduce*
sycophancy (Facilitator). On Qwen all 4 conformist roles either reduce
sycophancy at their tune-locked coefficient (Peacekeeper, Collaborator,
Facilitator) or are degraded (Pacifist), so the prediction does not
hold there. The single-vector "conformist family pushes toward
sycophancy" framing should be retired in favour of per-role
characterisation."**

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
