# Methods

## Models

| | Gemma 2 27B | Qwen 3 32B |
|---|---|---|
| HF path | `google/gemma-2-27b-it` | `Qwen/Qwen3-32B` |
| Decoder layers | 46 | 64 |
| Hidden size | 4608 | 5120 |
| Target steering layer | 22 | 32 |
| Single-token A/B IDs | 235280 / 235305 | 32 / 33 |
| Chat-template special | none | `enable_thinking=False` so the A/B logit is measured at the post-`</think>` position |
| Precision | bfloat16 | bfloat16 |

Target layers come from the `assistant_axis.models.MODEL_CONFIGS` entries
maintained by the `safety-research/assistant-axis` authors. In both cases
they are approximately mid-stack.

## Steering mechanism

All steering uses
`assistant_axis.steering.ActivationSteering` in `addition` mode, applied
at every token position of the prompt + assistant template. The hook adds
`coef × steering_vector` to the residual stream at the target layer's
output. Each steering vector is unit-normalised before use; coefficients
are the scalar multiplier.

## Conditions (24 per model)

- **Primary (2):** `assistant_axis` (full personas-space axis) and `caa`
  (Rimsky direction; negative coefficients reduce sycophancy because CAA
  points *from* honest *toward* sycophantic).
- **Critical roles (5):** `devils_advocate`, `contrarian`, `skeptic`,
  `judge`, `scientist`. Expected direction: decrease.
- **Conformist roles (4):** `peacekeeper`, `pacifist`, `collaborator`,
  `facilitator`. Expected direction: increase (bidirectionality check).
- **Random controls (10):** unit-Gaussian vectors, seeded
  `RANDOM_SEED_BASE + i` for `i ∈ [0, 10)`.
- **Standalone CAA-orthogonal residuals (3):** per-model, the three roles
  with the two lowest \|cos(role, CAA)\| (skeptic, contrarian) and the
  single highest \|cos(role, CAA)\| (Gemma → collaborator 0.165; Qwen →
  devils_advocate 0.108). Residual = `role - proj_CAA(role)`,
  unit-normalised. Expected direction: decrease (the hypothesis being
  tested is whether the residual *reduces* sycophancy, not whether it
  inherits the parent role's direction).

The condition naming and `EXPECTED_DIRECTION` dict live in each repo's
`scripts/config.py`.

## Vector provenance

- **Role vectors, default_vector, assistant_axis:** downloaded from
  `lu-christina/assistant-axis-vectors` on HuggingFace, using the
  `gemma-2-27b/` and `qwen-3-32b/` subdirs respectively. These are
  produced by the generation-plus-LLM-judge pipeline in
  `safety-research/assistant-axis` (see its
  `pipeline/{1_generate,2_activations,3_judge,4_vectors,5_axis}.py`). For
  a role `R`, `role_direction = role_vector[L] - default_vector[L]`,
  unit-normalised. For the axis, `axis_direction = axis[L]`
  unit-normalised.
- **CAA vector:** not shipped by the authors; extracted locally per-model
  using the Rimsky et al. 2024 method at the target layer, on 2000 A/B
  pairs from `sycophancy_on_nlp_survey` + `sycophancy_on_political_typology_quiz`
  (disjoint from the philpapers2020 eval set). Direction = mean(syc
  activation) − mean(honest activation); unit-normalised. Metadata in
  `{model}/caa_metadata.json`.
- **Residuals / CAA-aligned components:** computed from role × CAA via
  `v_residual = v_role - (v_role · v_CAA) v_CAA`, unit-normalised; symmetric
  construction for the aligned component.
- **Random controls:** `torch.randn` at seed `RANDOM_SEED_BASE + i`,
  unit-normalised.

## Eval and splits

- **Eval set:** `sycophancy_on_philpapers2020.jsonl` from `anthropics/evals`
  (commit blob `5525210614d4f26b1732042e7dcb7210d23fe5aa`), 300 base
  questions × 2 A/B orderings (counterbalanced) = 600 rows per seed.
- **Tune / test split:** 50 / 50 at base-question level
  (`TUNE_TEST_SEED = 99`); pairs stay together.
- **Sampling seeds:** Gemma tune 5 (42, 7, 123, 456, 789), test 3
  (42, 7, 123). Qwen tune 5 (42, 7, 123, 456, 789), test 3 (42, 7, 123).
  Standalone-residual conditions on Gemma are single-seed (seed 42 only)
  because they were added after the Gemma multi-seed run; on Qwen they
  are in the 3-seed aggregate.

## Coefficient sweep

9 points per model, symmetric around zero. The exact values differ
because Qwen's layer-32 activation norm is ~10× smaller than Gemma's at
layer 22, so the same absolute coefficient produces proportionally
stronger interventions:

- **Gemma:** `[-5000, -2000, -1000, -500, 0, +500, +1000, +2000, +5000]`
- **Qwen:**  `[-500, -200, -100, -50, 0, +50, +100, +200, +500]`

Pilot probes verified that on Qwen |coef|=5000 on a unit vector saturates
the next-token distribution (top-1 becomes whitespace / control tokens)
while |coef|≤200 keeps the answer distribution interpretable.

## Best-coefficient selection

For each condition, the direction-aware selector picks the coefficient
that maximises **excess over the random-mean logit at that coefficient**:
- `decrease` conditions (axis, critical roles, CAA, residuals): pick the
  coef with the largest positive excess (`rand_mean - cond_logit`);
- `increase` conditions (conformist roles): pick the largest negative
  excess;
- `unsigned` conditions (random controls): pick largest `|excess|`.

The selector runs per tune seed. Across-seed aggregation uses mode (with
tie-break by count then proximity to median); the aggregate is written
to `best_coefs_tune_aggregate.json` and is the input to the test run via
`--locked-coefs-from`.

## Metrics

- **Binary sycophancy rate:** fraction of rows where `argmax(lp_A, lp_B)`
  matches `sycophantic_answer`.
- **Sycophancy logit:**
  `syc_logit = logp(sycophantic_token) - logp(honest_token)`
  at the last prompt token, measured at coef=0 (baseline) and at each
  sweep coefficient.

## Statistical tests

- **Primary:** paired one-sided Wilcoxon signed-rank on base-level
  `syc_logit` differences (steered − baseline), per condition. Direction
  matches `EXPECTED_DIRECTION`; `less` for decrease, `greater` for
  increase, two-sided for unsigned.
- **Multiple-comparison correction:** Holm–Bonferroni across all 24
  conditions (the full family, including random controls).
- **Secondary:** McNemar on the 2×2 table of per-row correct/incorrect
  under steered vs baseline (exact test when the minority off-diagonal
  cell is < 5).
- **95% CI:** bootstrap on the mean of per-base paired differences
  (steered − baseline), 2000 iterations, percentile method.

## Degradation flag

A (condition, coefficient) cell is flagged degraded when both
`|rate - 0.5| < 0.03` **and** `|cond_logit - random_mean_logit(coef)| < 0.10`
— i.e. the model is near-uniform on A/B and its logit is
indistinguishable from what a random steering vector produces at the
same magnitude. Degraded cells are excluded from the "most effective
reducer" summary but are still reported in every table (e.g. Qwen
pacifist at +500 is degraded).

## Auxiliary analyses (test split)

- **Matched-coefficient decomposition** (`02c_evaluate_decomposition.py`):
  for each real condition, steer at its parent's best coef with (a) the
  unit-normalised CAA-aligned component and (b) the unit-normalised
  residual, separately; compare to the parent's effect.
- **Over-correction taxonomy** (`over_correction_check.py`): 16
  high-sycophancy probe prompts × 12 conditions at each condition's
  locked coef; classify each 150-token generation into
  {AGREE_CORRECT, DISAGREE_CORRECT, AGREE_WRONG, DISAGREE_WRONG, HEDGE,
  REFUSE} by keyword rules.
- **(Gemma only) Response-token projection** (`02d_response_capture.py`):
  mean axis-dot projection over 8 greedy response tokens. Not run on
  Qwen in this study.

## Software versions

Both pipelines run on `torch 2.7`, `transformers 5.5`,
`accelerate 1.13`, `statsmodels 0.14`. The `assistant_axis` library is
the HEAD of `safety-research/assistant-axis` at session time; the plotly
dependency is required by its PCA module.
