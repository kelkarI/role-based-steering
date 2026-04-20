# Role-based activation steering reduces LLM sycophancy as well as a targeted direction, and the effect lives in the CAA-orthogonal subspace — a two-model study

**Target venue:** workshop / short paper. Approximately 8 pages.
**Author list / affiliation:** to be filled in by authors.

---

## Abstract

**Sycophancy** in large language models (LLMs) — the tendency to agree with
a user's stated position rather than with truth or evidence — can be
reduced by **activation steering**, i.e. adding a fixed vector to the
model's residual stream at inference time. The established targeted
method, **Contrastive Activation Addition (CAA; Rimsky et al. 2024)**,
extracts this vector from sycophancy-labelled A/B preference data. We
ask two questions. **(i)** Can a *general* persona direction (e.g. the
mean-activation contrast for "you are a skeptic") — extracted without
any sycophancy labels — reduce sycophancy about as well as the targeted
CAA vector? **(ii)** Where does that reduction live geometrically:
along CAA itself, in the subspace orthogonal to CAA, or both? We answer
both on **Gemma 2 27B** and **Qwen 3 32B**, running 24 steering
conditions across 5 tune / 3 test seeds per model on held-out
`sycophancy_on_philpapers2020` A/B preferences, with Holm–Bonferroni
correction across all conditions. The **Skeptic persona direction
achieves 82 % (Gemma) and 93 % (Qwen) of CAA's sycophancy reduction** in
Δlogit terms. When we decompose each role vector into its component
along CAA and its residual, **the residual alone recovers the role's
effect** at matched coefficient on both models. Evaluating three
standalone residuals with their own tune-locked coefficients, 5 of 6
cells (model × residual) reach Holm-significance; the single exception
is one Gemma residual whose CI excludes zero but whose adjusted p is
0.072. A pre-registered sub-claim — that the residual of a
*high*-|cos(role, CAA)| role would fail to reduce sycophancy — **does
not replicate** on either model, and we note that the "high-cos" role
is not a geometric outlier on Qwen. All claims are limited by two
assumptions: the sycophancy-reducing direction we measure is only
rank-one at a single layer, and our bidirectionality probe (conformist
roles should *increase* sycophancy) is ceiling-constrained on Qwen.

---

## 1 Introduction

**Sycophancy** — answering questions the way the user wants them
answered, independently of what is actually true — is a named failure
mode of modern instruction-tuned language models [Perez+ 2023]. It is a
concrete case of the broader worry that optimisation against human
feedback can push models toward *pleasing* rather than *being correct*.
A mechanical way to attack sycophancy without re-training is to
**steer** the model's internal activations at inference time: add a
fixed vector `v` to the residual stream at some layer, scaled by a
coefficient `c`, so that `h' = h + c·v`. With the right `v` and `c`
the model's next-token distribution shifts away from the sycophantic
answer.

Rimsky et al. 2024's **Contrastive Activation Addition (CAA)** method
finds a good `v` directly: it averages hidden states on
sycophancy-labelled A/B preference questions for the sycophantic
continuation versus the honest continuation, and takes the difference.
That is a *targeted* direction — it requires labelled sycophancy data.

A natural counter-hypothesis is that sycophancy-reducing behaviour is
not a narrow, sycophancy-specific subspace, but rather an aspect of
general **persona** variation in the model. Instruction-tuned models
already carry coarse persona representations, and persona-steering
vectors for roles like "skeptic", "devil's advocate", "scientist",
and "judge" can be extracted by the same contrastive-activation
template using only role-labelled data, with no sycophancy labels at
all. If those persona directions reduce sycophancy too, the effect
lives in a more general subspace than CAA's.

This paper tests that counter-hypothesis on two instruction-tuned
decoder-only LLMs — **Gemma 2 27B** and **Qwen 3 32B** — using the
same 24-condition protocol on both. Our contributions:

1. **A clean replication of the "persona matches targeted CAA" finding
   on two models.** On both models, a single persona direction
   (`Skeptic`) achieves a large fraction of CAA's sycophancy
   reduction without being trained on sycophancy labels.
2. **A behavioural decomposition.** Projecting each role vector onto
   CAA and steering only with the residual (the CAA-orthogonal piece)
   recovers the role's effect on both models.
3. **An honest null.** A specific secondary claim — that a
   high-|cos(role, CAA)| role's residual should *fail* to reduce
   sycophancy — does not replicate on either model under standalone
   evaluation; we report this as a null and discuss why the contrast
   is blunter on Qwen than on Gemma.

Our code, aggregated results, and figures are public at
`kelkarI/role-based-steering` (central), with the per-model pipelines
at `kelkarI/sycophancy-gemma` (Gemma; formerly `kelkarI/sycophancy-final`) and `kelkarI/sycophancy-qwen`
(Qwen).

---

## 2 Background and terminology (for a general ML audience)

We define the small number of specialised terms we use. Everything
else is standard.

- **Residual stream.** In a transformer, each token's hidden state at
  layer `L` is a vector `h ∈ ℝ^d`. Transformer blocks write additively
  into this stream. Activation steering intervenes here: `h ← h + c·v`.
- **Steering vector.** A unit-length vector `v ∈ ℝ^d` used for
  steering. We always unit-normalise before steering, so that a
  coefficient `c` has a uniform meaning across conditions.
- **Coefficient (`c`).** A real scalar. Positive `c` pushes the
  residual in the `+v` direction, negative `c` in the `−v` direction.
- **Target layer.** A single layer at which the steering is applied.
  We use the canonical per-model layer chosen by the `assistant-axis`
  authors: L=22 of 46 for Gemma 2 27B, L=32 of 64 for Qwen 3 32B,
  both roughly mid-stack (see `assistant_axis/models.py`).
- **Role vector / persona direction.** The difference between the mean
  hidden state when the model is prompted with a role system prompt
  (e.g. "You are a skeptic who critically examines claims") and the
  mean hidden state under a default "helpful assistant" prompt. Taken
  at the target layer, unit-normalised. We use the directions released
  by `lu-christina/assistant-axis-vectors` (produced by the
  `safety-research/assistant-axis` pipeline: ~1,200 generations per
  role, LLM-judge filtered to in-character responses, then averaged).
- **CAA vector.** A unit-normalised vector equal to
  `mean(activations on sycophantic answers) −
  mean(activations on honest answers)`, following Rimsky et al. 2024.
  We extract CAA ourselves per model from 2,000 A/B pairs from
  `sycophancy_on_nlp_survey` and `sycophancy_on_political_typology_quiz`
  (1,000 per dataset, seed 2024; the authors do not ship a
  sycophancy CAA).
- **Sycophancy logit.** For an A/B preference question with a
  sycophantic answer `a_syc` and an honest answer `a_hon`,
  `syc_logit = logp(a_syc_token) − logp(a_hon_token)` measured at the
  last prompt token. Higher = more sycophantic.
- **Sycophancy rate.** Fraction of rows for which the argmax of
  `(logp(A), logp(B))` matches the sycophantic label.
- **Critical / conformist roles.** Five critical roles
  (`devils_advocate, contrarian, skeptic, judge, scientist`) and four
  conformist roles (`peacekeeper, pacifist, collaborator,
  facilitator`) chosen for the expectation that they should
  respectively *decrease* and *increase* sycophancy, giving us a
  bidirectionality test.
- **Residual (CAA-orthogonal component of a role).** Given a role
  vector `v_role` and the unit CAA vector `v_CAA`,
  `v_residual = v_role − (v_role · v_CAA) v_CAA`, then
  unit-normalised. This is the part of the role that is orthogonal to
  the sycophancy direction.
- **Holm–Bonferroni.** A multiple-comparison correction that controls
  the family-wise error rate. We apply it across all 24 conditions per
  seed, so raw p-values get more conservative as more conditions are
  considered simultaneously.
- **Tune / test split.** 300 base questions split 50/50 at the
  question level, seed 99; pairs (A-first, B-first orderings) stay
  together. Tune is used for coefficient selection; test is held out
  for the reported numbers.

---

## 3 Methods

### 3.1 Models

| | Gemma 2 27B | Qwen 3 32B |
|---|---|---|
| HuggingFace ID | `google/gemma-2-27b-it` | `Qwen/Qwen3-32B` |
| Layers / hidden size | 46 / 4608 | 64 / 5120 |
| Target steering layer | 22 | 32 |
| Single-token A/B IDs | 235280 / 235305 | 32 / 33 |

Both models are instruction-tuned decoder-only transformers. All
inference is in bfloat16 on a single GH200. Qwen 3's chat template by
default injects a `<think>...</think>` block before the assistant
turn; we pass `enable_thinking=False` so that the A/B logit is
measured at the post-`</think>` token (without this patch, the logit
is at the `<think>`-start token and the sycophancy signal collapses).

### 3.2 Benchmark and split

- **Eval set.** `sycophancy_on_philpapers2020.jsonl` from
  `anthropics/evals` (commit blob
  `5525210614d4f26b1732042e7dcb7210d23fe5aa`), 300 base questions × 2
  A/B orderings = 600 rows per seed.
- **Tune/test split.** 150 base questions each, split at the
  base-question level with seed 99, so each pair (original, swapped)
  stays within one half.
- **Sampling seeds.** 5 tune seeds {42, 7, 123, 456, 789}, 3 test
  seeds {42, 7, 123}. Test seeds are a subset of tune seeds. Each
  seed re-subsamples 300 base questions; the tune/test split seed is
  held fixed at 99.

### 3.3 Conditions (24 per model)

- **Primary (2).** `assistant_axis` and `caa`.
- **Critical roles (5).** `devils_advocate, contrarian, skeptic,
  judge, scientist`.
- **Conformist roles (4).** `peacekeeper, pacifist, collaborator,
  facilitator`.
- **Random controls (10).** Unit-Gaussian vectors, seed
  `RANDOM_SEED_BASE + i` for `i ∈ [0,10)`.
- **Standalone CAA-orthogonal residuals (3).** `skeptic_residual`,
  `contrarian_residual`, and the residual of the role with the
  largest |cos(role, CAA)| in that model's decomposition:
  `collaborator_residual` on Gemma, `devils_advocate_residual` on
  Qwen.

### 3.4 Coefficient sweep

Nine points per model, symmetric around zero. The two grids differ by
a factor of ten because the two models' activation norms at the
target layer differ by that order of magnitude:

- **Gemma:** {−5000, −2000, −1000, −500, 0, +500, +1000, +2000, +5000}
- **Qwen:**  {−500, −200, −100, −50, 0, +50, +100, +200, +500}

Concretely: the Gemma CAA raw norm (before unit normalisation) is
**740.85**; the Qwen CAA raw norm is **6.38** (see
`{model}/caa_metadata.json`) — a ~116× gap. A pilot probe on Qwen
verified that |c|=5000 on a unit vector saturates the next-token
distribution (top-1 becomes whitespace or control tokens). The 10×
rescale was chosen by hand so that locked best coefficients land in
the interior of the sweep on both models; it is **not** from a
formal activation-norm matching scheme, and we list this as a
limitation.

### 3.5 Coefficient selection (tune only)

Per tune seed, per condition, we pick the coefficient that maximises
excess over the random-mean logit at that coefficient in the expected
direction: *decrease* for axis / critical roles / CAA / residuals,
*increase* for conformist roles, *unsigned* (|excess|) for random
controls. The best coefficient across tune seeds is the
**mode** (ties broken by count, then by proximity to the median);
this per-condition locked coefficient is written to
`results/best_coefs_tune_aggregate.json` and is the input to the test
run.

### 3.6 Test-split evaluation and statistics

With best coefficients locked from tune, we evaluate on the held-out
test split per test seed. For each condition we report (a) the mean
sycophancy logit across seeds, (b) the cross-seed standard deviation,
(c) the per-seed paired one-sided Wilcoxon signed-rank p-value on
base-level sycophancy-logit differences (steered − baseline), and
(d) Holm–Bonferroni correction across the full 24-condition family
per seed. 95 % CIs are paired bootstrap on the mean of per-base
differences, 2,000 iterations. We flag a (condition, coef) cell
**degraded** when both the binary rate is within 0.03 of 50 % and the
logit is within 0.10 of the random-mean logit at that coefficient —
an indication that steering has just destroyed the distribution.

### 3.7 CAA train/eval hygiene

CAA is extracted from `sycophancy_on_nlp_survey` and
`sycophancy_on_political_typology_quiz` only; evaluation is on
`sycophancy_on_philpapers2020`. By sha256 hash on whitespace-normalised
question text, train and eval share **zero** questions out of 20,184
train and 9,867 eval (`paper/tables/caa_leakage_receipt.{json,csv}`,
same source files on both pipelines).

---

## 4 Results

All numbers are on the held-out test split with tune-locked
coefficients, Holm-corrected across all 24 conditions per seed. Qwen:
3 test seeds; Gemma: 3 test seeds for the 21 primary conditions and
1 seed (42) for the 3 standalone residuals (these conditions were
added after the Gemma multi-seed run had completed — see §6).

### 4.1 Baselines

| | Gemma 2 27B | Qwen 3 32B |
|---|---|---|
| Baseline mean syc logit (3 test seeds) | +1.015 ± 0.011 | +3.000 ± 0.136 |
| Baseline syc rate | 59.3 % | 84.0 % |

Qwen is substantially more sycophantic at baseline on this benchmark,
so raw Δlogit magnitudes are not directly comparable; we report
Δlogit fractions of baseline alongside raw numbers where relevant.

### 4.2 Finding 1 — a general persona direction matches the targeted CAA vector

Mean post-steer logit (test, multi-seed aggregate) at each condition's
locked best coefficient:

| Condition | Gemma post-steer logit ± std | Δ from baseline | Qwen post-steer logit ± std | Δ from baseline |
|---|---|---|---|---|
| `caa` | +0.135 ± 0.014 | **−0.88** | +1.035 ± 0.042 | **−1.97** |
| `skeptic` | +0.304 ± 0.027 | −0.71 | +1.177 ± 0.134 | −1.82 |
| ratio Skeptic/CAA |   | **0.82** |   | **0.93** |

*(Source: `{model}/results/multiseed_aggregate_test.json`.)*

Skeptic is a persona direction extracted from generation-based
role-playing data (`lu-christina/assistant-axis-vectors`) without any
sycophancy labels. It still recovers 82 % (Gemma) and 93 % (Qwen) of
the targeted CAA vector's sycophancy reduction — both significant
(3/3 seeds after Holm) and with small cross-seed std. On both models
all five critical roles reach Holm significance in 3/3 test seeds.

### 4.3 Finding 2 — the reduction lives in the CAA-orthogonal residual

We steer with each role's **residual** (the CAA-orthogonal piece,
unit-normalised) at the parent role's own locked coefficient (the
matched-coefficient decomposition eval,
`{model}/results/decomposition_eval_test.json`). If the sycophancy
effect of a role was purely its sycophancy-aligned shadow, the
residual would have no behavioural effect. It does:

**Gemma** (baseline +1.01):

| Role residual | Coef | Post-steer logit | Δ vs baseline |
|---|---|---|---|
| assistant_axis residual | +2000 | +0.738 | −0.27 |
| skeptic residual | +2000 | +0.263 | **−0.75** |
| judge residual | +2000 | +0.389 | −0.62 |
| facilitator residual | −5000 | +0.407 | −0.60 |
| devils_advocate residual | +2000 | +0.466 | −0.54 |
| scientist residual | +2000 | +0.487 | −0.52 |
| contrarian residual | +2000 | +0.682 | −0.33 |
| peacekeeper residual | +2000 | +0.862 | −0.15 |
| collaborator residual | +500 | +0.965 | −0.04 |
| pacifist residual | +2000 | +1.001 | −0.01 |

**Qwen** (baseline +3.00, single seed 123 for this eval):

| Role residual | Coef | Post-steer logit | Δ vs baseline |
|---|---|---|---|
| assistant_axis residual | +200 | +0.563 | **−2.46** |
| contrarian residual | −200 | +0.703 | −2.32 |
| devils_advocate residual | +200 | +0.895 | −2.13 |
| skeptic residual | +200 | +1.275 | −1.75 |
| judge residual | +200 | +1.310 | −1.71 |
| scientist residual | −100 | +1.940 | −1.08 |
| peacekeeper residual | −200 | +2.179 | −0.85 |
| facilitator residual | −200 | +2.490 | −0.53 |
| collaborator residual | −100 | +2.948 | −0.08 |
| pacifist residual | +500 | +0.019 | −2.98 (**degraded**, rate pinned at 50 %) |

On both models, every critical role's residual carries most of the
parent role's reduction at the matched coefficient. For example on
Gemma, the full Skeptic vector reduces Δlogit by −0.71 (§4.2); its
residual alone reduces Δlogit by −0.75. On Qwen the same pattern
holds (Skeptic parent −1.82; residual −1.75). The
CAA-aligned-component row (not shown above — see §6) does not carry
the role-specific effect; after unit normalisation it collapses to
±v_CAA per role (§6 item 7).

### 4.4 Finding 3 — standalone residuals reduce sycophancy

We then evaluate the three standalone residuals as fresh conditions
with their *own* tune-locked coefficients (not reusing the parent
role's), with Holm across all 24 conditions. Data source:
`{model}/results/residual_standalone_report_test.csv` on Gemma
(seed 42, single seed) and
`qwen/results/multiseed_aggregate_test.json` on Qwen (3 seeds).

| Residual | \|cos(role, CAA)\| | Gemma Δlogit [95 % CI] | Gemma Holm p | Qwen Δlogit (3 seeds) | Qwen Holm (per seed) |
|---|---|---|---|---|---|
| Skeptic ⊥ CAA | G:0.064 Q:0.105 | −0.75 [−0.96, −0.57] | 3.6e-11 **✱** | −1.80 ± 0.14 | 3/3 ✱ |
| Contrarian ⊥ CAA | G:0.033 Q:0.071 | −0.32 [−0.55, −0.10] | 0.072 **ns** (CI excludes 0) | −1.65 ± 0.12 | 3/3 ✱ |
| Collaborator ⊥ CAA (Gemma high-cos) | G:0.165 | −0.93 [−1.13, −0.76] | <1e-15 **✱** | — | — |
| Devil's Advocate ⊥ CAA (Qwen high-cos) | Q:0.108 | — | — | −2.19 ± 0.08 | 3/3 ✱ |

**5 of 6 (model × residual) cells reach Holm-significance.** The one
cell that does not is **Gemma `contrarian_residual`**: raw Wilcoxon
p=0.020, Holm-adjusted p=0.072 across 24 conditions. The 95 % CI
excludes zero (−0.55 to −0.10) but the adjusted p does not pass
α=0.05. We report this as a null. All three Qwen residuals are
Holm-significant in every test seed with small cross-seed std.

### 4.5 Finding 4 — a secondary claim does NOT replicate

Before standalone evaluation we considered a sub-claim: that a
*high*-|cos(role, CAA)| role's residual would be the *worst* reducer
of the three, because steering at the parent role's coefficient (in
the matched-coef eval above) gives the smallest Δ for high-cos roles
(Gemma collaborator Δ=−0.04 at +500; facilitator Δ=−0.60 at −5000).

That sub-claim does not replicate when each residual gets its own
tune-locked coefficient. On **Gemma**, `collaborator_residual` at its
own locked coef (+5000) is the largest reducer of the three
(Δ=−0.93), not the smallest. On **Qwen**, `devils_advocate_residual`
(the high-cos role there) at +200 is also the largest reducer
(Δ=−2.19).

### 4.6 Bidirectionality (conformist roles should *increase* sycophancy)

| | Gemma | Qwen |
|---|---|---|
| Baseline rate | 59.3 % | 84.0 % |
| Conformist roles Holm-sig for increase (3 test seeds) | 1 / 4 (Collaborator, Δrate +3.3 pp, Holm p=0.002) | 0 / 4 |

On Qwen, every conformist role has 0/3 seeds significant for an
increase. We attribute this to the baseline-ceiling effect: Qwen
starts at 84 % sycophancy, so there are only ~16 pp of rate
headroom (and a correspondingly compressed logit range) before
hitting ceiling. Pacifist's tune-locked coef of +500 drives the
Qwen rate to exactly 50 %, a degradation signature (rate near
uniform, logit near the random-mean logit). Gemma is the cleaner
bidirectionality measurement; Qwen is ceiling-constrained and we do
not draw conclusions from its conformist numbers.

---

## 5 Analysis: why the sign of cos(role, CAA) flips between models, and why it matters

An interesting geometric observation: on Gemma, nine of ten roles
have **positive** cos(role, CAA) (range 0.003 to 0.165); on Qwen,
seven of ten have **negative** cos (range −0.108 to +0.032). See
`{model}/caa_decomposition.json`.

In isolation this doesn't change the main story: the residual is
orthogonal to CAA by construction in both cases, and carries the
role's reduction in both cases. But it does matter for a weaker
secondary claim that an earlier draft made — that the
"CAA-aligned component of a role opposes the role's reduction
direction." That phrasing is true on Gemma (small positive cosines
get unit-normalised to ≈ +v_CAA, steering in the positive direction
*increases* sycophancy) but false on Qwen (small negative cosines
unit-normalise to ≈ −v_CAA, steering in the positive direction
*decreases* sycophancy). The decomposition mechanics are identical;
only the sign of one scalar per role differs.

A related subtlety we flag explicitly: **unit-normalising the
CAA-aligned component of a role collapses it to ±v_CAA.** The
projection `(v_role · v_CAA) v_CAA` is just a scalar multiple of
`v_CAA`; unit-normalisation preserves only the sign. Consequently, in
`decomposition_eval_test.json` multiple roles with the same sign of
`cos(role, CAA)` produce *identical* mean_syc_logit values at matched
coefficient, because they are literally the same steering
intervention. The behavioural decomposition finding (§4.3) is about
the residual, not about per-role CAA-aligned-component differences.

---

## 6 Limitations

These are genuine caveats a reviewer would (and should) raise.

1. **Single-seed standalone residuals on Gemma.** The three
   `*_residual` conditions were added to the Gemma pipeline *after*
   its multi-seed run had completed, so the Gemma residual numbers
   are from seed 42 alone. On Qwen all three are in the 3-seed
   aggregate and the cross-seed std is small (≤ 0.14 on mean
   post-steer logit), which is indirect evidence that the Gemma
   single-seed residual numbers are unlikely to flip under re-sampling.
   Direct evidence would require a Gemma multi-seed re-run
   (approximately 10 GPU-hours; not done here).

2. **One non-significant residual on Gemma.** Gemma
   `contrarian_residual` has an adjusted p of 0.072, which does not
   cross α=0.05 under Holm. Its 95 % CI (−0.55, −0.10) excludes zero;
   the adjusted p does not. We frame it as a null.

3. **High-cos vs low-cos contrast is a genuine test on Gemma and a
   nominal one on Qwen.** Gemma's |cos(role, CAA)| values range from
   0.003 to 0.165, and the top role (collaborator 0.165) is a real
   outlier — next-highest is facilitator at 0.146, median across
   roles is ≈ 0.07. On Qwen the values range from 0.017 to 0.108,
   and the top role (devils_advocate 0.108) is barely separated from
   second (skeptic 0.105, a gap of 0.003 — below any sensible noise
   floor). The Qwen version of §4.5 should therefore be read as
   "the test cannot distinguish high-cos from low-cos roles on
   Qwen because the roles are not meaningfully separated in cosine
   space." A stronger test on Qwen would require either a
   naturally-occurring role with cos ≥ 0.15 (none in our set) or a
   synthetic role with injected CAA alignment.

4. **Coefficient scale is hand-tuned, not formally calibrated.** The
   10× Qwen/Gemma sweep rescale is informed by the ~116× gap in CAA
   raw norm but is otherwise a round-number choice that keeps locked
   best coefficients in the interior of each sweep. A principled
   calibration (e.g. match KL to the unsteered distribution, match
   Mahalanobis distance in activation space) is future work.

5. **Qwen bidirectionality is ceiling-constrained.** 84 % baseline
   sycophancy on Qwen leaves only ~16 pp of rate headroom for
   conformist roles to push higher. We do not draw conformist-role
   conclusions on Qwen. On Gemma only 1 of 4 conformist roles reaches
   Holm significance for an increase; bidirectionality is **weakly
   supported** overall, not strongly.

6. **Single benchmark.** All numbers are on `philpapers2020`.
   Generalisation to open-ended sycophancy (free-form flattery,
   agreement with user-asserted false claims) is only indirectly
   probed, via an over-correction taxonomy
   (`{model}/results/over_correction_eval_test.json`).

7. **Unit-normalised CAA-component is degenerate.** As noted in §5,
   the per-role CAA-aligned component collapses to ±v_CAA under unit
   normalisation, so the behavioural decomposition claim is about
   the residual. Per-role caa_component rows in the auxiliary tables
   should be read as ±v_CAA, grouped by the sign of
   `cos(role, CAA)`, not as distinct role-specific interventions.

8. **Model-specific patches.** On Qwen we pass
   `enable_thinking=False` to the chat template; without it the A/B
   logit is measured at a `<think>`-start token and the sycophancy
   signal collapses. This is a tokenisation quirk, not a protocol
   change, but it is a patch that needs to be re-checked for any
   future third model.

9. **Rank-one, single-layer interventions.** All steering is
   addition of a single unit vector at a single layer.
   Multi-vector or multi-layer (e.g. capping-style) steering is
   out of scope.

---

## 7 Related work

- **CAA.** Rimsky, Belrose, Teplyashin, Panickssery et al. 2024,
  *Steering Llama 2 via Contrastive Activation Addition*,
  arXiv:2312.06681. We follow their extraction recipe verbatim for
  the CAA direction (2,000 A/B pairs, mean(syc) − mean(honest) at the
  target layer, unit-normalise).
- **Persona / assistant-axis vectors.** The
  `lu-christina/assistant-axis-vectors` HuggingFace dataset (tied to
  `safety-research/assistant-axis`) releases role and axis vectors
  produced by a generate-then-LLM-judge pipeline for three models
  including our two. We use the `gemma-2-27b/` and `qwen-3-32b/`
  subdirs unchanged.
- **Sycophancy benchmark.** Perez et al. 2023,
  *Discovering Language Model Behaviors with Model-Written
  Evaluations*, arXiv:2212.09251. We use the A/B preference subset
  `sycophancy_on_philpapers2020` for evaluation and hold out
  `sycophancy_on_nlp_survey` plus
  `sycophancy_on_political_typology_quiz` for CAA training only.
- **Related activation-steering / ablation work.** Panickssery, Gale,
  Lin, et al. 2023 (contrastive persona steering), Zou et al. 2023
  (representation engineering), Turner et al. 2023 (activation
  steering for behaviour editing). We share the contrastive-mean
  extraction template with all three; our contribution is the
  orthogonal-decomposition analysis on sycophancy specifically,
  across two models.

---

## 8 Conclusion

Targeted CAA steering of sycophancy and general persona-direction
steering **coincide behaviourally on both Gemma 2 27B and Qwen 3
32B**: a persona direction extracted without any sycophancy labels
achieves 82 % (Gemma) and 93 % (Qwen) of CAA's sycophancy reduction.
That effect **lives in the CAA-orthogonal component of the role
vector** on both models: steering with only the residual at the
parent role's best coefficient recovers most of the parent's
reduction. When evaluated as standalone conditions with their own
tune-locked coefficients, 5 of 6 residual cells across the two
models clear Holm correction; the one exception is a null we report
honestly.

Our negative result on the sub-claim that a high-cos residual should
*fail* to reduce sycophancy does not replicate on either model,
though we note the high-cos role is not a geometric outlier on Qwen
(§6 item 3), so the Qwen half of that negative result should be read
as an inconclusive test rather than a strong refutation.

Next steps, in decreasing priority: (i) Gemma multi-seed re-run
against the expanded 24-condition list, so the three Gemma residuals
get cross-seed error bars; (ii) a calibrated coefficient scheme
matched on activation-space distance rather than raw-norm intuition;
(iii) open-ended sycophancy probes (beyond the taxonomy in
`over_correction_eval_test.json`) for a richer behavioural readout.

---

## Appendix A — Reproduction

Three repositories, all public on GitHub under `kelkarI/`:

- `sycophancy-gemma` — Gemma 2 27B pipeline + 3 residual conditions.
  (Previously named `sycophancy-final`; renamed for naming parity with
  `sycophancy-qwen`.)
- `sycophancy-qwen` — Qwen 3 32B pipeline, multi-seed end-to-end.
- `role-based-steering` — this paper's aggregated results + figures +
  `paper/{RESULTS, METHODS, LIMITATIONS}.md`.

Full-pipeline command sequences are listed in each repo's `README.md`;
the paper's central table is regenerated by
`paper/scripts/build_unified_table.py` against files in `gemma/` and
`qwen/`. Per-row `all_results_*.json` (~18 MB each) are gitignored
for size; everything summary-level is committed. Raw steering vectors
(`*.pt`) are gitignored per the
`lu-christina/assistant-axis-vectors` licence — regenerate via
`scripts/build_vectors_from_official.py` in the Qwen repo or
`scripts/01_prepare_steering_vectors.py` in the Gemma repo, both of
which pull from the HF dataset.

## Appendix B — Full per-condition test-split table

See `paper/tables/main_results_both_models.csv` (28 rows, one per
(model, condition)) and `.md` for a compact rendering, including
Δlogit, Δrate (pp), cross-seed std where multi-seed, Holm p (seed
42), tune-locked best coefficient, and `cos(role, CAA)`.

## Appendix C — CAA train/eval hygiene

From `paper/tables/caa_leakage_receipt.{json,csv,txt}`:

- Eval: `sycophancy_on_philpapers2020.jsonl`, 9,867 questions.
- Train (union of `nlp_survey` + `political_typology_quiz`): 20,184
  questions.
- Overlap by sha256 hash on whitespace-normalised question text: **0**.
