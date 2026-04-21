# Main results — Gemma vs Qwen

Held-out test split. Δ logit and Δ rate are **per-seed paired** (each seed's post-steer value minus *that seed's* baseline, then averaged). Baseline = cross-seed mean (Gemma: 1.0146 over [1.0095, 1.0038, 1.0305]; Qwen: 3.000 over [3.0242, 2.8221, 3.1533]).

**Holm correction** in `multiseed_aggregate_test.json` applies across the **per-seed family used by the source pipeline (14 conditions: 11 main + 3 standalone residuals)**. The previous wording "24 conditions" in `RESULTS.md` was incorrect; see `AUDIT_NOTES.md`.

† marks single-seed rows (Gemma residuals, run on seed 42 only). Their baseline is the seed-42 baseline (1.0038 / 0.5933).

‡ marks degraded cells (model collapse: rate locked at 0.5; the apparent Δ is a saturation artefact).

| Model | Condition | Best coef | Baseline logit | Δ logit (mean ± std) | Δ rate (pp) ± std | Sig seeds (Holm) |
|---|---|---|---|---|---|---|
| gemma-2-27b-it | assistant_axis | +2000 | 1.0146 | -0.375 ± 0.018 | -5.00 ± 0.58 | 3/3 |
| gemma-2-27b-it | caa | -2000 | 1.0146 | -0.879 ± 0.001 | -8.89 ± 0.19 | 3/3 |
| gemma-2-27b-it | devils_advocate | +2000 | 1.0146 | -0.521 ± 0.016 | -8.67 ± 0.88 | 3/3 |
| gemma-2-27b-it | contrarian | +2000 | 1.0146 | -0.286 ± 0.019 | -3.33 ± 0.88 | 3/3 |
| gemma-2-27b-it | skeptic | +2000 | 1.0146 | -0.711 ± 0.013 | -9.56 ± 0.38 | 3/3 |
| gemma-2-27b-it | judge | +2000 | 1.0146 | -0.556 ± 0.003 | -9.33 ± 0.58 | 3/3 |
| gemma-2-27b-it | scientist | +2000 | 1.0146 | -0.509 ± 0.011 | -7.78 ± 1.07 | 3/3 |
| gemma-2-27b-it | peacekeeper | +2000 | 1.0146 | -0.052 ± 0.004 | +1.78 ± 1.64 | 0/3 |
| gemma-2-27b-it | pacifist | +2000 | 1.0146 | +0.100 ± 0.001 | +0.33 ± 1.76 | 1/3 |
| gemma-2-27b-it | collaborator | +500 | 1.0146 | +0.045 ± 0.006 | +1.67 ± 1.45 | 2/3 |
| gemma-2-27b-it | facilitator | -5000 | 1.0146 | -0.727 ± 0.008 | -9.22 ± 0.69 | 0/3 |
| gemma-2-27b-it | skeptic_residual† | +2000 | 1.0038 | -0.748 (single seed) | -9.33 (single seed) | 1/1 ✱ |
| gemma-2-27b-it | contrarian_residual† | +2000 | 1.0038 | -0.316 (single seed) | -4.00 (single seed) | 0/1 ns |
| gemma-2-27b-it | collaborator_residual† | +5000 | 1.0038 | -0.924 (single seed) | -8.33 (single seed) | 1/1 ✱ |
| qwen3-32b | assistant_axis | +200 | 2.9999 | -2.410 ± 0.092 | -18.56 ± 2.46 | 3/3 |
| qwen3-32b | caa | -200 | 2.9999 | -1.965 ± 0.126 | -20.89 ± 2.01 | 3/3 |
| qwen3-32b | devils_advocate | +200 | 2.9999 | -2.272 ± 0.195 | -16.56 ± 1.58 | 3/3 |
| qwen3-32b | contrarian | -200 | 2.9999 | -2.275 ± 0.106 | -15.22 ± 0.51 | 3/3 |
| qwen3-32b | skeptic | +200 | 2.9999 | -1.823 ± 0.058 | -18.11 ± 1.58 | 3/3 |
| qwen3-32b | judge | +200 | 2.9999 | -1.699 ± 0.075 | -4.44 ± 1.07 | 3/3 |
| qwen3-32b | scientist | -100 | 2.9999 | -0.984 ± 0.064 | -6.44 ± 1.50 | 3/3 |
| qwen3-32b | peacekeeper | -200 | 2.9999 | -0.709 ± 0.109 | +0.11 ± 0.19 | 0/3 |
| qwen3-32b | pacifist‡ | +500 | 2.9999 | -2.979 ± 0.174 | -34.00 ± 1.86 | 0/3 |
| qwen3-32b | collaborator | -100 | 2.9999 | -0.029 ± 0.016 | -1.67 ± 0.67 | 0/3 |
| qwen3-32b | facilitator | -200 | 2.9999 | -0.469 ± 0.099 | -0.56 ± 1.84 | 0/3 |
| qwen3-32b | skeptic_residual | +200 | 2.9999 | -1.799 ± 0.062 | -16.22 ± 1.39 | 3/3 |
| qwen3-32b | contrarian_residual | -100 | 2.9999 | -1.647 ± 0.054 | -10.11 ± 1.84 | 3/3 |
| qwen3-32b | devils_advocate_residual | +200 | 2.9999 | -2.193 ± 0.181 | -15.56 ± 2.04 | 3/3 |

## Provenance

- Per-condition multi-seed values: `{model}/results/multiseed_aggregate_test.json` (`mean_logit`, `rates`).
- Per-seed baselines: `seed_{42,7,123}/sycophancy_rates_test.json["<any-real-cond>"]["0.0"]["mean_syc_logit"]` in the upstream source repos. The cross-seed means are also surfaced in `kelkari/sycophancy-clean-results/data/{model}_clean.json["baselines_per_seed"]`.
- Δ logit computed as `mean_i(post_steer_logit_seed_i - baseline_logit_seed_i)` (per-seed paired). This differs from `mean_i(post_steer_logit_seed_i) - mean_j(baseline_logit_seed_j)` only via the (small) covariance between baseline and post-steer values across seeds; for paired runs the paired form is the appropriate one.
- Cosines and per-row Holm p-values inherited from the original CSV (single-seed, seed-42).
