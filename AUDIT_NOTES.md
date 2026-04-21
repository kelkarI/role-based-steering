# Audit notes — branch `claude/audit-repo-consistency-vGCJJ`

This branch resolves a peer-review audit of the role-based-steering repo
against `kelkari/sycophancy-clean-results` (Repo 3) and the upstream
source pipelines (`kelkarI/sycophancy-final/experiment-main/` for Gemma
and `kelkarI/sycophancy-qwen` for Qwen). All changes here are
documentation / metadata / arithmetic — no model re-runs.

## Summary of fixes applied

| Tag | What was wrong | What this branch does |
|---|---|---|
| **C-2** | `gemma/results/over_correction_eval.json` was byte-identical (SHA `6b2615a5...`) to `gemma/results/over_correction_eval_test.json`; ambiguous which was canonical. | Deleted the unsuffixed `over_correction_eval.json`; `_test.json` is now canonical (matches Qwen's layout). |
| **A-1, M-4** | `paper/tables/main_results_both_models.{csv,md}` used a single-seed Gemma baseline (+1.009 ≈ seed 123) for every Gemma row, mislabelled as the cross-seed baseline. | Both files now use the cross-seed mean baseline (+1.0146; mean of [+1.0095, +1.0038, +1.0305]). |
| **A-2, M-3** | `cross_model/residual_standalone_gemma_vs_qwen.md` used two different Qwen baselines within the same file (3.024 in the top tables, 3.000 in the multi-seed table). | All tables in that file now use either (a) cross-seed mean +3.000 + per-seed paired Δ, or (b) explicitly labelled single-seed (seed-123) values for traceability. |
| **A-3, M-2** | `cross_model/residual_standalone_gemma_vs_qwen.md` claimed `devils_advocate_residual` Δlogit = −2.052 (and `contrarian_residual` = −2.153) in the interpretation paragraph. Neither value is reproducible from any data file. | Replaced with multi-seed paired Δ values: devils_advocate −2.193, contrarian −1.647, skeptic −1.799. The corrected ordering (high-cos > low-cos in reduction magnitude) actually *strengthens* the "does NOT hold" verdict for the residual claim. |
| **A-5, M-7** | Repo 2 used "mean-then-subtract" Δ aggregation (post_steer_mean − single baseline) while Repo 3 used "subtract-then-mean" (per-seed paired Δ then average). | Repo 2 CSV/MD/RESULTS.md now use per-seed paired Δ throughout, matching Repo 3. |
| **A-8, M-5** | `paper/RESULTS.md` said "Holm across all 24 conditions per seed" but `clean-results/README.md` and `build_data.py` said "Holm across the 14-condition primary family". | RESULTS.md and cross_model md now correctly say "Holm across the 14-condition family (11 main + 3 standalone residuals; 10 random controls excluded)". The `n_significant` in `multiseed_aggregate_test.json` was already this correction; the doc text was wrong. |
| **A-9, M-6** | The headline CSV mixed multi-seed (3-seed) main rows with single-seed (seed 42) Gemma residual rows without flagging the asymmetry. | CSV now has explicit `n_seeds` and `baseline_reference` columns; MD table marks single-seed rows with † and uses `(single seed)` instead of mean ± std. |
| **N-1** | RESULTS.md skeptic "Δ = −0.71 ± 0.03" used the post-steer-logit std (0.027), not the Δ-std (0.013). | All "± x" values are now the Δ-std (per-seed paired). |
| **N-2** | RESULTS.md rounded Gemma CAA Δ to −0.87; precise value is −0.879 → should round to −0.88. | All values quoted to 3 decimal places where data supports it. |
| **N-3** | "Skeptic / CAA = 0.82" came from rounded inputs (0.71/0.87); precise ratio is 0.808 → 0.81. | Updated to 0.81 (Gemma); 0.93 (Qwen) is correct as-is. |
| **C-3, S-1** | The "conformist roles increase sycophancy" framing in §7 was undermined by `facilitator`'s tune-locked Δ (−0.72 on Gemma, −0.47 on Qwen — both *reductions*, opposite to the prediction). | RESULTS.md §7 now reports the per-conformist-role direction breakdown explicitly and notes that the single-vector "conformist family pushes toward sycophancy" framing is not supported. Cross-model md adds the same caveat. |
| **M-1, H-5, H-11** | No metadata file pinning model revision, persona-vector revision, producing-pipeline commit. | Added `gemma/results/_metadata.json` and `qwen/results/_metadata.json` with everything that *can* be pinned now (model id, target layer, hook site, prompt template, eval set, coefficient sweep, seeds, baselines, Holm family, precision conventions). The HF revision SHAs and producing-commit SHA are marked `TODO_PIN_AT_RUNTIME` and `TODO_PIN_FROM_PRODUCING_RUN` because they are not recorded anywhere reachable from this repo and must be filled in by whoever next runs the pipeline. |
| **M-11** | Mixed precision (bf16 forward + fp32 vector math + bf16-or-fp32 addition inside `assistant-axis`) was undocumented. | Documented in both per-model `_metadata.json` files under `precision_hazards`. |

## Numbers that changed and by how much

The shifts are all small (<0.025 logit) but consistent.

| Quantity | Before | After | Δ |
|---|---|---|---|
| Gemma baseline_logit | +1.009 | +1.0146 | +0.0056 |
| Gemma skeptic Δlogit | −0.706 | −0.711 | −0.005 |
| Gemma CAA Δlogit | −0.874 | −0.879 | −0.005 |
| Gemma facilitator Δlogit | −0.722 | −0.727 | −0.005 |
| Qwen baseline_logit | +3.024 | +3.000 | −0.024 |
| Qwen skeptic Δlogit | −1.847 | −1.823 | +0.024 |
| Qwen CAA Δlogit | −1.989 | −1.965 | +0.024 |
| Qwen devils_advocate_residual Δlogit | −2.130 (single seed) / −2.052 (untraceable) | −2.193 (multi-seed paired) | various |
| Qwen contrarian_residual Δlogit | −1.684 (single seed) / −2.153 (untraceable) | −1.647 (multi-seed paired) | various |

After these changes, every Δ value in `paper/`, `cross_model/`, and the
`*_metadata.json` files is consistent with `clean-results/data/*_clean.json`
and with the underlying `multiseed_aggregate_test.json` files in this
repo (modulo rounding to 3 decimals).

## Things this branch does NOT fix (require runtime / source access)

These findings from the audit cannot be resolved from documentation
alone. They are recorded as `TODO_*` markers in the new `_metadata.json`
files.

| Finding | Why it can't be fixed here |
|---|---|
| **M-1 (HF model revision SHA)** | The producing runs did not pin a HF revision via `from_pretrained(..., revision=...)`. The next run should do so and write the resolved SHA into `_metadata.json["subject_model_revision"]`. |
| **M-9 (per-seed result files absent from this repo)** | The per-seed `seed_{42,7,123}/sycophancy_rates_test.json` files only exist in the upstream source repos (`kelkarI/sycophancy-final/experiment-main/results/` and `kelkarI/sycophancy-qwen/results/`). Repo 2 publishes only the multi-seed aggregate. To make Repo 2 self-sufficient, copy the per-seed dirs into `gemma/results/seed_*/` and `qwen/results/seed_*/`. |
| **M-8 (random null missing from Repo 2 aggregates)** | `multiseed_aggregate_test.json` has no `random_*` keys. Adding them requires per-seed source data (above) or recomputing a pooled random null from the upstream `sycophancy_rates_test.json` files. |
| **C-1 (vkmk1 fork divergence)** | `vkmk1/sandbagging-personas/experiment-main/` defines conformist roles `{servant, diplomat, disciple, peacemaker}` and a 12-non-zero coefficient grid; the published results use `{peacekeeper, pacifist, collaborator, facilitator}` and an 8-non-zero grid. The actual producing code lives in `kelkarI/sycophancy-final` (out of scope of this audit). The vkmk1 README has been clarified separately on the matching audit branch in that repo. |
| **H-4 (verify A_TOKEN_ID/B_TOKEN_ID)** | Should add an assertion in `02_evaluate_steering.py` (in the upstream code repo, not here): `assert tokenizer.encode(" A", add_special_tokens=False)[-1] == A_TOKEN_ID`. |
| **H-12 (Qwen `enable_thinking=False` not in vkmk1 proxy)** | The vkmk1 fork is missing this patch. The actual Qwen pipeline in `kelkarI/sycophancy-qwen/scripts/02_evaluate_steering.py` should have it (per cross_model md note); flagged for the upstream maintainer. |

## Files changed in this commit

- `paper/tables/main_results_both_models.csv` — rewritten with paired Δ + cross-seed baseline; new `n_seeds` and `baseline_reference` columns
- `paper/tables/main_results_both_models.md` — rewritten matching CSV
- `paper/RESULTS.md` — baseline reconciliation, Holm family corrected, `±` semantics fixed, facilitator counter-example acknowledged in §7
- `cross_model/residual_standalone_gemma_vs_qwen.md` — baselines reconciled, untraceable −2.052 / −2.153 numbers replaced with reproducible multi-seed paired Δ, §"Pattern check" table expanded
- `cross_model/residual_standalone_gemma_vs_qwen.csv` — Qwen residuals now use multi-seed paired Δ, Gemma residuals use seed-42 baseline (matching the actual single-seed run); Qwen `target_layer` corrected from 31 to 32 (off-by-one)
- `gemma/results/_metadata.json` — NEW: full provenance + Holm family + precision conventions
- `qwen/results/_metadata.json` — NEW: same for Qwen
- `gemma/results/over_correction_eval.json` — DELETED (was a byte-identical duplicate of `over_correction_eval_test.json`)
- `AUDIT_NOTES.md` — this file
