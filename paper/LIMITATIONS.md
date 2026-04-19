# Limitations

## Asymmetries between the two models

1. **Gemma standalone residuals are single-seed.** The three residual
   conditions (skeptic, contrarian, collaborator ⊥ CAA) were added to the
   Gemma pipeline after its multi-seed run had already completed.
   Consequently the Δ logit and Holm p-values reported for Gemma residuals
   are from one base-question subsample (seed 42) with n = 150 test bases.
   On Qwen these three conditions have 3 test seeds (42, 7, 123). The
   Qwen cross-seed σ on the residual mean_logit is ≤ 0.14, which is
   small relative to the effect sizes (Δ = −1.6 to −2.2), so a Gemma
   multi-seed re-run is unlikely to flip the core verdict that the
   residuals reduce sycophancy. It would, however, give Gemma residuals
   comparable error bars to the other Gemma conditions.

2. **Coefficient scales differ.** Gemma uses ±500..±5000 and Qwen uses
   ±50..±500. Necessary because Qwen's layer-32 activation norm is
   smaller; |coef|=5000 on a unit vector saturates Qwen's next-token
   distribution. Both sweeps span roughly the same dose-response regime
   relative to each model's activation scale, but absolute coefficient
   magnitudes are not comparable across models.

3. **Target layers differ.** Gemma uses 22/46; Qwen uses 32/64. Both are
   the canonical mid-stack layer per `assistant_axis.models.MODEL_CONFIGS`
   for that model. A layer sensitivity sweep was not run.

4. **Chat-template handling.** The Qwen pipeline patches
   `02_evaluate_steering.py` to pass `enable_thinking=False` so the A/B
   logprob measurement is at the post-`</think>` token. Gemma's template
   ignores this flag. Without the patch the Qwen logits are measured at
   the `<think>`-start token and the sycophancy signal collapses
   (baseline logit ≈ 0 at every coefficient). This is a tokenization
   quirk, not a protocol change, but it is a model-specific patch.

## Cross-condition considerations

5. **The "high-\|cos\| role" is different on each model.** On Gemma the
   role with the largest \|cos(role, CAA)\| is collaborator (0.165); on
   Qwen it is devils_advocate (0.108). The test uses a per-model rule
   ("pick the highest-cos role") rather than a fixed role. That means the
   *specific vector* being evaluated as the "high-cos residual" differs
   between models — only the rule is the same. Reported findings about
   the high-cos residual are therefore claims about the extreme of each
   model's own decomposition, not about a single chosen role.

6. **Near-orthogonality.** On both models every role is within ~10° of
   orthogonal to CAA (angles 80.5°-96.2°). The absolute values of
   cos(role, CAA) are small (≤ 0.17 on Gemma, ≤ 0.11 on Qwen). Analyses
   that contrast "high" vs "low" cosine roles are therefore operating in
   a narrow range; the "low vs high" distinction should be read as a
   *within-decomposition* ordering, not as a large geometric separation.

7. **The sign of cos(role, CAA) flips between models.** On Gemma most
   roles have nominally positive cosines; on Qwen most have nominally
   negative cosines. This sign determines the direction of the
   unit-normalised CAA-aligned component after projection+normalisation
   and therefore whether the aligned component helps or hurts sycophancy
   reduction at a positive steering coefficient. The Gemma study's
   phrasing that the CAA-aligned component "opposes the role's reduction
   direction" is therefore *model-specific*, not a structural property of
   the decomposition. The phrasing should be softened in any
   cross-model write-up.

## Conformist / bidirectionality

8. **Qwen bidirectionality is dominated by a ceiling effect.** Qwen
   baseline sycophancy is 84%; there is roughly 16 percentage points of
   headroom for a conformist role to push the model *more* sycophantic
   before hitting 100%. In practice the Qwen pipeline never produces a
   significant increase in sycophancy from any conformist role across 3
   test seeds; the direction-aware selector for conformist roles on Qwen
   often lands at a saturation-edge coefficient (e.g. Pacifist at +500
   drives the rate to exactly 50%, a degradation signature). The Qwen
   data therefore does not provide a clean test of the bidirectionality
   claim — only the Gemma data does, and there only one of four
   conformist roles (Collaborator) reaches significance.

## Scope and generalisation

9. **One benchmark.** All numbers are on `philpapers2020`. Two other
   sycophancy benchmarks exist in the same series (`nlp_survey`,
   `political_typology_quiz`) but those are used for CAA training in this
   study, so evaluating on them would leak. Generalisation to non-A/B
   sycophancy (open-ended flattery, agreement with incorrect user claims)
   is partly probed by `over_correction_check.py` (16 probes classified
   into a 6-way taxonomy) but is not the primary reported result.

10. **Two models.** Gemma 2 27B and Qwen 3 32B are both
    instruction-tuned, decoder-only, at similar parameter counts.
    Replication on Llama 3.3 70B (for which
    `lu-christina/assistant-axis-vectors` also ships vectors) was not
    done here.

## Methodology

11. **CAA training-set overlap.** The CAA direction is extracted from
    NLP-survey + political-typology-quiz, and the eval is philpapers2020.
    These are disjoint at the question-text level (verified by hash
    receipt on Gemma side; same `anthropics/evals` source files used on
    Qwen). CAA's ability to generalise to philpapers is therefore a
    transfer result, not a within-domain fit. That said, all three are
    Anthropic-authored A/B preferences in a similar format.

12. **The 02d response-token projection analysis was not run on Qwen.**
    On Gemma, mean axis-dot over 8 greedy response tokens predicts
    sycophancy better than the prompt-last-token projection does. This
    analysis was skipped on Qwen for runtime reasons; the prompt-last-token
    projection analysis is reported for both models.

13. **Seeded random controls.** Random vectors are unit-Gaussian seeded
    by `RANDOM_SEED_BASE + i` — the same seeds on both models, which
    produces different vectors because the hidden dimension differs
    (4608 vs 5120). So the *specific* random directions are
    model-specific. The distribution of random controls is matched
    (both Gaussian-unit), so the random-mean comparison the selector uses
    is internally consistent per model.

## Not tested / deliberately out of scope

- Layer sensitivity of the residual-steering finding.
- Generalisation of a Gemma-extracted residual to Qwen and vice versa
  (impossible anyway — 4608-d vs 5120-d vectors are not compatible).
- Response-token projection on Qwen.
- Any qualitative / free-form output analysis beyond the
  over-correction-taxonomy classifier.
- Multi-layer / rank-R steering interventions; all steering is single
  layer, rank 1.
