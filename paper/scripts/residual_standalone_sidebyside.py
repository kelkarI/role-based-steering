"""Side-by-side Gemma vs Qwen report on the standalone residual conditions.

Merges results/residual_standalone_report_test.csv from each tree and writes:
  /lambda/nfs/filesystem/residual_standalone_gemma_vs_qwen.csv
  /lambda/nfs/filesystem/residual_standalone_gemma_vs_qwen.md
"""
import csv
import json
import os

GEMMA = "/lambda/nfs/filesystem/sycophancy-final/experiment-main"
QWEN  = "/lambda/nfs/filesystem/sycophancy-qwen"
OUT   = "/lambda/nfs/filesystem"


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_best_coefs(path):
    with open(path) as f:
        return json.load(f)


def load_decomp(path):
    with open(path) as f:
        return json.load(f)


def main():
    gemma_rows = read_csv(f"{GEMMA}/results/residual_standalone_report_test.csv")
    qwen_rows  = read_csv(f"{QWEN}/results/residual_standalone_report_test.csv")
    gemma_best = load_best_coefs(f"{GEMMA}/results/best_coefs_tune.json")["best_coefs"]
    qwen_best  = load_best_coefs(f"{QWEN}/results/best_coefs_tune.json")["best_coefs"]

    # Baselines (from residual_standalone summaries — baseline is on-page)
    # We also have them in sycophancy_rates_test.json under any non-random cond coef 0.0.
    def baseline(root):
        with open(f"{root}/results/sycophancy_rates_test.json") as f:
            r = json.load(f)
        s = next(iter(r.values()))["0.0"]
        return s["mean_syc_logit"], s["binary_rate"]

    g_base = baseline(GEMMA)
    q_base = baseline(QWEN)

    # Stitch combined CSV
    combined = []
    for r in gemma_rows:
        r2 = dict(r); r2["model"] = "gemma-2-27b-it"; r2["target_layer"] = 22
        combined.append(r2)
    for r in qwen_rows:
        r2 = dict(r); r2["model"] = "qwen3-32b"; r2["target_layer"] = 31
        combined.append(r2)
    fns = ["model", "target_layer"] + [k for k in combined[0].keys() if k not in {"model", "target_layer"}]
    out_csv = f"{OUT}/residual_standalone_gemma_vs_qwen.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader(); w.writerows(combined)
    print(f"wrote {out_csv}")

    # Markdown side-by-side
    def fmt_row(r):
        dlo = float(r["delta_logit"])
        dlo_lo = float(r["delta_logit_ci_lo"])
        dlo_hi = float(r["delta_logit_ci_hi"])
        dr = float(r["delta_rate_pp"])
        dr_lo = float(r["delta_rate_ci_lo_pp"])
        dr_hi = float(r["delta_rate_ci_hi_pp"])
        p = r.get("wilcoxon_p_holm") or ""
        if p:
            try:
                pn = float(p)
                p_str = "< 1e-15" if pn < 1e-15 else f"{pn:.3g}"
            except ValueError:
                p_str = str(p)
        else:
            p_str = "n/a"
        sig = "✱" if r.get("significant_after_holm") in ("True", "true", True) else "ns"
        parent_cos = float(r["parent_cosine_with_caa"])
        best_c = float(r["best_coef"])
        parent_c = float(r["parent_best_coef_tune"])
        differ = "differs" if best_c != parent_c else "same"
        return (
            f"| {r['condition']:22s} | {parent_cos:+.4f} | {best_c:+.0f} | {parent_c:+.0f} ({differ}) "
            f"| {dlo:+.3f} [{dlo_lo:+.3f}, {dlo_hi:+.3f}] | {dr:+.1f} [{dr_lo:+.1f}, {dr_hi:+.1f}] "
            f"| {p_str} | {sig} |"
        )

    md = []
    md += [
        "# Standalone CAA-orthogonal residual conditions — Gemma vs Qwen",
        "",
        "Both models were evaluated with the same 24-condition pipeline "
        "(assistant_axis, 5 critical roles, 4 conformist roles, CAA, 10 random "
        "controls, 3 standalone residuals), Holm correction applied across all "
        "24 conditions on the held-out test split, with coefficients locked from "
        "each model's own tune split. The standalone residual for each model is "
        "the role vector minus its projection onto CAA, unit-normalised, then "
        "run through the full coefficient sweep as an independent steering "
        "condition with a fresh tune-locked coefficient.",
        "",
        "## Model differences",
        "",
        "| Property | Gemma-2-27b-it | Qwen 3 32B |",
        "|---|---|---|",
        "| target layer | 22 / 46 | 32 / 64 |",
        "| coefficient sweep | ±500, ±1000, ±2000, ±5000 | ±50, ±100, ±200, ±500 |",
        f"| baseline mean syc logit (test) | {g_base[0]:+.3f} | {q_base[0]:+.3f} |",
        f"| baseline sycophancy rate (test) | {g_base[1]*100:.1f}% | {q_base[1]*100:.1f}% |",
        "| role vector source | lu-christina/assistant-axis-vectors (`gemma-2-27b/`) | lu-christina/assistant-axis-vectors (`qwen-3-32b/`) |",
        "| max |cos(role, CAA)| among critical+conformist roles | 0.165 (collaborator) | 0.108 (devils_advocate) |",
        "",
        "The coefficient-sweep rescale on Qwen was required because Qwen's "
        "layer-32 default-activation norm (~166) is much smaller than Gemma's "
        "layer-22 norm, so |coef|=5000 completely saturates Qwen's next-token "
        "distribution (top-1 becomes whitespace / control tokens). The rescaled "
        "Qwen grid was verified on a pilot probe to give a clean dose-response "
        "from baseline-like at |coef|=50 to anti-sycophantic at |coef|=200 to "
        "saturated at |coef|=500. Gemma's original grid was retained unchanged.",
        "",
        "## Standalone residual results (held-out test split, Holm across 24 conditions)",
        "",
        "### Gemma-2-27b-it",
        "",
        f"Baseline logit: {g_base[0]:+.3f}   Baseline rate: {g_base[1]*100:.1f}%",
        "",
        "| Condition | \\|cos(role, CAA)\\| | Best coef (locked) | Parent's best coef | Δ logit [95% CI] | Δ rate pp [95% CI] | Holm p | sig |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in gemma_rows:
        md.append(fmt_row(r))
    md += [
        "",
        "### Qwen 3 32B",
        "",
        f"Baseline logit: {q_base[0]:+.3f}   Baseline rate: {q_base[1]*100:.1f}%",
        "",
        "| Condition | \\|cos(role, CAA)\\| | Best coef (locked) | Parent's best coef | Δ logit [95% CI] | Δ rate pp [95% CI] | Holm p | sig |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in qwen_rows:
        md.append(fmt_row(r))

    # Pattern verdict
    def reduces(r):
        return (
            float(r["delta_logit"]) < 0
            and float(r["delta_logit_ci_hi"]) < 0
            and r.get("significant_after_holm") in ("True", "true", True)
        )

    def verdict(rows):
        by_cos = sorted(rows, key=lambda r: abs(float(r["parent_cosine_with_caa"])))
        low = by_cos[:-1]
        high = by_cos[-1:]
        low_all = all(reduces(r) for r in low)
        high_any = any(reduces(r) for r in high)
        hold = low_all and not high_any
        return low, high, low_all, high_any, hold

    g_low, g_high, g_la, g_ha, g_hold = verdict(gemma_rows)
    q_low, q_high, q_la, q_ha, q_hold = verdict(qwen_rows)

    # Multi-seed panel
    import numpy as np
    qwen_mseed_path = f"{QWEN}/results/multiseed_aggregate_test.json"
    if os.path.exists(qwen_mseed_path):
        with open(qwen_mseed_path) as f:
            mseed = json.load(f)
        # Also pull per-seed baselines for Qwen and normalise
        baselines = []
        for s in mseed.get("seeds", [42, 7, 123]):
            p = f"{QWEN}/results/seed_{s}/sycophancy_rates_test.json"
            if os.path.exists(p):
                with open(p) as f:
                    r = json.load(f)
                baselines.append(r["assistant_axis"]["0.0"]["mean_syc_logit"])
        b_mean = float(np.mean(baselines)) if baselines else None
        b_std  = float(np.std(baselines))  if baselines else None

        md += [
            "",
            "## Qwen multi-seed aggregate (mean ± std across 3 test seeds)",
            "",
            "Re-ran the full 24-condition pipeline on Qwen across 5 tune seeds "
            "(42, 7, 123, 456, 789) and 3 test seeds (42, 7, 123), matching the "
            "Gemma pipeline's multi-seed protocol. Tune-locked coefficients "
            "came from the mode-across-seeds aggregate "
            "(results/best_coefs_tune_aggregate.json). Wilcoxon+Holm applied "
            "per seed.",
            "",
            f"Baseline across 3 test seeds: logit = "
            f"{b_mean:+.3f} ± {b_std:.3f}",
            "",
            "| Condition | mean post-steer logit ± std | sig seeds / n | mean Δ logit |",
            "|---|---|---|---|",
        ]
        ordered = [
            "assistant_axis", "caa",
            "devils_advocate", "contrarian", "skeptic", "judge", "scientist",
            "peacekeeper", "pacifist", "collaborator", "facilitator",
            "skeptic_residual", "contrarian_residual", "devils_advocate_residual",
        ]
        for c in ordered:
            d = mseed["per_condition"].get(c)
            if not d:
                continue
            dl = (d["mean_logit"] - b_mean) if b_mean is not None else None
            md.append(
                f"| {c:26s} | {d['mean_logit']:+.3f} ± {d['std_logit']:.3f} "
                f"| {d['n_significant']}/{d['n_seeds']} "
                f"| {'—' if dl is None else f'{dl:+.3f}'} |"
            )
        md += [
            "",
            "On Qwen, 3/3 seeds show Holm-significant reductions for CAA, "
            "assistant_axis, all 5 critical roles, and all 3 standalone "
            "residuals. Conformist roles: 0/3. The cross-seed std on the "
            "post-steer logit is small (≤ 0.25 for all significant "
            "conditions), indicating the single-seed numbers reported above "
            "are stable.",
            "",
            "### Gemma residual parity note",
            "",
            "Gemma's multi-seed aggregate (`experiment-main/results/"
            "multiseed_aggregate.json`) contains only the 21 original "
            "conditions — the 3 standalone residuals were added to the "
            "pipeline single-seed. The Gemma residual numbers in the table "
            "above are therefore from one seed-42 tune/test subsample. "
            "The Qwen residual numbers shown at the head of this report "
            "(from the same seed-123 subsample that results/ ends in after "
            "the multi-seed run) are consistent with the 3-seed aggregate "
            "in this panel (all three conditions Δlogit CI far from zero "
            "in every seed), so a re-run of Gemma multi-seed is unlikely "
            "to flip the core verdict — but it would produce comparable "
            "error bars.",
        ]

    md += [
        "",
        "## Pattern check vs. matched-coefficient decomposition finding",
        "",
        "Claim under test: low-|cos(role, CAA)| residuals reduce sycophancy when "
        "steered as standalone directions; the high-|cos(role, CAA)| residual does "
        "not. This is the standalone-evaluation analogue of the matched-coefficient "
        "decomposition finding.",
        "",
        "| Model | Low-cos residuals | All low-cos reduce syc? | High-cos residual | High-cos also reduces syc? | Verdict |",
        "|---|---|---|---|---|---|",
        f"| Gemma | {', '.join(r['parent_role'] for r in g_low)} | {g_la} | {g_high[0]['parent_role']} | {g_ha} | {'holds' if g_hold else '**does NOT hold**'} |",
        f"| Qwen  | {', '.join(r['parent_role'] for r in q_low)} | {q_la} | {q_high[0]['parent_role']} | {q_ha} | {'holds' if q_hold else '**does NOT hold**'} |",
        "",
        "### Interpretation",
        "",
        "On **Gemma**, the high-cosine residual (`collaborator`, |cos|=0.165) "
        "reduces sycophancy at its own tune-locked coefficient (+5000) by Δlogit "
        "= −0.929, strongly significant after Holm. The matched-coefficient "
        "decomposition finding does not replicate when residuals are steered "
        "standalone. The low-cosine `contrarian` residual also fails to reach "
        "Holm significance (p=0.072) — the other half of the claim also breaks.",
        "",
        "On **Qwen**, the highest-|cos(role, CAA)| role using the official "
        "`lu-christina/assistant-axis-vectors/qwen-3-32b` release at layer 32 "
        "is `devils_advocate` (|cos|=0.108); skeptic is second at 0.105. All "
        "three residuals significantly reduce sycophancy after Holm, each with "
        "a 95% paired-bootstrap CI that excludes zero. Devils_advocate's "
        "residual Δlogit = −2.052 is actually the MIDDLE of the three on "
        "Qwen — contrarian (Δ=−2.153, |cos|=0.071) edges it out despite "
        "having a lower cosine with CAA. So the expected low-cos > high-cos "
        "ordering does not appear.",
        "",
        "Taken together, the *standalone* evaluation on both models does not "
        "support the matched-coefficient claim that the CAA-orthogonal residual "
        "of a high-|cos(CAA)| role fails to reduce sycophancy. Given each "
        "residual its own tune-locked coefficient, the high-cos residual reduces "
        "sycophancy at least as well as the low-cos residuals on both models.",
        "",
        "### Methodological notes",
        "",
        "1. **Vector source.** Both models use persona/assistant-axis vectors from "
        "`lu-christina/assistant-axis-vectors` (the generation-plus-LLM-judge "
        "pipeline from `safety-research/assistant-axis`). Gemma uses the "
        "`gemma-2-27b/` subdir at layer 22; Qwen uses the `qwen-3-32b/` subdir "
        "at layer 32 (the canonical target layer from "
        "`assistant_axis.models.MODEL_CONFIGS`).",
        "",
        "2. **CAA extraction.** The authors don't ship a sycophancy CAA vector "
        "-- that's computed on top by both experiments using Rimsky et al. 2024: "
        "1000 pairs from NLP-survey and 1000 from political-typology-quiz, "
        "contrasting last-token activations on (question + syc-answer) vs "
        "(question + honest-answer) at the target layer of each model.",
        "",
        "3. **Coefficient rescaling.** Qwen's coefficient grid is 10× smaller "
        "than Gemma's (±50..±500 vs ±500..±5000). At layer 32 on Qwen, "
        "|coef|=5000 on a unit vector saturates the next-token distribution. "
        "Both experiments therefore span roughly the same dose-response regime "
        "relative to their own activation scale, but the saturation cliff "
        "lives at different absolute coefficient values in the two models.",
        "",
        "4. **Qwen chat template.** Qwen 3's default chat template injects an "
        "empty `<think></think>` block when `enable_thinking=False`. The Qwen "
        "pipeline patches `build_prompt` in `02_evaluate_steering.py` to pass "
        "`enable_thinking=False` so the A/B logprob measurement is at the "
        "post-`</think>` position (where the model would normally emit its "
        "answer). Without this patch, the measured logits are at the "
        "`<think>`-start position and the sycophancy signal collapses.",
        "",
        "5. **Baseline sycophancy rate differs between models.** Gemma 2 27B "
        f"baseline is {g_base[0]:+.2f} logit / {g_base[1]*100:.0f}% syc rate; "
        f"Qwen 3 32B baseline is {q_base[0]:+.2f} logit / {q_base[1]*100:.0f}%. "
        "Qwen starts substantially more sycophantic on this benchmark, which "
        "amplifies the absolute magnitude of Δlogit that any given intervention "
        "can produce (Qwen has more room to reduce). Cross-model magnitude "
        "comparisons should be read relative to each model's baseline.",
    ]

    out_md = f"{OUT}/residual_standalone_gemma_vs_qwen.md"
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
