"""Build paper/tables/main_results_both_models.{csv,tex} from per-model
aggregates.

Pulls for each model:
  - baseline (mean±std across test seeds, from seed_*/sycophancy_rates_test.json
    on disk -- but the repo only ships the final aggregate, so we read from
    {model}/results/multiseed_aggregate_test.json + the single summary_test.txt
    for the rows that were single-seed).
  - per-condition mean Δ logit, cross-seed std, significant seed count.
  - tune-locked best coef (from best_coefs_tune_aggregate.json where the
    multi-seed run produced one, else best_coefs_tune.json).

One row per (model, condition). Columns chosen for a paper main table.
"""
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = [
    ("gemma-2-27b-it", ROOT / "gemma", 22),
    ("qwen3-32b",      ROOT / "qwen",  32),
]

ORDER = [
    # (condition, pretty_label, family)
    ("assistant_axis",             "Assistant Axis",        "primary"),
    ("caa",                        "CAA (targeted)",        "primary"),
    ("devils_advocate",            "Devil's Advocate",      "critical"),
    ("contrarian",                 "Contrarian",            "critical"),
    ("skeptic",                    "Skeptic",               "critical"),
    ("judge",                      "Judge",                 "critical"),
    ("scientist",                  "Scientist",             "critical"),
    ("peacekeeper",                "Peacekeeper",           "conformist"),
    ("pacifist",                   "Pacifist",              "conformist"),
    ("collaborator",               "Collaborator",          "conformist"),
    ("facilitator",                "Facilitator",           "conformist"),
    # Residuals -- per model the "high-cos" one differs; include all that
    # exist in each model's results
    ("skeptic_residual",           "Skeptic \u22A5 CAA",           "residual_standalone"),
    ("contrarian_residual",        "Contrarian \u22A5 CAA",        "residual_standalone"),
    ("collaborator_residual",      "Collaborator \u22A5 CAA",      "residual_standalone"),
    ("facilitator_residual",       "Facilitator \u22A5 CAA",       "residual_standalone"),
    ("devils_advocate_residual",   "Devil's Advocate \u22A5 CAA",  "residual_standalone"),
]


def model_row(model_name, model_dir, layer, cond):
    """Return a row dict for a (model, condition). None if condition not present."""
    agg_path = model_dir / "results/multiseed_aggregate_test.json"
    tests_path = model_dir / "results/statistical_tests_test.json"
    bc_path = model_dir / "results/best_coefs_tune_aggregate.json"
    sing_path = model_dir / "results/best_coefs_tune.json"
    rates_path = model_dir / "results/sycophancy_rates_test.json"
    decomp_path = model_dir / "caa_decomposition.json"

    # Prefer aggregate (mode across tune seeds), fall back per-condition to
    # single-seed best_coefs_tune.json for conditions the aggregate predates
    # (e.g. Gemma's 3 standalone residuals, which were added post-aggregate).
    best_coefs = {}
    if sing_path.exists():
        best_coefs.update(json.loads(sing_path.read_text())["best_coefs"])
    if bc_path.exists():
        best_coefs.update(json.loads(bc_path.read_text())["best_coefs"])

    rates = json.loads(rates_path.read_text()) if rates_path.exists() else {}
    decomp = json.loads(decomp_path.read_text()) if decomp_path.exists() else {}
    tests = json.loads(tests_path.read_text()) if tests_path.exists() else {}

    # Baseline from single-seed rates (any real cond coef 0.0)
    first_cond = next(iter(rates))
    base_logit = rates[first_cond]["0.0"]["mean_syc_logit"]
    base_rate = rates[first_cond]["0.0"]["binary_rate"]

    if cond not in rates:
        return None  # condition not present in this model

    # Multi-seed stats where available
    agg = None
    if agg_path.exists():
        a = json.loads(agg_path.read_text()).get("per_condition", {}).get(cond)
        if a:
            agg = {
                "mean_logit": a["mean_logit"],
                "std_logit": a["std_logit"],
                "n_seeds": a["n_seeds"],
                "n_significant": a["n_significant"],
            }

    # Single-seed fallback from rates at best coef
    coef = float(best_coefs.get(cond, 0.0))
    k = f"{coef}" if coef != 0.0 else "0.0"
    cell = rates.get(cond, {}).get(k, {})
    ss_logit = cell.get("mean_syc_logit")
    ss_rate = cell.get("binary_rate")

    wilc = tests.get("primary_wilcoxon_best_vs_baseline", {}).get(cond, {})
    p_raw = wilc.get("p_value_raw")
    p_adj = wilc.get("p_value_adjusted")
    sig_after_mcc = wilc.get("significant_after_mcc")

    cos_caa = (decomp.get(cond, {}).get("cosine_with_caa")
               if cond in decomp else None)

    mean_logit = agg["mean_logit"] if agg else ss_logit
    std_logit  = agg["std_logit"]  if agg else None
    delta_logit = (mean_logit - base_logit) if mean_logit is not None else None

    return {
        "model": model_name,
        "layer": layer,
        "condition": cond,
        "best_coef_locked": coef,
        "baseline_logit": round(base_logit, 3),
        "baseline_rate": round(base_rate, 4),
        "post_steer_logit_mean": (round(mean_logit, 3) if mean_logit is not None else None),
        "post_steer_logit_std_across_seeds": (round(std_logit, 3) if std_logit is not None else None),
        "delta_logit": (round(delta_logit, 3) if delta_logit is not None else None),
        "post_steer_rate": (round(ss_rate, 4) if ss_rate is not None else None),
        "delta_rate_pp": (round((ss_rate - base_rate) * 100, 2) if ss_rate is not None else None),
        "seeds_significant_over_total": (f"{agg['n_significant']}/{agg['n_seeds']}" if agg else None),
        "wilcoxon_p_raw_single_seed": p_raw,
        "wilcoxon_p_holm_single_seed": p_adj,
        "significant_after_holm_single_seed": sig_after_mcc,
        "cosine_role_vs_caa": (round(cos_caa, 4) if cos_caa is not None else None),
    }


def main():
    rows = []
    for model_name, model_dir, layer in MODELS:
        for cond, _, _ in ORDER:
            r = model_row(model_name, model_dir, layer, cond)
            if r:
                rows.append(r)

    csv_path = ROOT / "paper/tables/main_results_both_models.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}  ({len(rows)} rows)")

    # Paper-friendly compact Markdown
    md = ["# Main results — Gemma vs Qwen\n",
          "Held-out test split, Holm–Bonferroni across all 24 conditions.",
          "Qwen rows are mean ± cross-seed std (3 test seeds).",
          "Gemma rows are single-seed for the 3 residual conditions (added after",
          "the Gemma multi-seed run) and multi-seed for the 21 other conditions.\n",
          "| Model | Cond | best coef | baseline logit | Δ logit | Δ rate pp | sig seeds | Holm p (seed 42) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['model']} | {r['condition']} | {r['best_coef_locked']:+.0f} "
            f"| {r['baseline_logit']:+.3f} "
            f"| {r['delta_logit']:+.3f}"
            f"{(' ± ' + str(r['post_steer_logit_std_across_seeds'])) if r.get('post_steer_logit_std_across_seeds') else ''} "
            f"| {r.get('delta_rate_pp','')} "
            f"| {r.get('seeds_significant_over_total') or '1 seed'} "
            f"| {r.get('wilcoxon_p_holm_single_seed')} |"
        )
    md_path = ROOT / "paper/tables/main_results_both_models.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
