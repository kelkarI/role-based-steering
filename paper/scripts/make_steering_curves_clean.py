"""Filtered steering-curve figures for both models (kept conditions only).

Re-draws the `fig1_steering_curves.png` sweeps under `gemma/figures/` and
`qwen/figures/` but restricts the conditions plotted to the "clean" set
used in the sycophancy-clean-results paper package:

  critical   : skeptic, devils_advocate, judge
  conformist : peacekeeper, pacifist, collaborator
  targeted   : caa
  null       : random mean (n=10) with +/- std band
  reference  : baseline (coef = 0)

Reads from `{gemma,qwen}/results/sycophancy_rates_test.json` (identical
to the rates files the source pipelines use for their crowded fig1) and
writes:

  gemma/figures/fig1_steering_curves_clean.png
  qwen/figures/fig1_steering_curves_clean.png
  paper/figures/gemma_fig1_steering_curves_clean.png
  paper/figures/qwen_fig1_steering_curves_clean.png

Deterministic: no RNG. Legend order fixed (CAA, critical, conformist,
random, baseline).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]  # role-based-steering/
PAPER_FIG = REPO / "paper" / "figures"

MODELS = [
    {
        "key": "gemma",
        "title": "Gemma 2 27B",
        "rates": REPO / "gemma" / "results" / "sycophancy_rates_test.json",
        "out_repo": REPO / "gemma" / "figures" / "fig1_steering_curves_clean.png",
        "out_paper": PAPER_FIG / "gemma_fig1_steering_curves_clean.png",
        "target_layer": 22,
    },
    {
        "key": "qwen",
        "title": "Qwen 3 32B",
        "rates": REPO / "qwen" / "results" / "sycophancy_rates_test.json",
        "out_repo": REPO / "qwen" / "figures" / "fig1_steering_curves_clean.png",
        "out_paper": PAPER_FIG / "qwen_fig1_steering_curves_clean.png",
        "target_layer": 32,
    },
]

CRITICAL_KEPT   = ["skeptic", "devils_advocate", "judge"]
CONFORMIST_KEPT = ["peacekeeper", "pacifist", "collaborator"]
KEPT_ORDER      = ["caa"] + CRITICAL_KEPT + CONFORMIST_KEPT
N_RANDOM        = 10
RANDOM_CONDS    = [f"random_{i}" for i in range(N_RANDOM)]

PALETTE = {
    "caa":             "#4d4d4d",
    "skeptic":         "#08519c",
    "devils_advocate": "#2171b5",
    "judge":           "#4292c6",
    "peacekeeper":     "#d94801",
    "pacifist":        "#f16913",
    "collaborator":    "#fd8d3c",
    "random":          "#bdbdbd",
}
LABELS = {
    "caa":             "CAA (targeted)",
    "skeptic":         "Skeptic",
    "devils_advocate": "Devil's Advocate",
    "judge":           "Judge",
    "peacekeeper":     "Peacekeeper",
    "pacifist":        "Pacifist",
    "collaborator":    "Collaborator",
}


def _baseline(rates):
    cond = next(c for c in rates if not c.startswith("random_"))
    cell = rates[cond]["0.0"]
    return float(cell["mean_syc_logit"]), float(cell["binary_rate"])


def _series(rates, cond, coeffs, metric):
    ys = []
    for c in coeffs:
        k = f"{c}" if c != 0.0 else "0.0"
        cell = rates[cond].get(k)
        if cell is None:
            ys.append(np.nan)
            continue
        ys.append(float(cell["binary_rate"]) * 100 if metric == "rate"
                  else float(cell["mean_syc_logit"]))
    return ys


def _random_band(rates, coeffs, metric):
    means, stds = [], []
    for c in coeffs:
        k = f"{c}" if c != 0.0 else "0.0"
        vs = []
        for rc in RANDOM_CONDS:
            cell = rates.get(rc, {}).get(k)
            if cell is None:
                continue
            vs.append(float(cell["binary_rate"]) * 100 if metric == "rate"
                      else float(cell["mean_syc_logit"]))
        means.append(float(np.mean(vs)) if vs else np.nan)
        stds.append(float(np.std(vs)) if vs else 0.0)
    return means, stds


def _panel(ax, rates, metric, ylabel, title, put_legend):
    cond_any = next(c for c in rates if not c.startswith("random_"))
    coeffs = sorted((float(k) for k in rates[cond_any]))
    for c in KEPT_ORDER:
        if c not in rates:
            continue
        ax.plot(coeffs, _series(rates, c, coeffs, metric),
                color=PALETTE[c], marker="o", markersize=4, lw=1.7,
                label=LABELS[c])
    rm, rs = _random_band(rates, coeffs, metric)
    ax.plot(coeffs, rm, color=PALETTE["random"], ls="--", marker="s",
            markersize=3, lw=1.4, label=f"Random mean (n={N_RANDOM})")
    ax.fill_between(coeffs,
                    [m - s for m, s in zip(rm, rs)],
                    [m + s for m, s in zip(rm, rs)],
                    color=PALETTE["random"], alpha=0.22, linewidth=0)
    bl_logit, bl_rate = _baseline(rates)
    ax.axhline(bl_rate * 100 if metric == "rate" else bl_logit,
               ls=":", color="black", alpha=0.8, lw=1.0,
               label="baseline = " + (f"{bl_rate*100:.1f}%" if metric == "rate"
                                      else f"{bl_logit:+.2f}"))
    ax.set_xlabel("Steering coefficient")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle=":")
    if put_legend:
        ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.9)


def make_model_figure(model):
    rates = json.loads(model["rates"].read_text())
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(10, 9))
    _panel(ax_a, rates, "rate",
           "Sycophancy rate (%)",
           f"{model['title']}: binary sycophancy rate vs steering coefficient "
           f"(kept conditions, L{model['target_layer']}, test split)",
           put_legend=True)
    _panel(ax_b, rates, "logit",
           r"Mean sycophancy logit  $\log p(\mathrm{syc}) - \log p(\mathrm{hon})$",
           f"{model['title']}: continuous sycophancy preference vs steering coefficient",
           put_legend=False)
    plt.tight_layout()
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    model["out_repo"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(model["out_repo"], dpi=150, bbox_inches="tight")
    fig.savefig(model["out_paper"], dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {model['out_repo']}")
    print(f"saved {model['out_paper']}")


def main():
    for m in MODELS:
        make_model_figure(m)


if __name__ == "__main__":
    main()
