"""
Generate all paper-quality figures from saved experiment results.
Run after train_ablation.py and bench_context.py.
"""
import sys, os, json, csv, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict

matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04, "axes.linewidth": 0.8, "lines.linewidth": 1.5,
})

RESULTS_DIR = "results"
OUT         = os.path.join(RESULTS_DIR, "figures")
os.makedirs(OUT, exist_ok=True)

TC = "#2166AC"; MC = "#D6604D"; HC = "#1B7837"; GC = "#762A83"

# ─────────────────────────────────────────────────────────────────────────────
# Load aggregated results
# ─────────────────────────────────────────────────────────────────────────────

def load_ablation():
    path = os.path.join(RESULTS_DIR, "ablation_aggregate.csv")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["Val_PPL_mean"] = float(r["Val_PPL_mean"])
            r["Val_PPL_std"]  = float(r["Val_PPL_std"])
            r["Params"]       = int(r["Params"])
            rows.append(r)
    return rows


def load_context():
    path = os.path.join(RESULTS_DIR, "context_length_benchmark.json")
    with open(path) as f:
        return json.load(f)


def load_training_curves():
    """Load per-seed curves for ABR-2L to show variance bands."""
    import glob
    curves = defaultdict(list)
    for fpath in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        with open(fpath) as f:
            data = json.load(f)
        if "curve_steps" not in data:
            continue
        label = data["label"]
        curves[label].append({
            "steps":      data["curve_steps"],
            "train_loss": data["curve_train"],
            "val_loss":   data["curve_val"],
        })
    return curves


# ─────────────────────────────────────────────────────────────────────────────
# Fig A — Ablation bar chart (PPL mean ± std)
# ─────────────────────────────────────────────────────────────────────────────

def fig_ablation_bar(rows):
    labels = [r["Model"] for r in rows]
    means  = [r["Val_PPL_mean"] for r in rows]
    stds   = [r["Val_PPL_std"]  for r in rows]

    colors = [TC, MC, "#888888", "#AAAAAA", HC, HC]
    colors = colors[:len(rows)]

    fig, ax = plt.subplots(figsize=(7, 2.9))
    x = np.arange(len(rows))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                  edgecolor="#333", linewidth=0.7, width=0.62,
                  error_kw={"linewidth": 0.9}, zorder=3)
    bars[-1].set_edgecolor("#005500"); bars[-1].set_linewidth(1.4)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.3,
                f"{m:.2f}", ha="center", va="bottom", fontsize=6.5)

    short = [l.replace("ABR-", "").replace(" (Ours)", "\n(Ours)") for l in labels]
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7)
    ax.set_ylabel("Validation Perplexity (↓)"); ax.set_title("Ablation Study: Val PPL (mean ± std, 3 seeds)")
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.25)

    path = os.path.join(OUT, "ablation_ppl.png")
    fig.tight_layout(); fig.savefig(path)
    print(f"Saved: {path}"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig B — Context length table: PPL heatmap + bar
# ─────────────────────────────────────────────────────────────────────────────

def fig_context_length(ctx_rows):
    model_names = ["Transformer", "Mamba", "ABR-2L"]
    ctx_lens    = sorted(set(r["Context"] for r in ctx_rows))

    ppl = {m: [] for m in model_names}
    tps = {m: [] for m in model_names}
    mem = {m: [] for m in model_names}
    lat = {m: [] for m in model_names}

    for ctx in ctx_lens:
        for r in ctx_rows:
            if r["Context"] == ctx:
                ppl[r["Model"]].append(r["PPL_mean"])
                tps[r["Model"]].append(r["Tokens_per_s"])
                mem[r["Model"]].append(r["Peak_mem_GB"])
                lat[r["Model"]].append(r["Latency_ms"])

    fig, axes = plt.subplots(1, 4, figsize=(7, 2.5),
                             gridspec_kw={"wspace": 0.45})
    colors = {"Transformer": TC, "Mamba": MC, "ABR-2L": HC}
    markers = {"Transformer": "o", "Mamba": "s", "ABR-2L": "*"}

    for mname in model_names:
        axes[0].plot(ctx_lens, ppl[mname], marker=markers[mname],
                     color=colors[mname], label=mname, lw=1.3)
        axes[1].plot(ctx_lens, tps[mname], marker=markers[mname],
                     color=colors[mname], lw=1.3)
        axes[2].plot(ctx_lens, mem[mname], marker=markers[mname],
                     color=colors[mname], lw=1.3)
        axes[3].plot(ctx_lens, lat[mname], marker=markers[mname],
                     color=colors[mname], lw=1.3)

    titles = ["Perplexity (↓)", "Throughput tok/s (↑)",
              "Peak Memory GB (↓)", "Latency ms (↓)"]
    for i, (ax, t) in enumerate(zip(axes, titles)):
        ax.set_xlabel("Context Length"); ax.set_title(t, fontsize=8)
        ax.set_xticks(ctx_lens)
        ax.grid(linestyle="--", alpha=0.4); ax.spines[["top","right"]].set_visible(False)
        if i == 0:
            ax.legend(fontsize=6.5)

    path = os.path.join(OUT, "context_length_all.png")
    fig.tight_layout(); fig.savefig(path)
    print(f"Saved: {path}"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig C — Training curves with variance bands
# ─────────────────────────────────────────────────────────────────────────────

def fig_training_curves(curves_dict):
    keys_of_interest = ["Transformer-Only", "Mamba-Only",
                        "ABR-2L-state=ON (Ours)"]
    colors_map = {"Transformer-Only": TC, "Mamba-Only": MC,
                  "ABR-2L-state=ON (Ours)": HC}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.7),
                                    gridspec_kw={"wspace": 0.38})

    for key in keys_of_interest:
        runs = curves_dict.get(key, [])
        if not runs:
            continue
        col   = colors_map.get(key, "#555")
        lbl   = key.replace(" (Ours)", "")
        steps = runs[0]["steps"]

        train_mat = np.array([r["train_loss"] for r in runs])
        val_mat   = np.array([r["val_loss"]   for r in runs])

        for ax, mat in [(ax1, train_mat), (ax2, val_mat)]:
            m = mat.mean(0); s = mat.std(0)
            ax.plot(steps, m, color=col, label=lbl, lw=1.3)
            ax.fill_between(steps, m - s, m + s, color=col, alpha=0.15)

    for ax, t in [(ax1, "Training Loss"), (ax2, "Validation Loss")]:
        ax.set_xlabel("Step"); ax.set_ylabel("CE Loss"); ax.set_title(t)
        ax.legend(fontsize=6.5); ax.grid(linestyle="--", alpha=0.4)
        ax.spines[["top","right"]].set_visible(False)

    path = os.path.join(OUT, "training_curves.png")
    fig.tight_layout(); fig.savefig(path)
    print(f"Saved: {path}"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig D — Efficiency scatter: PPL vs Params coloured by Latency
# ─────────────────────────────────────────────────────────────────────────────

def fig_efficiency_scatter(abl_rows, ctx_rows):
    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    # Use context=256 latency
    lat_map = {}
    for r in ctx_rows:
        if r["Context"] == 256:
            lat_map[r["Model"]] = r["Latency_ms"]

    for row in abl_rows:
        name = row["Model"]
        ppl  = row["Val_PPL_mean"]
        params = row["Params"] / 1e6
        lat  = lat_map.get(name, 50)
        col  = HC if "Ours" in name else (TC if "Trans" in name else
               MC if "Mamba" in name else "#888")
        sc = ax.scatter(params, ppl, c=lat, cmap="YlOrRd", vmin=0, vmax=150,
                        s=70, edgecolors=col, linewidths=1.2, zorder=3)
        ax.annotate(name.split("-")[0].replace("Transformer", "Trans").replace("ABR","ABR"),
                    (params, ppl), textcoords="offset points",
                    xytext=(4, 3), fontsize=5.5)

    plt.colorbar(sc, ax=ax, label="Latency (ms)", shrink=0.8)
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Validation PPL (↓)")
    ax.set_title("Efficiency: PPL vs Params\n(colour = latency at ctx=256)")
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)

    path = os.path.join(OUT, "efficiency_scatter.png")
    fig.tight_layout(); fig.savefig(path)
    print(f"Saved: {path}"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading results...")
    abl_rows  = load_ablation()
    ctx_rows  = load_context()
    curves    = load_training_curves()

    print("Generating figures...")
    fig_ablation_bar(abl_rows)
    fig_context_length(ctx_rows)
    fig_training_curves(curves)
    fig_efficiency_scatter(abl_rows, ctx_rows)

    print(f"\nAll figures saved to: {OUT}/")


if __name__ == "__main__":
    main()
