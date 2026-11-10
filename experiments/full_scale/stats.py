"""
Statistics for multi-seed results (PRD P1.5 / P2.3, fixes B8).

    summarize(values)     mean, std (ddof=1), n, 95% t-interval
    welch(a, b)           Welch t-test + Cohen's d / Hedges' g
    compare_table(...)    Welch tests for the PRD key pairs, with Holm-adjusted p
    load_results(dir)     read <dir>/*/result.json written by provenance.finish_run
    aggregate(dir, ...)   per-config summaries + key-pair table, optional CSVs
"""
import csv
import glob
import json
import math
import os

import numpy as np
from scipy import stats as _st


def summarize(values, confidence=0.95):
    x = np.asarray(list(values), dtype=float)
    n = int(x.size)
    mean = float(x.mean()) if n else float("nan")
    if n < 2:
        return {"mean": mean, "std": float("nan"), "n": n, "sem": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "ci_half": float("nan")}
    std = float(x.std(ddof=1))
    sem = std / math.sqrt(n)
    half = float(_st.t.ppf(0.5 + confidence / 2, df=n - 1) * sem)
    return {"mean": mean, "std": std, "n": n, "sem": sem,
            "ci_low": mean - half, "ci_high": mean + half, "ci_half": half}


def welch(a, b):
    """Two-sided Welch t-test of mean(a) vs mean(b). Positive t means a > b."""
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        nan = float("nan")
        return {"t": nan, "p": nan, "df": nan, "mean_diff": nan,
                "cohen_d": nan, "hedges_g": nan, "n_a": int(na), "n_b": int(nb)}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    res = _st.ttest_ind(a, b, equal_var=False)
    se2a, se2b = va / na, vb / nb
    denom = se2a ** 2 / (na - 1) + se2b ** 2 / (nb - 1)
    df = (se2a + se2b) ** 2 / denom if denom > 0 else float("nan")
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    diff = float(a.mean() - b.mean())
    d = diff / pooled if pooled > 0 else float("nan")
    g = d * (1 - 3 / (4 * (na + nb) - 9))
    return {"t": float(res.statistic), "p": float(res.pvalue), "df": float(df),
            "mean_diff": diff, "cohen_d": float(d), "hedges_g": float(g),
            "n_a": int(na), "n_b": int(nb)}


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (NaNs are ignored and kept as NaN)."""
    idx = [i for i, p in enumerate(pvalues) if not math.isnan(p)]
    order = sorted(idx, key=lambda i: pvalues[i])
    m = len(order)
    adj = [float("nan")] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def compare_table(values_by_label, pairs=None, metric="test_ppl"):
    """Welch tests for (a, b, description) pairs over per-seed metric values.

    ``values_by_label``: {label: [value per seed]}. Pairs with a missing side
    are reported with n=0 and NaN statistics, never silently dropped.
    """
    if pairs is None:
        from config import KEY_PAIRS
        pairs = KEY_PAIRS
    rows = []
    for a, b, desc in pairs:
        va, vb = values_by_label.get(a, []), values_by_label.get(b, [])
        w = welch(va, vb)
        sa, sb = summarize(va), summarize(vb)
        rows.append({"comparison": desc, "a": a, "b": b, "metric": metric,
                     "mean_a": sa["mean"], "std_a": sa["std"],
                     "mean_b": sb["mean"], "std_b": sb["std"], **w})
    for r, p in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = p
    return rows


def load_results(out_dir):
    """All result.json files under out_dir (one level of run directories)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*", "result.json"))):
        with open(path) as f:
            blob = json.load(f)
        res = dict(blob.get("results", blob)) if isinstance(blob, dict) else {}
        meta = blob.get("meta", {}) if isinstance(blob, dict) else {}
        # A run is reportable only if both the meta and the results say so.
        res["reportable"] = bool(res.get("reportable", True)) and \
            bool(meta.get("reportable", True))
        res.setdefault("run_id", meta.get("run_id"))
        res["_path"] = path
        rows.append(res)
    return rows


def _write_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def aggregate(out_dir, metrics=("val_ppl", "test_ppl", "val_loss", "test_loss"),
              pairs=None, write=True, include_nonreportable=False):
    """Group results by label (and non-default block size); write summary.csv and
    key_pairs.csv into out_dir. Non-reportable runs are excluded unless asked."""
    from config import BLOCK_SIZE
    runs = [r for r in load_results(out_dir)
            if include_nonreportable or r.get("reportable", True)]
    groups = {}
    for r in runs:
        key = r.get("label")
        cfg = r.get("config") or {}
        if cfg.get("expert_type") == "abr" and r.get("block_size") not in (None, BLOCK_SIZE):
            key = f"{key}@b{r.get('block_size')}"      # block-size sweep (P2.4)
        groups.setdefault(key, []).append(r)
    summary = []
    for key, rs in sorted(groups.items()):
        row = {"config": key, "n_runs": len(rs),
               "seeds": " ".join(str(r.get("seed")) for r in rs),
               "params": rs[0].get("params")}
        for m in metrics:
            s = summarize([r[m] for r in rs if r.get(m) is not None])
            row.update({f"{m}_mean": s["mean"], f"{m}_std": s["std"],
                        f"{m}_ci_low": s["ci_low"], f"{m}_ci_high": s["ci_high"]})
        summary.append(row)
    by_label = {k: [r["test_ppl"] for r in rs if r.get("test_ppl") is not None]
                for k, rs in groups.items()}
    table = compare_table(by_label, pairs, metric="test_ppl")
    if write:
        _write_csv(summary, os.path.join(out_dir, "summary.csv"))
        _write_csv(table, os.path.join(out_dir, "key_pairs.csv"))
    return summary, table


def format_table(rows):
    lines = [f"{'comparison':<40} {'t':>8} {'p':>8} {'p_holm':>8} {'g':>7}"]
    for r in rows:
        lines.append(f"{r['comparison']:<40} {r['t']:>+8.3f} {r['p']:>8.4f} "
                     f"{r['p_holm']:>8.4f} {r['hedges_g']:>+7.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Aggregate results into summary CSVs")
    ap.add_argument("out_dir", nargs="?", default="results/phase2")
    ap.add_argument("--include-nonreportable", action="store_true")
    a = ap.parse_args()
    _, tbl = aggregate(a.out_dir, include_nonreportable=a.include_nonreportable)
    print(format_table(tbl))
