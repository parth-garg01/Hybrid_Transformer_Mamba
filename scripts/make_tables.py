#!/usr/bin/env python3
"""Regenerate every Phase 2 table from the committed Kaggle result files (PRD P2.6).

Reads results/kaggle/queue/<session>/output/<run>/<timestamp>/result.json for every
session in results/kaggle/queue.json that is not marked discarded, keeps reportable
runs that reached the full step budget, and writes into results/phase2/:

    runs.csv        one row per run (source path + SHA-256 of its result.json)
    summary.csv     per config: mean, std (ddof=1), 95% CI of val/test PPL, params, speed
    key_pairs.csv   Welch tests for config.KEY_PAIRS on test PPL (Holm over confirmatory)
    routing.csv     per config and layer: mean fraction of blocks sent to attention
    sweep.csv       block-size sweep of the pre-registered config (block 32 = main runs)
    tables.md       the same tables in Markdown

Output is deterministic: running it twice on the same files gives identical bytes.

    python scripts/make_tables.py [--allow-missing]
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments", "full_scale"))
import config as C          # noqa: E402
import stats as S           # noqa: E402

QUEUE = os.path.join(ROOT, "results", "kaggle", "queue.json")
OUT = os.path.join(ROOT, "results", "phase2")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect():
    with open(QUEUE) as f:
        sessions = json.load(f).get("sessions", {})
    discarded = {n for n, s in sessions.items() if s.get("status") == "discarded"}
    runs, skipped = {}, []
    pattern = os.path.join(ROOT, "results", "kaggle", "queue", "*", "output", "*", "*", "result.json")
    for path in sorted(glob.glob(pattern)):
        rel = os.path.relpath(path, ROOT)
        session = rel.split(os.sep)[3]
        if session in discarded:
            skipped.append((rel, "discarded session"))
            continue
        with open(path) as f:
            blob = json.load(f)
        r, meta = blob["results"], blob.get("meta", {})
        if not (r.get("reportable") and meta.get("reportable", True)) or r.get("debug"):
            skipped.append((rel, "not reportable"))
            continue
        if r.get("steps") != C.TRAIN_STEPS:
            skipped.append((rel, f"steps={r.get('steps')}"))
            continue
        key = (r["label"], int(r["seed"]), int(r.get("block_size") or C.BLOCK_SIZE))
        if key in runs:
            raise SystemExit(f"duplicate result for {key}: {runs[key]['_path']} and {rel}")
        r["_path"], r["_sha256"], r["_session"] = rel, sha256(path), session
        runs[key] = r
    return runs, skipped


def expected():
    exp = {(lab, int(s), C.BLOCK_SIZE) for lab, s, _ in C.planned_runs()}
    lab = getattr(C, "SWEEP_LABEL", None)
    return exp, lab


def attn_frac(r):
    st = r.get("final_routing_stats") or {}
    return st.get("transformer_frac_hard", st.get("transformer_pct"))


def fmt(x, nd=2):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def write_csv(rows, name):
    path = os.path.join(OUT, name)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-missing", action="store_true",
                    help="write tables even if planned runs are missing (marks them in tables.md)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    runs, skipped = collect()
    exp, _ = expected()
    main_keys = {k for k in runs if k[2] == C.BLOCK_SIZE}
    missing = sorted(exp - main_keys)
    unexpected = sorted(main_keys - exp)
    if (missing or unexpected) and not a.allow_missing:
        raise SystemExit(f"main grid mismatch: missing={missing} unexpected={unexpected}")

    # runs.csv
    run_rows = []
    for (lab, seed, blk), r in sorted(runs.items()):
        run_rows.append({"label": lab, "seed": seed, "block_size": blk, "params": r.get("params"),
                         "val_ppl": r["val_ppl"], "test_ppl": r["test_ppl"],
                         "attn_frac": attn_frac(r), "gpu_hours": r.get("gpu_hours"),
                         "steps_per_s": r["steps"] / r["wall_time_s"] if r.get("wall_time_s") else None,
                         "n_sessions": r.get("n_sessions"), "session": r["_session"],
                         "result_json": r["_path"], "sha256": r["_sha256"]})
    write_csv(run_rows, "runs.csv")

    # summary.csv (main grid, block 32)
    by_label = defaultdict(list)
    for (lab, seed, blk), r in sorted(runs.items()):
        if blk == C.BLOCK_SIZE:
            by_label[lab].append(r)
    summary = []
    for lab, rs in sorted(by_label.items(), key=lambda kv: S.summarize(r["test_ppl"] for r in kv[1])["mean"]):
        row = {"config": lab, "n": len(rs), "seeds": " ".join(str(r["seed"]) for r in rs),
               "params": rs[0].get("params")}
        for m in ("val_ppl", "test_ppl"):
            s = S.summarize(r[m] for r in rs)
            row.update({f"{m}_mean": s["mean"], f"{m}_std": s["std"],
                        f"{m}_ci_low": s["ci_low"], f"{m}_ci_high": s["ci_high"]})
        fr = [attn_frac(r) for r in rs if attn_frac(r) is not None]
        row["attn_frac_mean"] = sum(fr) / len(fr) if fr else None
        row["steps_per_s_mean"] = sum(r["steps"] / r["wall_time_s"] for r in rs) / len(rs)
        row["gpu_hours_mean"] = sum(r.get("gpu_hours") or 0 for r in rs) / len(rs)
        summary.append(row)
    write_csv(summary, "summary.csv")

    # key_pairs.csv (test PPL, the primary metric; val PPL reported alongside)
    test_vals = {lab: [r["test_ppl"] for r in rs] for lab, rs in by_label.items()}
    val_vals = {lab: [r["val_ppl"] for r in rs] for lab, rs in by_label.items()}
    pairs = S.compare_table(test_vals, C.KEY_PAIRS, metric="test_ppl")
    vpairs = S.compare_table(val_vals, C.KEY_PAIRS, metric="val_ppl")
    for p, v in zip(pairs, vpairs):
        p["val_mean_diff"], p["val_p"] = v["mean_diff"], v["p"]
    write_csv(pairs, "key_pairs.csv")

    # routing.csv (learned/random routing configs, per layer)
    routing = []
    for lab, rs in sorted(by_label.items()):
        layers = defaultdict(list)
        for r in rs:
            for L in (r.get("final_routing_stats") or {}).get("per_layer", []):
                layers[L["layer"]].append(L.get("transformer_frac_hard", L.get("transformer_pct")))
        for li, v in sorted(layers.items()):
            s = S.summarize(v)
            routing.append({"config": lab, "layer": li, "n": s["n"],
                            "attn_frac_mean": s["mean"], "attn_frac_std": s["std"],
                            "attn_frac_min": min(v), "attn_frac_max": max(v)})
    write_csv(routing, "routing.csv")

    # sweep.csv
    sweep_labels = sorted({k[0] for k in runs if k[2] != C.BLOCK_SIZE})
    sweep = []
    for lab in sweep_labels:
        for blk in C.BLOCK_SIZE_SWEEP:
            rs = [r for (l, s, b), r in sorted(runs.items())
                  if l == lab and b == blk and (blk != C.BLOCK_SIZE or s in C.BLOCK_SWEEP_SEEDS)]
            if not rs:
                sweep.append({"config": lab, "block_size": blk, "n": 0})
                continue
            sv, st = S.summarize(r["val_ppl"] for r in rs), S.summarize(r["test_ppl"] for r in rs)
            sweep.append({"config": lab, "block_size": blk, "n": len(rs),
                          "seeds": " ".join(str(r["seed"]) for r in rs),
                          "val_ppl_mean": sv["mean"], "val_ppl_std": sv["std"],
                          "test_ppl_mean": st["mean"], "test_ppl_std": st["std"],
                          "steps_per_s_mean": sum(r["steps"] / r["wall_time_s"] for r in rs) / len(rs)})
    write_csv(sweep, "sweep.csv")

    # tables.md
    md = ["# Phase 2 tables", "",
          "Generated by `scripts/make_tables.py` from the result files listed in `runs.csv`. Do not edit by hand.", ""]
    if missing or unexpected:
        md += [f"**Incomplete grid.** Missing: {missing}. Unexpected: {unexpected}.", ""]
    md += ["## Main results (block 32, 60k steps)", "",
           "| Config | n | Params | Val PPL | Test PPL (mean ± sd) | Test 95% CI | Attn frac | steps/s |",
           "|---|---|---|---|---|---|---|---|"]
    for r in summary:
        md.append(f"| {r['config']} | {r['n']} | {r['params'] / 1e6:.2f}M | {fmt(r['val_ppl_mean'])} | "
                  f"{fmt(r['test_ppl_mean'])} ± {fmt(r['test_ppl_std'])} | "
                  f"[{fmt(r['test_ppl_ci_low'])}, {fmt(r['test_ppl_ci_high'])}] | "
                  f"{fmt(r['attn_frac_mean'])} | {fmt(r['steps_per_s_mean'])} |")
    md += ["", "## Key comparisons (test PPL, Welch t-test; Holm over the confirmatory family)", "",
           "| Comparison | Family | Δ test PPL (a − b) | t | df | p | p (Holm) | Hedges' g |",
           "|---|---|---|---|---|---|---|---|"]
    for p in pairs:
        md.append(f"| {p['comparison']} | {p['family']} | {fmt(p['mean_diff'])} | {fmt(p['t'])} | "
                  f"{fmt(p['df'], 1)} | {fmt(p['p'], 4)} | {fmt(p['p_holm'], 4)} | {fmt(p['hedges_g'])} |")
    md += ["", "## Routing: fraction of blocks sent to attention, per layer", "",
           "| Config | Layer | n | Mean | sd | Min | Max |", "|---|---|---|---|---|---|---|"]
    for r in routing:
        md.append(f"| {r['config']} | {r['layer']} | {r['n']} | {fmt(r['attn_frac_mean'])} | "
                  f"{fmt(r['attn_frac_std'])} | {fmt(r['attn_frac_min'])} | {fmt(r['attn_frac_max'])} |")
    if sweep:
        md += ["", "## Block-size sweep (seeds 42, 123; block 32 = main runs with those seeds)", "",
               "| Config | Block | n | Val PPL | Test PPL | steps/s |", "|---|---|---|---|---|---|"]
        for r in sweep:
            md.append(f"| {r['config']} | {r['block_size']} | {r['n']} | {fmt(r.get('val_ppl_mean'))} | "
                      f"{fmt(r.get('test_ppl_mean'))} | {fmt(r.get('steps_per_s_mean'))} |")
    if skipped:
        md += ["", "## Result files not used", ""] + [f"- `{p}`: {why}" for p, why in skipped]
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {OUT}: {len(runs)} runs ({len(main_keys)} main), {len(skipped)} skipped, "
          f"missing={len(missing)}")


if __name__ == "__main__":
    main()
