"""
Sweep driver: runs train.py for many (config, seed) pairs, then aggregates.

Each run is one ``train.main([...])`` call, so it gets provenance, W&B,
checkpoint/resume and the full val/test evaluation. Re-running the same
command skips finished runs and resumes an interrupted one.

    python train_ablation.py                                  # all PHASE2_CONFIGS, PRD seeds
    python train_ablation.py --configs ABR-4L-CBSC Jamba-1:3 --max-hours 11.5 --wandb offline
    python train_ablation.py --block-size-sweep ABR-4L-CBSC  # P2.4: 8/16/32/64 x 3 seeds
    python train_ablation.py --legacy                         # old 9 configs x 3 seeds, 30k steps

``--legacy`` reproduces the old script: LEGACY_CONFIGS, seeds [42, 123, 7],
30k steps, 50M train tokens, aux loss from the detached last layer, 200
random val batches. Output defaults to results/legacy.
Unknown options (``--device``, ``--wandb``, ``--debug``, ``--allow-gru-fallback``,
``--steps`` ...) are passed through to train.py.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C   # noqa: E402
import train         # noqa: E402


def plan(args):
    """List of (label, seed, block_size)."""
    if args.block_size_sweep:
        seeds = args.seeds or C.BLOCK_SWEEP_SEEDS
        return [(args.block_size_sweep, s, b) for b in C.BLOCK_SIZE_SWEEP for s in seeds]
    pool = C.LEGACY_CONFIGS if args.legacy else C.PHASE2_CONFIGS
    labels = args.configs or [c["label"] for c in pool]
    runs = []
    for label in labels:
        C.get_config(label, legacy=args.legacy)            # validate early
        seeds = args.seeds or (C.SEEDS_OTHER if args.legacy else C.seeds_for(label))
        runs += [(label, s, None) for s in seeds]
    return runs


def run_sweep(runs, out, legacy, max_hours, passthrough):
    t0 = time.time()
    for i, (label, seed, block) in enumerate(runs, 1):
        argv = ["--config", label, "--seed", str(seed), "--out", out, *passthrough]
        if legacy:
            argv.append("--legacy")
        if block is not None:
            argv += ["--block-size", str(block)]
        if max_hours is not None:
            left = max_hours - (time.time() - t0) / 3600
            if left <= 0:
                print("[sweep] time budget used up; rerun the same command to continue")
                return False
            argv += ["--max-hours", f"{left:.4f}"]
        print(f"\n[sweep] ({i}/{len(runs)}) {label} seed={seed}"
              + (f" block={block}" if block else ""), flush=True)
        status = train.main(argv)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if status != train.DONE:
            print("[sweep] run incomplete (time budget); rerun the same command to resume")
            return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--block-size-sweep", metavar="LABEL", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-hours", type=float, default=None,
                    help="total budget for this session across all runs")
    args, passthrough = ap.parse_known_args(argv)
    out = args.out or ("results/legacy" if args.legacy else C.PHASE2_RESULTS_DIR)
    runs = plan(args)
    print(f"[sweep] {len(runs)} runs -> {out}")
    finished = run_sweep(runs, out, args.legacy, args.max_hours, passthrough)
    if finished:
        import stats
        _, table = stats.aggregate(out, include_nonreportable=("--debug" in passthrough
                                                              or "--allow-gru-fallback"
                                                              in passthrough))
        print(f"\n[sweep] wrote {out}/summary.csv and {out}/key_pairs.csv")
        if not args.legacy:
            print(stats.format_table(table))
    return finished


if __name__ == "__main__":
    main()
