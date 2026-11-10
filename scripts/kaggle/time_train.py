#!/usr/bin/env python3
"""Time a short train.py run: timestamps every `step=N` line and reports steps/s.

    python scripts/kaggle/time_train.py --config ABR-4L-CBSC [--steps 300] [--skip 100] [-- extra train args]

Runs `experiments/full_scale/train.py --config X --seed 42 --stop-after-steps N` with the
real 60k-step schedule (so nothing but the step count differs from a real run), marked
--debug (non-reportable), W&B disabled, outputs in $ABR_OUTPUT_DIR/train_out. Writes
$ABR_OUTPUT_DIR/timing.json. steps/s is measured between step `--skip` and the last step,
so start-up, data loading and Triton JIT compilation are excluded (reported separately).
The cheap periodic evals (--eval-batches 2 every 50 steps) are included; they cost far less
than the default eval every 1000 steps over 50 batches would per step.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

STEP_RE = re.compile(r"step=\s*(\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--skip", type=int, default=100)
    # train.py prints a `step=` line only at evaluations; make them frequent and cheap
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-batches", type=int, default=2)
    ap.add_argument("--total-steps", type=int, default=60_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("extra", nargs="*")
    a = ap.parse_args()
    out = os.environ.get("ABR_OUTPUT_DIR", "time_out")
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-u", "experiments/full_scale/train.py", "--config", a.config,
           "--seed", str(a.seed), "--stop-after-steps", str(a.steps), "--eval-every", str(a.eval_every), "--eval-batches", str(a.eval_batches),
           "--out", os.path.join(out, "train_out"), "--device", a.device, "--wandb", "disabled",
           "--debug", *a.extra]
    print("$", " ".join(cmd), flush=True)
    t0 = time.time()
    marks = []
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        sys.stdout.write(line)
        m = STEP_RE.search(line)
        if m:
            marks.append((int(m.group(1)), time.time() - t0))
    rc = p.wait()
    res = {"config": a.config, "seed": a.seed, "exit_code": rc, "cmd": cmd,
           "wall_s": round(time.time() - t0, 1), "marks": marks,
           "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import torch
        res["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    win = [(s, t) for s, t in marks if s >= a.skip]
    if len(win) >= 2:
        (s0, t0_), (s1, t1) = win[0], win[-1]
        sps = (s1 - s0) / (t1 - t0_)
        res.update(steps_per_s=round(sps, 3), window=[s0, s1],
                   est_hours_per_run=round(a.total_steps / sps / 3600, 2),
                   time_to_first_step_s=round(marks[0][1], 1) if marks else None)
    with open(os.path.join(out, "timing.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "marks"}), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
