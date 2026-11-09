"""
LEGACY wrapper. The old script trained four ABR configs for 60k steps into
results_60k/. Phase 2 trains every model for 60k steps, so this is now just a
shortcut for train_ablation.py.

    python train_abr_long.py            # Phase 2 learned-routing ABR variants (2L and 4L)
    python train_abr_long.py --legacy   # old: 4 legacy ABR configs x [42,123,7], 60k steps,
                                        # results_60k/

Other options are passed through to train_ablation.py / train.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C        # noqa: E402
import train_ablation     # noqa: E402


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--configs" not in argv:
        if "--legacy" in argv:
            labels = C.LEGACY_LONG_LABELS
        else:
            labels = [c["label"] for c in C.PHASE2_CONFIGS
                      if c["expert_type"] == "abr" and c["routing_mode"] == "learned"]
        argv += ["--configs", *labels]
    if "--legacy" in argv:
        if "--steps" not in argv:
            argv += ["--steps", "60000"]
        if "--out" not in argv:
            argv += ["--out", "results_60k"]
    return train_ablation.main(argv)


if __name__ == "__main__":
    main()
