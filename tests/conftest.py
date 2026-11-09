import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_SCALE = os.path.join(ROOT, "experiments", "full_scale")
if FULL_SCALE not in sys.path:
    sys.path.insert(0, FULL_SCALE)
