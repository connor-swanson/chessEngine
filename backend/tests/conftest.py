import os
import sys

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_BACKEND)
for p in (_BACKEND, os.path.join(_ROOT, "train")):
    if p not in sys.path:
        sys.path.insert(0, p)
