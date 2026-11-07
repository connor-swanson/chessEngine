import os
import sys

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_BACKEND)
for p in (_BACKEND, os.path.join(_ROOT, "train")):
    if p not in sys.path:
        sys.path.insert(0, p)

# API tests write to a throwaway database, never data/chess.db.
import tempfile

os.environ.setdefault(
    "CHESS_DB_URL",
    "sqlite+aiosqlite:///" + os.path.join(tempfile.mkdtemp(prefix="chess-test-"), "test.db"),
)
