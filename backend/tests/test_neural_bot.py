import os
import subprocess
import sys

import chess
import pytest

torch = pytest.importorskip("torch")

from app.bots import BOT_FACTORIES, available_bots, make_bot  # noqa: E402
from app.bots.neural_bot import NeuralBot  # noqa: E402


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from chesslib.model import ChessNet, save_checkpoint
    path = str(tmp_path_factory.mktemp("ckpt") / "tiny.pt")
    save_checkpoint(path, ChessNet())
    return path


@pytest.mark.parametrize("sims", [0, 16])
def test_neural_bot_legal_moves(ckpt, sims):
    bot = NeuralBot(ckpt, sims=sims)
    board = chess.Board()
    for _ in range(4):
        move = bot.select_move(board)
        assert move in board.legal_moves
        board.push(move)


def test_missing_checkpoint_raises_when_used(tmp_path):
    bot = NeuralBot(str(tmp_path / "nope.pt"), sims=0)
    with pytest.raises(RuntimeError, match="unavailable"):
        bot.select_move(chess.Board())


def test_registry_has_engines_and_reports_availability():
    for n in ("engine", "engine-fast", "engine-policy", "sft", "mcts", "random"):
        assert n in BOT_FACTORIES
    info = available_bots()
    assert info["random"]["available"] is True
    assert isinstance(make_bot("engine"), NeuralBot)


def test_registry_import_does_not_import_torch():
    backend = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = "import sys, app.bots; sys.exit(1 if 'torch' in sys.modules else 0)"
    r = subprocess.run([sys.executable, "-c", code], cwd=backend, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_bots_endpoint():
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("aiosqlite")
    from fastapi.testclient import TestClient
    from app.main import app
    r = TestClient(app).get("/bots")
    assert r.status_code == 200
    data = r.json()
    assert "random" in data["available"]
    assert {b["name"] for b in data["bots"]} >= {"engine", "random"}
