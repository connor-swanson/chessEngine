from __future__ import annotations

import importlib.util
import os
import sys
import threading
from typing import Optional, Tuple

import chess

from .base import SyncBot

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRAIN_DIR = os.path.join(_REPO_ROOT, "train")
# Override with CHESS_CHECKPOINT_DIR (e.g. for testing with throwaway checkpoints).
CHECKPOINT_DIR = os.environ.get("CHESS_CHECKPOINT_DIR") or os.path.join(TRAIN_DIR, "checkpoints")


def checkpoint_path(name: str) -> str:
    return os.path.join(CHECKPOINT_DIR, name)


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def availability(checkpoint: str) -> Tuple[bool, Optional[str]]:
    """(available, reason_if_not) without importing torch."""
    if not torch_available():
        return False, "torch is not installed in this environment"
    if not os.path.isfile(checkpoint):
        return False, f"checkpoint not found: {checkpoint}"
    return True, None


class NeuralBot(SyncBot):
    """Trained network (train/chesslib) playing via MCTS (sims > 0) or raw policy (sims == 0).

    torch / chesslib are imported and the checkpoint loaded lazily on first use.
    A fresh search tree is built per call; searches are serialised by a lock.
    """

    def __init__(self, checkpoint: str, sims: int = 200, num_threads: int = 2):
        self.checkpoint = checkpoint
        self.sims = sims
        self.num_threads = num_threads
        self._model = None
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return availability(self.checkpoint)[0]

    def _load(self):
        ok, reason = availability(self.checkpoint)
        if not ok:
            raise RuntimeError(f"Neural bot unavailable: {reason}")
        if TRAIN_DIR not in sys.path:
            sys.path.insert(0, TRAIN_DIR)
        import torch
        from chesslib.model import load_checkpoint

        torch.set_num_threads(self.num_threads)
        self._model = load_checkpoint(self.checkpoint, device="cpu")

    def select_move(self, board: chess.Board) -> chess.Move:
        with self._lock:
            if self._model is None:
                self._load()
            import torch
            from chesslib.arena import MCTSPlayer, PolicyPlayer

            board = board.copy()
            with torch.no_grad():
                if self.sims <= 0:
                    return PolicyPlayer(self._model).move(board)
                return MCTSPlayer(self._model, sims=self.sims).move(board)
