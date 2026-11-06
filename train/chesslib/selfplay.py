"""Self-play game generation with MCTS, plus the replay buffer that stores it.

Each recorded position keeps:
  planes (21,8,8) u8 · legal action indices · search policy π over those legal
  moves · the game outcome z for the side to move (filled in at game end).
"""
from __future__ import annotations

import multiprocessing as mp
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import chess
import numpy as np
import torch

from .encoding import ACTION_SIZE, encode_board, legal_moves_and_indices, planes_to_float
from .mcts import MCTS, Evaluator, SearchConfig, Tree, choose_move
from .model import ChessNet


@dataclass
class SelfPlayConfig:
    sims: int = 200                    # simulations for a "full" search (recorded for training)
    fast_sims: int = 0                 # >0 enables playout-cap randomisation (KataGo)
    full_search_prob: float = 1.0      # ... probability a move gets the full search
    temp_plies: int = 20               # sample ∝ visits for this many plies, then play the best
    temperature: float = 1.0
    max_plies: int = 300               # adjudicate as a draw after this many plies
    resign_threshold: float = 0.95     # resign when own root value < -threshold twice in a row
    resign_disable_prob: float = 0.2   # fraction of games that never resign (checks false resigns)
    opening_random_plies: int = 0      # uniformly random moves before the search takes over
    concurrent_games: int = 64
    search: SearchConfig = field(default_factory=SearchConfig)


@dataclass
class GameRecord:
    planes: List[np.ndarray]
    legal: List[np.ndarray]
    pi: List[np.ndarray]
    turn: List[bool]
    result: str = "*"
    plies: int = 0
    resigned: bool = False
    would_resign: Optional[bool] = None   # resignation disabled: side that *would* have resigned

    def false_resign(self) -> Optional[bool]:
        """In a no-resign game: would resigning have thrown away a draw or win?"""
        if self.would_resign is None:
            return None
        lost = (self.result == "0-1") if self.would_resign == chess.WHITE else (self.result == "1-0")
        return not lost

    def outcomes(self) -> np.ndarray:
        """z for each recorded position (side-to-move view)."""
        w = {"1-0": 1.0, "0-1": -1.0}.get(self.result, 0.0)
        return np.array([w if t else -w for t in self.turn], np.float32)


def _random_opening(rng: random.Random, plies: int) -> chess.Board:
    while True:
        b = chess.Board()
        for _ in range(plies):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        if not b.is_game_over():
            return b


def play_selfplay_games(model: ChessNet, n_games: int, cfg: SelfPlayConfig, device="cpu",
                        seed: Optional[int] = None, progress=None) -> List[GameRecord]:
    """Play ``n_games`` self-play games, ``cfg.concurrent_games`` at a time."""
    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)
    mcts = MCTS(Evaluator(model, device), cfg.search, rng=rng_np)
    finished: List[GameRecord] = []
    started = 0
    live: List[dict] = []

    def start():
        nonlocal started
        started += 1
        board = _random_opening(rng_py, cfg.opening_random_plies) if cfg.opening_random_plies else chess.Board()
        return {"tree": Tree(board), "rec": GameRecord([], [], [], []), "low": [0, 0],
                "can_resign": rng_py.random() >= cfg.resign_disable_prob}

    while len(finished) < n_games:
        while len(live) < cfg.concurrent_games and started < n_games:
            live.append(start())
        full = [g for g in live if cfg.fast_sims <= 0 or rng_py.random() < cfg.full_search_prob]
        fast = [g for g in live if g not in full]
        mcts.search([g["tree"] for g in full], cfg.sims, add_noise=True)
        if fast:
            mcts.search([g["tree"] for g in fast], cfg.fast_sims, add_noise=False)
        is_full = set(id(g) for g in full)
        still = []
        for g in live:
            t, rec = g["tree"], g["rec"]
            b = t.board
            moves, pi = t.policy()
            if id(g) in is_full:
                rec.planes.append(encode_board(b))
                rec.legal.append(legal_moves_and_indices(b)[1].astype(np.int16))
                rec.pi.append(pi.astype(np.float16))
                rec.turn.append(b.turn)
            # resignation
            side = 0 if b.turn == chess.WHITE else 1
            q = t.root.q()
            g["low"][side] = g["low"][side] + 1 if q < -cfg.resign_threshold else 0
            if g["low"][side] >= 2 and not g["can_resign"] and rec.would_resign is None:
                rec.would_resign = b.turn
            if g["can_resign"] and g["low"][side] >= 2:
                rec.result, rec.resigned = ("0-1" if b.turn == chess.WHITE else "1-0"), True
            else:
                ply = len(b.move_stack)
                temp = cfg.temperature if ply < cfg.temp_plies + cfg.opening_random_plies else 0.0
                t.advance(choose_move(t, temp, rng_np))
                outcome = b.outcome(claim_draw=True)
                if outcome is not None:
                    rec.result = outcome.result()
                elif len(b.move_stack) >= cfg.max_plies:
                    rec.result = "1/2-1/2"
            if rec.result != "*":
                rec.plies = len(b.move_stack)
                finished.append(rec)
                if progress is not None:
                    progress(rec)
            else:
                still.append(g)
        live = still
    return finished


# ── multiprocessing: several CPU workers, each batching its own games ──────
def _worker(args):
    state_dict, model_config, n_games, cfg, seed, device = args
    torch.set_num_threads(1)
    model = ChessNet(**model_config)
    model.load_state_dict(state_dict)
    return play_selfplay_games(model.eval(), n_games, cfg, device, seed)


def parallel_selfplay(model: ChessNet, n_games: int, cfg: SelfPlayConfig, workers: int = 4,
                      seed: int = 0, pool=None, device: str = "mps") -> List[GameRecord]:
    """Split ``n_games`` across ``workers`` processes (spawned; safe in notebooks).

    Each worker batches ``cfg.concurrent_games`` games through its own copy of
    the network.  On Apple silicon a few workers sharing the GPU ("mps") is
    much faster than CPU workers; the tree search itself is Python and runs on
    the CPU in every worker, so more workers = more CPU cores in use.
    """
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    per = [n_games // workers + (1 if i < n_games % workers else 0) for i in range(workers)]
    jobs = [(sd, model.config, n, cfg, seed * 1000 + i, device) for i, n in enumerate(per) if n > 0]
    own = pool is None
    pool = pool or mp.get_context("spawn").Pool(len(jobs))
    try:
        parts = pool.map(_worker, jobs)
    finally:
        if own:
            pool.close()
            pool.join()
    return [g for p in parts for g in p]


# ── replay buffer ───────────────────────────────────────────────────────────
class ReplayBuffer:
    """Sliding window of the most recent ``capacity`` positions."""

    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.planes: List[np.ndarray] = []
        self.legal: List[np.ndarray] = []
        self.pi: List[np.ndarray] = []
        self.z: List[float] = []

    def __len__(self):
        return len(self.z)

    def add_games(self, games: List[GameRecord]) -> None:
        for g in games:
            self.planes += g.planes
            self.legal += g.legal
            self.pi += g.pi
            self.z += list(g.outcomes())
        extra = len(self.z) - self.capacity
        if extra > 0:
            for lst in (self.planes, self.legal, self.pi, self.z):
                del lst[:extra]

    def sample(self, batch_size: int, device="cpu", rng=None) -> Dict[str, torch.Tensor]:
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, len(self.z), batch_size)
        x = planes_to_float(np.stack([self.planes[i] for i in idx]))
        pi = np.zeros((batch_size, ACTION_SIZE), np.float32)
        mask = np.zeros((batch_size, ACTION_SIZE), bool)
        for r, i in enumerate(idx):
            pi[r, self.legal[i]] = self.pi[i]
            mask[r, self.legal[i]] = True
        z = np.array([self.z[i] for i in idx], np.float32)
        wdl = np.where(z > 0, 0, np.where(z < 0, 2, 1))
        return {"x": x.to(device), "pi": torch.from_numpy(pi).to(device),
                "mask": torch.from_numpy(mask).to(device), "z": torch.from_numpy(z).to(device),
                "wdl": torch.from_numpy(wdl).to(device)}
