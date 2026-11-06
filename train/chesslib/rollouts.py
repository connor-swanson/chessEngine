"""Batched game rollouts for the policy-gradient notebooks (REINFORCE, PPO).

The learner samples moves straight from its policy head (no search).  Many
games run at once so one network call serves every game whose learner is to
move.  For each learner move we keep what a policy-gradient update needs:

    planes · legal action indices · which legal move was played (``choice``)
    · log π_old(a|s) · V_old(s) · side to move · the final result.

Opponents are "move functions" that take a list of boards and return one move
per board — a frozen network (batched) or a scripted bot (looped).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import chess
import numpy as np
import torch

from .encoding import encode_board, legal_moves_and_indices, planes_to_float
from .model import ChessNet, wdl_to_scalar

MoveFn = Callable[[List[chess.Board]], List[chess.Move]]


@dataclass
class Trajectory:
    planes: List[np.ndarray] = field(default_factory=list)
    legal: List[np.ndarray] = field(default_factory=list)
    choice: List[int] = field(default_factory=list)      # index into legal
    logp: List[float] = field(default_factory=list)
    value: List[float] = field(default_factory=list)
    turn: List[bool] = field(default_factory=list)
    learner_color: Optional[bool] = None                 # None = learner played both sides
    result: str = "*"
    termination: str = ""
    plies: int = 0

    def learner_score(self) -> float:
        """+1 / 0 / -1 from the learner's side (self-play: from White's side)."""
        w = {"1-0": 1.0, "0-1": -1.0}.get(self.result, 0.0)
        return w if self.learner_color in (None, chess.WHITE) else -w

    def returns(self) -> np.ndarray:
        """Monte-Carlo return G for every recorded move (side-to-move view)."""
        w = {"1-0": 1.0, "0-1": -1.0}.get(self.result, 0.0)
        return np.array([w if t else -w for t in self.turn], np.float32)


class PolicySampler:
    """Batched move sampling from a network's policy head."""

    def __init__(self, model: ChessNet, device="cpu", temperature: float = 1.0, rng=None):
        # NOTE: moves ``model`` to ``device`` (in place, like ``nn.Module.to``).
        self.model, self.device, self.temperature = model.to(device), torch.device(device), temperature
        self.rng = rng or np.random.default_rng()

    @torch.inference_mode()
    def evaluate(self, boards: Sequence[chess.Board]):
        planes = np.stack([encode_board(b) for b in boards])
        ml = [legal_moves_and_indices(b) for b in boards]
        logits, wdl = self.model(planes_to_float(planes).to(self.device))
        values = wdl_to_scalar(wdl).float().cpu().numpy()
        logits = logits.float().cpu().numpy()
        out = []
        for i, (moves, idx) in enumerate(ml):
            lg = logits[i, idx].astype(np.float64)
            lg = lg - lg.max()
            logp = lg - np.log(np.exp(lg).sum())          # log π over legal moves (T = 1)
            out.append((moves, idx, logp))
        return planes, out, values

    def pick(self, logp: np.ndarray) -> int:
        if self.temperature <= 1e-6:
            return int(np.argmax(logp))
        p = np.exp(logp / self.temperature)
        return int(self.rng.choice(len(p), p=p / p.sum()))

    def __call__(self, boards: List[chess.Board]) -> List[chess.Move]:
        _, out, _ = self.evaluate(boards)
        return [moves[self.pick(lp)] for moves, _, lp in out]


def bot_move_fn(name: str, seed: int = 0) -> MoveFn:
    """Wrap a backend bot (e.g. "ab1") as a batched move function."""
    from .arena import BotPlayer
    bot = BotPlayer(name, seed)
    return lambda boards: [bot.move(b) for b in boards]


def random_move_fn(seed: int = 0) -> MoveFn:
    rng = random.Random(seed)
    return lambda boards: [rng.choice(list(b.legal_moves)) for b in boards]


def _opening(rng: random.Random, plies: int) -> chess.Board:
    while True:
        b = chess.Board()
        for _ in range(plies):
            b.push(rng.choice(list(b.legal_moves)))
        if not b.is_game_over():
            return b


def play_rollouts(learner: PolicySampler, n_games: int, opponent: Optional[MoveFn] = None,
                  opening_plies: int = 4, max_plies: int = 300, seed: int = 0) -> List[Trajectory]:
    """Play ``n_games`` simultaneously.

    ``opponent=None`` → self-play: the learner moves for both colours and every
    move is recorded.  Otherwise the learner gets White in half the games.
    """
    rng = random.Random(seed)
    games = []
    for i in range(n_games):
        color = None if opponent is None else (chess.WHITE if i % 2 == 0 else chess.BLACK)
        games.append((_opening(rng, opening_plies), Trajectory(learner_color=color)))
    live = list(range(n_games))
    while live:
        learn = [g for g in live if games[g][1].learner_color in (None, games[g][0].turn)]
        other = [g for g in live if g not in set(learn)]
        if learn:
            boards = [games[g][0] for g in learn]
            planes, out, values = learner.evaluate(boards)
            for k, g in enumerate(learn):
                board, tr = games[g]
                moves, idx, logp = out[k]
                c = learner.pick(logp)
                tr.planes.append(planes[k]); tr.legal.append(idx.astype(np.int16))
                tr.choice.append(c); tr.logp.append(float(logp[c])); tr.value.append(float(values[k]))
                tr.turn.append(board.turn)
                board.push(moves[c])
        if other:
            for g, mv in zip(other, opponent([games[g][0] for g in other])):
                games[g][0].push(mv)
        still = []
        for g in live:
            board, tr = games[g]
            out = board.outcome(claim_draw=True)
            if out is not None:
                tr.result, tr.termination = out.result(), out.termination.name
            elif len(board.move_stack) >= max_plies:
                tr.result, tr.termination = "1/2-1/2", "MOVE_CAP"
            if tr.result != "*":
                tr.plies = len(board.move_stack)
            else:
                still.append(g)
        live = still
    return [tr for _, tr in games]


def summarize(trajs: Sequence[Trajectory]) -> dict:
    s = np.array([t.learner_score() for t in trajs])
    terms = {}
    for t in trajs:
        terms[t.termination] = terms.get(t.termination, 0) + 1
    return {"games": len(trajs), "W": int((s > 0).sum()), "D": int((s == 0).sum()), "L": int((s < 0).sum()),
            "score": float(((s + 1) / 2).mean()), "avg_plies": float(np.mean([t.plies for t in trajs])),
            "terminations": terms}
