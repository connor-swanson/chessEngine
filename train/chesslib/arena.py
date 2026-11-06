"""Players, matches and Elo estimates.

Why this module exists: an evaluation that plays two deterministic players from
the start position replays the same game over and over.  Here every match uses
*paired random openings* — each opening is played twice with colours swapped —
so games differ and colour advantage cancels out.

Players are described by small picklable dicts ("specs") so matches can run in
parallel worker processes::

    {"type": "random"}
    {"type": "bot", "name": "ab2"}                         # backend/app/bots registry
    {"type": "stockfish", "elo": 1320, "movetime": 0.1}    # UCI_LimitStrength
    {"type": "stockfish", "skill": 0, "depth": 1}
    {"type": "policy", "checkpoint": "...pt"}              # raw network, no search
    {"type": "mcts", "checkpoint": "...pt", "sims": 200}
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import chess
import numpy as np
import torch

from .data import PROJECT_ROOT
from .mcts import MCTS, Evaluator, SearchConfig, Tree, choose_move

_BACKEND = str(PROJECT_ROOT / "backend")


# ── players ─────────────────────────────────────────────────────────────────
class Player:
    def move(self, board: chess.Board) -> chess.Move:
        raise NotImplementedError

    def close(self):
        pass


class RandomPlayer(Player):
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def move(self, board):
        return self.rng.choice(list(board.legal_moves))


class BotPlayer(Player):
    def __init__(self, name: str, seed=None):
        if _BACKEND not in sys.path:
            sys.path.insert(0, _BACKEND)
        from app.bots import BOT_REGISTRY  # noqa
        self.bot = BOT_REGISTRY[name]
        if seed is not None:
            for obj in (self.bot, getattr(self.bot, "base_bot", None)):
                if obj is not None and hasattr(obj, "rng"):
                    obj.rng.seed(seed)
        self.name = name

    def move(self, board):
        return self.bot.select_move(board.copy())


class StockfishPlayer(Player):
    def __init__(self, elo: Optional[int] = None, skill: Optional[int] = None,
                 movetime: Optional[float] = 0.1, depth: Optional[int] = None, nodes: Optional[int] = None):
        import chess.engine
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        opts = {"Threads": 1, "Hash": 16}
        if elo is not None:
            opts.update({"UCI_LimitStrength": True, "UCI_Elo": elo})
        if skill is not None:
            opts["Skill Level"] = skill
        self.engine.configure(opts)
        self.limit = chess.engine.Limit(time=movetime if depth is None and nodes is None else None,
                                        depth=depth, nodes=nodes)

    def move(self, board):
        return self.engine.play(board, self.limit).move

    def close(self):
        self.engine.quit()


class PolicyPlayer(Player):
    """The raw network: play the most probable legal move, no search."""

    def __init__(self, model, temperature: float = 0.0, seed=None):
        self.eval = Evaluator(model)
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def move(self, board):
        from .encoding import encode_board, legal_moves_and_indices
        moves, idx = legal_moves_and_indices(board)
        pri, _ = self.eval(encode_board(board)[None], [idx])
        p = pri[0]
        if self.temperature <= 0:
            return moves[int(np.argmax(p))]
        p = p ** (1 / self.temperature)
        return moves[int(self.rng.choice(len(moves), p=p / p.sum()))]


class MCTSPlayer(Player):
    """Network + tree search; reuses its tree between moves."""

    def __init__(self, model, sims: int = 200, leaves_per_round: int = 8, device="cpu", **search_kw):
        cfg = SearchConfig(leaves_per_round=leaves_per_round, **search_kw)
        self.mcts = MCTS(Evaluator(model, device), cfg)
        self.sims = sims
        self.tree: Optional[Tree] = None

    def _sync(self, board: chess.Board) -> Tree:
        t = self.tree
        if t is not None:
            mine, theirs = t.board.move_stack, board.move_stack
            if len(theirs) >= len(mine) and theirs[:len(mine)] == mine and \
                    t.board.root().fen() == board.root().fen():
                for m in theirs[len(mine):]:
                    t.advance(m)
                return t
        self.tree = Tree(board.copy())
        return self.tree

    def move(self, board):
        t = self._sync(board)
        self.mcts.search([t], self.sims, add_noise=False)
        return choose_move(t, 0.0)


_MODEL_CACHE: Dict[str, object] = {}


def _load(path):
    if path not in _MODEL_CACHE:
        from .model import load_checkpoint
        _MODEL_CACHE[path] = load_checkpoint(path, "cpu")
    return _MODEL_CACHE[path]


def make_player(spec: dict, seed=None) -> Player:
    s = dict(spec)
    kind = s.pop("type")
    s.pop("label", None)
    if kind == "random":
        return RandomPlayer(seed)
    if kind == "bot":
        return BotPlayer(s["name"], seed)
    if kind == "stockfish":
        return StockfishPlayer(**s)
    if kind == "policy":
        return PolicyPlayer(_load(s.pop("checkpoint")), seed=seed, **s)
    if kind == "mcts":
        return MCTSPlayer(_load(s.pop("checkpoint")), **s)
    raise ValueError(kind)


def label(spec: dict) -> str:
    if "label" in spec:
        return spec["label"]
    s = {k: v for k, v in spec.items() if k not in ("type", "checkpoint")}
    name = spec["type"] if spec["type"] != "bot" else spec["name"]
    if spec["type"] == "bot":
        s.pop("name")
    return name + ("(" + ",".join(f"{k}={v}" for k, v in s.items()) + ")" if s else "")


# ── matches ─────────────────────────────────────────────────────────────────
def random_openings(n: int, plies: int = 4, seed: int = 0) -> List[List[str]]:
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(plies):
            b.push(rng.choice(list(b.legal_moves)))
        if not b.is_game_over():
            out.append([m.uci() for m in b.move_stack])
    return out


def play_game(a: Player, b: Player, opening: Sequence[str], a_white: bool, max_plies: int = 400):
    """Returns (score for A in {1, 0.5, 0}, termination name, plies, pgn-ish move list)."""
    board = chess.Board()
    for u in opening:
        board.push_uci(u)
    while True:
        out = board.outcome(claim_draw=True)
        if out is not None:
            break
        if len(board.move_stack) >= max_plies:
            return 0.5, "MOVE_CAP", len(board.move_stack), [m.uci() for m in board.move_stack]
        player = a if board.turn == a_white else b
        board.push(player.move(board))
    if out.winner is None:
        score = 0.5
    else:
        score = 1.0 if out.winner == a_white else 0.0
    return score, out.termination.name, len(board.move_stack), [m.uci() for m in board.move_stack]


_WORKER_PLAYERS: Dict[str, Player] = {}


def _game_job(args):
    spec_a, spec_b, opening, a_white, max_plies, seed = args
    torch.set_num_threads(1)
    players = []
    for spec in (spec_a, spec_b):
        key = json.dumps(spec, sort_keys=True)
        if key not in _WORKER_PLAYERS:
            _WORKER_PLAYERS[key] = make_player(spec, seed)
        players.append(_WORKER_PLAYERS[key])
    return play_game(players[0], players[1], opening, a_white, max_plies)


@dataclass
class MatchResult:
    a: str
    b: str
    wins: int = 0
    draws: int = 0
    losses: int = 0
    terminations: Dict[str, int] = field(default_factory=dict)
    games: List[Tuple] = field(default_factory=list)

    @property
    def n(self):
        return self.wins + self.draws + self.losses

    @property
    def score(self):
        return (self.wins + 0.5 * self.draws) / max(self.n, 1)

    def elo_diff(self) -> Tuple[float, float]:
        """Elo of A minus B, with a ~95% half-width."""
        return elo_from_score(self.score, self.n)

    def __repr__(self):
        d, ci = self.elo_diff()
        return (f"{self.a} vs {self.b}: +{self.wins} ={self.draws} -{self.losses} "
                f"(score {self.score:.1%}, Elo diff {d:+.0f} ± {ci:.0f})")


def elo_from_score(score: float, n: int) -> Tuple[float, float]:
    s = min(max(score, 0.5 / max(n, 1)), 1 - 0.5 / max(n, 1))
    diff = -400 * math.log10(1 / s - 1)
    se = math.sqrt(s * (1 - s) / max(n, 1))
    lo = -400 * math.log10(1 / min(max(s - 1.96 * se, 1e-3), 1 - 1e-3) - 1)
    hi = -400 * math.log10(1 / min(max(s + 1.96 * se, 1e-3), 1 - 1e-3) - 1)
    return diff, (hi - lo) / 2


def play_match(spec_a: dict, spec_b: dict, n_games: int = 40, workers: int = 8, opening_plies: int = 4,
               seed: int = 0, max_plies: int = 400, pool=None, verbose: bool = True) -> MatchResult:
    """``n_games`` (rounded up to even) from paired random openings, run in parallel."""
    openings = random_openings((n_games + 1) // 2, opening_plies, seed)
    jobs = [(spec_a, spec_b, o, aw, max_plies, seed * 7919 + i)
            for i, o in enumerate(openings) for aw in (True, False)]
    res = MatchResult(label(spec_a), label(spec_b))
    if workers <= 1:
        outs = [_game_job(j) for j in jobs]
    else:
        own = pool is None
        pool = pool or mp.get_context("spawn").Pool(workers)
        try:
            outs = pool.map(_game_job, jobs, chunksize=1)
        finally:
            if own:
                # terminate, not close: workers hold Stockfish engine threads (non-daemon),
                # so a graceful close()+join() would wait forever.
                pool.terminate(); pool.join()
    for (score, term, plies, moves), job in zip(outs, jobs):
        res.wins += score == 1.0
        res.draws += score == 0.5
        res.losses += score == 0.0
        res.terminations[term] = res.terminations.get(term, 0) + 1
        res.games.append((job[3], score, term, plies, moves))
    if verbose:
        print(res)
    return res


def fit_elo(results: Sequence[Tuple[float, MatchResult]]) -> Tuple[float, float]:
    """Maximum-likelihood rating of one player from matches against anchors.

    ``results`` = [(anchor_elo, MatchResult with our player as A), ...].
    Returns (elo, ~95% half-width).
    """
    def nll(r):
        tot = 0.0
        for anchor, m in results:
            p = 1 / (1 + 10 ** ((anchor - r) / 400))
            p = min(max(p, 1e-9), 1 - 1e-9)
            s = m.wins + 0.5 * m.draws
            tot -= s * math.log(p) + (m.n - s) * math.log(1 - p)
        return tot
    grid = np.arange(-500, 3500, 1.0)
    vals = np.array([nll(r) for r in grid])
    best = float(grid[int(np.argmin(vals))])
    ok = grid[vals <= vals.min() + 1.92]          # likelihood-ratio 95% interval
    return best, float((ok.max() - ok.min()) / 2)
