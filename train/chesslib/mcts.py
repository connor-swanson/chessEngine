"""Batched PUCT Monte-Carlo Tree Search (AlphaZero-style).

One search tree per game.  Many trees advance together: every *round*, each
tree walks down to one (or a few) leaves, all leaves from all trees are scored
by the network in **one batch**, and the results are backed up.  Batching is
what makes a Python MCTS usable — one forward pass of 256 boards costs about
the same as one pass of a single board.

Conventions
-----------
* Every value is from the point of view of the **side to move** at the node it
  belongs to.  Edge statistics ``W[a]`` on a node are from that node's mover's
  view, so selection maximises ``Q + U`` directly and backup flips the sign
  once per ply.
* Terminal positions are scored exactly (mate = -1 for the side to move,
  draws = 0).  Inside the tree a position repeated once is scored as a draw —
  standard engine practice; it stops the search wasting time on shuffling.
* Unvisited moves get a pessimistic "first play urgency" Q of
  ``V(node) - fpu_reduction`` so the search focuses on the policy's top moves.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import chess
import numpy as np
import torch

from .encoding import encode_board, legal_moves_and_indices, planes_to_float
from .model import ChessNet


# ── network evaluation ──────────────────────────────────────────────────────
class Evaluator:
    """Scores a batch of positions: masked-softmax priors + scalar value."""

    def __init__(self, model: ChessNet, device="cpu"):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.positions_evaluated = 0

    @torch.inference_mode()
    def __call__(self, planes: np.ndarray, legal: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
        x = planes_to_float(planes).to(self.device)
        logits, wdl = self.model(x)
        lmax = max(len(l) for l in legal)
        idx = np.zeros((len(legal), lmax), np.int64)
        pad = np.ones((len(legal), lmax), bool)
        for i, l in enumerate(legal):
            idx[i, :len(l)] = l
            pad[i, :len(l)] = False
        g = logits.gather(1, torch.from_numpy(idx).to(self.device))
        g = g.masked_fill(torch.from_numpy(pad).to(self.device), -1e9)
        p = torch.softmax(g, dim=1).float().cpu().numpy()
        w = torch.softmax(wdl, dim=1).float().cpu().numpy()
        self.positions_evaluated += len(legal)
        return [p[i, :len(l)] for i, l in enumerate(legal)], w[:, 0] - w[:, 2]


# ── tree ─────────────────────────────────────────────────────────────────────
class Node:
    __slots__ = ("moves", "P", "P_raw", "N", "W", "children", "v", "terminal", "pending")

    def __init__(self):
        self.moves: Optional[List[chess.Move]] = None   # set on expansion
        self.P = self.P_raw = self.N = self.W = None
        self.children: Optional[list] = None
        self.v = 0.0               # network value of this node
        self.terminal: Optional[float] = None
        self.pending = False       # waiting for a network evaluation

    @property
    def expanded(self) -> bool:
        return self.P is not None

    def visits(self) -> int:
        return 0 if self.N is None else int(self.N.sum())

    def q(self) -> float:
        """Mean value of this node from its mover's view (network value if unvisited)."""
        n = self.visits()
        return self.v if n == 0 else float(self.W.sum() / n)


def terminal_value(board: chess.Board, moves: list, root: bool) -> Optional[float]:
    """Exact value for the side to move if the game is over here, else None."""
    if not moves:
        return -1.0 if board.is_check() else 0.0
    if board.is_insufficient_material() or board.halfmove_clock >= 100:
        return 0.0
    if board.is_repetition(3 if root else 2):
        return 0.0
    return None


@dataclass
class SearchConfig:
    c_puct: float = 2.0
    fpu_reduction: float = 0.25
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    leaves_per_round: int = 1     # >1 uses virtual loss to batch leaves within one tree
    virtual_loss: float = 1.0


class Tree:
    """Search tree for one game.  ``board`` is the live game board."""

    def __init__(self, board: chess.Board):
        self.board = board
        self.root = Node()

    def advance(self, move: chess.Move) -> None:
        """Play ``move`` on the board and reuse the matching subtree."""
        nxt = None
        if self.root.expanded:
            for i, m in enumerate(self.root.moves):
                if m == move:
                    nxt = self.root.children[i]
                    break
        self.board.push(move)
        self.root = nxt if (nxt is not None and nxt.expanded) else Node()
        if self.root.expanded:
            self.root.P = self.root.P_raw   # drop the old noise

    def policy(self) -> Tuple[List[chess.Move], np.ndarray]:
        """Root moves and visit-count distribution π."""
        n = self.root.N.astype(np.float64)
        return self.root.moves, n / n.sum()


def _expand(node: Node, board: chess.Board, root: bool) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Generate moves; mark terminal; return (planes, legal_idx) to evaluate or None."""
    moves, idx = legal_moves_and_indices(board)
    node.terminal = terminal_value(board, moves, root)
    if node.terminal is not None:
        return None
    node.moves = moves
    node.pending = True
    return encode_board(board), idx


def _apply_eval(node: Node, priors: np.ndarray, value: float) -> None:
    node.P_raw = node.P = priors.astype(np.float64)
    n = len(priors)
    node.N = np.zeros(n, np.float64)
    node.W = np.zeros(n, np.float64)
    node.children = [None] * n
    node.v = float(value)
    node.pending = False


def _select(node: Node, cfg: SearchConfig) -> int:
    total = node.N.sum()
    u = cfg.c_puct * node.P * math.sqrt(total + 1.0) / (1.0 + node.N)
    fpu = node.q() - cfg.fpu_reduction
    q = np.where(node.N > 0, node.W / np.maximum(node.N, 1e-9), fpu)
    return int(np.argmax(q + u))


def _backup(path, value: float, vl: float) -> None:
    """``value`` is from the leaf's mover view; flip once per edge going up."""
    for node, a in reversed(path):
        value = -value
        node.N[a] += 1.0 - vl
        node.W[a] += value + vl


class MCTS:
    def __init__(self, evaluator: Callable, config: Optional[SearchConfig] = None, rng=None):
        self.evaluate = evaluator
        self.cfg = config or SearchConfig()
        self.rng = rng or np.random.default_rng()

    def _prepare_roots(self, trees: Sequence[Tree], add_noise: bool) -> None:
        todo = []
        for t in trees:
            r = t.root
            if not r.expanded and r.terminal is None:
                job = _expand(r, t.board, root=True)
                if job is not None:
                    todo.append((r, job))
        if todo:
            pri, val = self.evaluate(np.stack([j[0] for _, j in todo]), [j[1] for _, j in todo])
            for (node, _), p, v in zip(todo, pri, val):
                _apply_eval(node, p, v)
        if add_noise:
            a, eps = self.cfg.dirichlet_alpha, self.cfg.dirichlet_eps
            for t in trees:
                r = t.root
                if r.expanded and len(r.P_raw) > 1:
                    r.P = (1 - eps) * r.P_raw + eps * self.rng.dirichlet([a] * len(r.P_raw))

    def search(self, trees: Sequence[Tree], n_sims: int, add_noise: bool = False) -> None:
        """Run until every (non-terminal) root has at least ``n_sims`` visits."""
        cfg = self.cfg
        self._prepare_roots(trees, add_noise)
        active = [t for t in trees if t.root.expanded]
        vl = cfg.virtual_loss if cfg.leaves_per_round > 1 else 0.0
        while active:
            jobs, paths, nodes = [], [], []
            for t in active:
                for _ in range(cfg.leaves_per_round):
                    node, board, path = t.root, t.board, []
                    while True:
                        a = _select(node, cfg)
                        node.N[a] += vl
                        node.W[a] -= vl
                        path.append((node, a))
                        board.push(node.moves[a])
                        child = node.children[a]
                        if child is None:
                            child = node.children[a] = Node()
                        if child.terminal is not None:
                            _backup(path, child.terminal, vl)
                            break
                        if child.pending:                 # collision with another leaf this round
                            for n_, a_ in path:
                                n_.N[a_] -= vl
                                n_.W[a_] += vl
                            break
                        if not child.expanded:
                            job = _expand(child, board, root=False)
                            if job is None:
                                _backup(path, child.terminal, vl)
                            else:
                                jobs.append(job); paths.append(path); nodes.append(child)
                            break
                        node = child
                    for _ in path:
                        board.pop()
            if jobs:
                pri, val = self.evaluate(np.stack([j[0] for j in jobs]), [j[1] for j in jobs])
                for node, path, p, v in zip(nodes, paths, pri, val):
                    _apply_eval(node, p, v)
                    _backup(path, float(v), vl)
            active = [t for t in active if t.root.N.sum() < n_sims]


# ── convenience: play one move with search ───────────────────────────────────
def choose_move(tree: Tree, temperature: float = 0.0, rng=None) -> chess.Move:
    moves, pi = tree.policy()
    if temperature <= 1e-6:
        return moves[int(np.argmax(pi))]
    rng = rng or np.random.default_rng()
    p = pi ** (1.0 / temperature)
    return moves[int(rng.choice(len(moves), p=p / p.sum()))]
