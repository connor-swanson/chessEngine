"""Interpretability helpers shared by the notebooks in ``interp/``.

The notebooks keep each *method* (integrated gradients, probing, patching, …)
in the notebook itself, where it is being taught.  This module holds the plumbing
around them: turning a chess question into a differentiable function of the
input planes, mapping attributions back onto real board squares, removing
pieces to make counterfactual positions, and drawing heatmaps on a board.

Coordinates
-----------
The network sees every position from the side to move's point of view (see
:mod:`chesslib.encoding`): for Black the board is mirrored top-to-bottom.  An
attribution tensor therefore lives in *encoded* coordinates.  Everything here
that returns a per-square map converts it back to **real squares**: a ``(64,)``
array indexed by python-chess square number (a1 = 0, h8 = 63).
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import chess
import numpy as np
import torch
import torch.nn as nn

from .encoding import NUM_PLANES, encode_board, legal_mask, move_to_index, planes_to_float
from .model import ChessNet, masked_log_softmax

# --------------------------------------------------------------------------- planes

PIECE_NAMES = ["pawn", "knight", "bishop", "rook", "queen", "king"]
PLANE_GROUPS: Dict[str, List[int]] = {
    **{f"our {p}s": [i] for i, p in enumerate(PIECE_NAMES)},
    **{f"their {p}s": [6 + i] for i, p in enumerate(PIECE_NAMES)},
    "repetition": [12, 13],
    "castling": [14, 15, 16, 17],
    "en passant": [18],
    "50-move clock": [19],
    "constant (ones)": [20],
}
#: planes that mark *where something is* on the board (pieces and the e.p. square)
SQUARE_PLANES = list(range(12)) + [18]
#: planes that hold one number for the whole board, broadcast to all 64 squares
GLOBAL_PLANES = [12, 13, 14, 15, 16, 17, 19, 20]


def board_tensor(board: chess.Board) -> torch.Tensor:
    """(1, 21, 8, 8) float input for ``board`` (uses its move stack for repetitions)."""
    return planes_to_float(encode_board(board))[None]


def to_real_squares(enc_map: np.ndarray, turn: chess.Color) -> np.ndarray:
    """(8, 8) map in encoded [rank][file] order -> (64,) map indexed by real square."""
    m = np.asarray(enc_map)
    if turn == chess.BLACK:
        m = m[::-1]
    return np.ascontiguousarray(m).reshape(64)


def square_map(attr: torch.Tensor, turn: chess.Color) -> np.ndarray:
    """Collapse a (21, 8, 8) attribution to one number per real square.

    Only the square planes (pieces, en passant) are summed: those are the parts
    of the input that *belong* to a square.  Global planes are reported
    separately by :func:`group_totals`.
    """
    a = attr.detach().reshape(NUM_PLANES, 8, 8).cpu().numpy()
    return to_real_squares(a[SQUARE_PLANES].sum(0), turn)


def group_totals(attr: torch.Tensor) -> Dict[str, float]:
    """Total attribution of each plane group (see :data:`PLANE_GROUPS`)."""
    a = attr.detach().reshape(NUM_PLANES, -1).sum(1).cpu().numpy()
    return {name: float(a[idx].sum()) for name, idx in PLANE_GROUPS.items()}


# --------------------------------------------------------------------------- targets

@dataclass
class Target:
    """A scalar the network computes, written as a function of the input planes.

    ``fn`` maps a batch ``(B, 21, 8, 8)`` to ``(B,)``.  ``for_board`` rebuilds
    the same question for a different board (used by occlusion, where removing
    a piece changes the set of legal moves).
    """
    name: str
    fn: Callable[[torch.Tensor], torch.Tensor]
    for_board: Callable[[chess.Board], Optional["Target"]]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


def value_target(model: nn.Module) -> Target:
    """Expected score for the side to move, P(win) − P(loss), in [-1, 1]."""
    def fn(x):
        p = torch.softmax(model(x)[1], dim=-1)
        return p[:, 0] - p[:, 2]
    t = Target("value", fn, lambda b: None)
    t.for_board = lambda b: t
    return t


def policy_target(model: nn.Module, board: chess.Board, move: chess.Move) -> Optional[Target]:
    """log π(move | board): the log-probability of ``move`` among the legal moves.

    We explain the log-probability rather than the raw logit because only the
    *differences* between legal-move logits matter — adding a constant to every
    logit changes nothing the engine does.  Returns ``None`` if ``move`` is not
    legal on ``board``.
    """
    if move not in board.legal_moves:
        return None
    idx = move_to_index(move, board.turn)
    mask = torch.as_tensor(legal_mask(board))[None]

    def fn(x):
        return masked_log_softmax(model(x)[0], mask.expand(x.shape[0], -1))[:, idx]
    return Target(f"log π({board.san(move)})", fn,
                  lambda b, m=move: policy_target(model, b, m))


def top_moves(model: nn.Module, board: chess.Board, k: int = 3):
    """[(move, prob), …] for the network's k most likely legal moves, and the WDL probs."""
    with torch.no_grad():
        p, v = model(board_tensor(board))
    mask = torch.as_tensor(legal_mask(board))[None]
    probs = masked_log_softmax(p, mask)[0].exp()
    best = torch.topk(probs, k)
    from .encoding import index_to_move
    moves = [(index_to_move(int(i), board), float(q)) for q, i in zip(best.values, best.indices)]
    return moves, torch.softmax(v, -1)[0].numpy()


# --------------------------------------------------------------------------- pieces

def removable_squares(board: chess.Board, keep: Sequence[int] = ()) -> List[int]:
    """Squares whose piece can be removed and still leave a legal position.

    Kings are never removed, and a removal that would put the side *not* to move
    in check (an impossible position) is skipped.
    """
    out = []
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING or sq in keep:
            continue
        b = remove_pieces(board, [sq])
        if not b.was_into_check():
            out.append(sq)
    return sorted(out)


def remove_pieces(board: chess.Board, squares: Sequence[int]) -> chess.Board:
    """Copy of ``board`` with the pieces on ``squares`` removed.

    python-chess clears the move stack when a piece is removed, so the new
    board has no history: its repetition planes are always zero.  (The position
    without the piece never occurred earlier in the game anyway.)
    """
    b = board.copy(stack=False)
    for sq in squares:
        b.remove_piece_at(sq)
    # castling rights for a rook that no longer exists are meaningless
    b.castling_rights = b.clean_castling_rights()
    return b


def kings_only_baseline(x: torch.Tensor) -> torch.Tensor:
    """``x`` with every piece except the two kings removed (other planes kept)."""
    b = x.clone()
    for p in [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 18]:
        b[:, p] = 0
    return b


# --------------------------------------------------------------------------- models

def randomized_copy(model: ChessNet, upto: str = "all", seed: int = 0) -> ChessNet:
    """Copy of ``model`` with weights re-initialised (Adebayo et al. 2018 sanity check).

    ``upto`` is one of the names in :func:`randomization_stages`; that stage and
    every stage *above* it (closer to the output) are re-initialised.  Call
    :func:`recalibrate_batchnorm` on the result before using it.
    """
    stages = randomization_stages(model)
    names = list(stages)
    if upto == "all":
        upto = names[-1]
    m = copy.deepcopy(model)
    torch.manual_seed(seed)
    m_stages = randomization_stages(m)
    for name in names[: names.index(upto) + 1]:
        for mod in m_stages[name]:
            for sub in mod.modules():
                if hasattr(sub, "reset_parameters"):
                    sub.reset_parameters()      # BatchNorm: also resets running stats
    return m.eval()


def randomization_stages(model: ChessNet) -> Dict[str, List[nn.Module]]:
    """Layers grouped from the output down: heads, block 6, …, block 1, stem."""
    stages = {"heads": [model.query, model.key, model.promo, model.value]}
    for i in reversed(range(len(model.tower))):
        stages[f"block {i + 1}"] = [model.tower[i]]
    stages["stem"] = [model.stem]
    return stages


@torch.no_grad()
def recalibrate_batchnorm(model: nn.Module, x: torch.Tensor) -> nn.Module:
    """Re-estimate BatchNorm running statistics on inputs ``x`` (model returned in eval mode).

    A randomly initialised network's BN layers have running mean 0 / variance 1,
    which lets activations grow layer after layer.  Re-estimating the statistics
    makes the random network a fair "same architecture, no learning" control.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None          # cumulative average over the batches below
    model.train()
    for i in range(0, len(x), 512):
        model(x[i:i + 512])
    return model.eval()


# --------------------------------------------------------------------------- data

def sample_positions(games, n: int, seed: int = 0, min_ply: int = 8) -> List[chess.Board]:
    """``n`` positions (with real move history) drawn from ``games`` (GameRecords).

    One position per game, at a uniformly random ply, skipping finished positions.
    """
    rng = random.Random(seed)
    games = [g for g in games if len(g.moves) > min_ply + 4]
    out = []
    for g in rng.sample(games, min(n, len(games))):
        ply = rng.randrange(min_ply, len(g.moves) - 2)
        b = chess.Board()
        for u in g.moves[:ply]:
            b.push_uci(u)
        if not b.is_game_over():
            out.append(b)
    return out


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (no SciPy needed); nan if either input is constant."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


# --------------------------------------------------------------------------- drawing

_GLYPH = {chess.KING: "♚", chess.QUEEN: "♛", chess.ROOK: "♜",
          chess.BISHOP: "♝", chess.KNIGHT: "♞", chess.PAWN: "♟"}
_LIGHT, _DARK = "#ece4d4", "#b9a68a"


def plot_board(ax, board: chess.Board, values: Optional[np.ndarray] = None, *,
               move: Optional[chess.Move] = None, vmax: Optional[float] = None,
               cmap: str = "RdBu_r", title: Optional[str] = None,
               hatch: Sequence[int] = (), orient: Optional[chess.Color] = None,
               coords: bool = True, fontsize: float = 22) -> None:
    """Draw ``board`` on ``ax`` with an optional per-square heatmap.

    ``values`` is a (64,) array by real square (nan = no value).  The colour
    scale is symmetric around zero (red = pushes the output up, blue = down).
    ``hatch`` squares get diagonal hatching (used for "could not measure").
    The board is drawn from ``orient``'s side (default: the side to move).
    """
    import matplotlib.patches as patches
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    orient = board.turn if orient is None else orient
    cm = plt.get_cmap(cmap)
    if values is not None:
        v = np.asarray(values, float)
        vmax = vmax or (np.nanmax(np.abs(v)) if np.isfinite(v).any() else 1.0) or 1.0

    def xy(sq):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        return (f, r) if orient == chess.WHITE else (7 - f, 7 - r)

    for sq in chess.SQUARES:
        x, y = xy(sq)
        light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
        ax.add_patch(patches.Rectangle((x, y), 1, 1, color=_LIGHT if light else _DARK, lw=0))
        if values is not None and np.isfinite(v[sq]) and v[sq] != 0:
            c = cm(0.5 + 0.5 * np.clip(v[sq] / vmax, -1, 1))
            ax.add_patch(patches.Rectangle((x, y), 1, 1, color=c, lw=0,
                                           alpha=0.25 + 0.7 * min(1.0, abs(v[sq]) / vmax)))
        if sq in hatch:
            ax.add_patch(patches.Rectangle((x, y), 1, 1, fill=False, hatch="///",
                                           edgecolor="#555", lw=0))
        piece = board.piece_at(sq)
        if piece:
            white = piece.color == chess.WHITE
            ax.text(x + 0.5, y + 0.47, _GLYPH[piece.piece_type], ha="center", va="center",
                    fontsize=fontsize, color="white" if white else "#111",
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="#111" if white else "#eee")])
    if move is not None:
        (x0, y0), (x1, y1) = xy(move.from_square), xy(move.to_square)
        ax.annotate("", xy=(x1 + 0.5, y1 + 0.5), xytext=(x0 + 0.5, y0 + 0.5),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.7",
                                    lw=3.2, color="#1a7f37", alpha=0.85, shrinkA=6, shrinkB=6))
    ax.set_xlim(0, 8); ax.set_ylim(0, 8); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if coords:
        files = "abcdefgh" if orient == chess.WHITE else "hgfedcba"
        ranks = "12345678" if orient == chess.WHITE else "87654321"
        for i in range(8):
            ax.text(i + 0.5, -0.22, files[i], ha="center", va="center", fontsize=8, color="#666")
            ax.text(-0.22, i + 0.5, ranks[i], ha="center", va="center", fontsize=8, color="#666")
    if title:
        ax.set_title(title, fontsize=10)


def show_boards(panels: Sequence[dict], ncols: int = 4, size: float = 3.0,
                suptitle: Optional[str] = None, shared_scale: bool = False):
    """Draw several boards in a grid.  Each panel is a dict of :func:`plot_board`
    keyword arguments plus ``board`` (and optionally ``values``).

    With ``shared_scale`` every heatmap uses the same colour range, so colours
    can be compared across panels; otherwise each panel is scaled to its own
    largest value (compare *patterns*, not intensities).
    """
    import matplotlib.pyplot as plt
    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * (size + 0.35)),
                             squeeze=False)
    vmax = None
    if shared_scale:
        vals = [np.abs(np.asarray(p["values"], float)) for p in panels if p.get("values") is not None]
        vmax = max(np.nanmax(v) for v in vals) if vals else None
    for ax, p in zip(axes.flat, panels):
        p = dict(p)
        board = p.pop("board")
        values = p.pop("values", None)
        p.setdefault("vmax", vmax)
        p.setdefault("fontsize", size * 6.5)
        plot_board(ax, board, values, **p)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig
