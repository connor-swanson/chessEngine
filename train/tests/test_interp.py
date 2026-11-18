"""Tests for chesslib.interp (the interpretability notebooks' plumbing)."""
from __future__ import annotations

import sys
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chesslib.interp import (board_tensor, group_totals, kings_only_baseline, policy_target,  # noqa: E402
                             rank_corr, randomized_copy, recalibrate_batchnorm, remove_pieces,
                             removable_squares, square_map, value_target, PLANE_GROUPS)
from chesslib.model import ChessNet  # noqa: E402


def small_net():
    torch.manual_seed(0)
    return ChessNet(blocks=2, filters=16).eval()


def test_square_map_lands_on_real_squares_for_both_colours():
    for fen in ["4k3/8/8/8/8/8/1N6/4K3 w - - 0 1", "4k3/1n6/8/8/8/8/8/4K3 b - - 0 1"]:
        b = chess.Board(fen)
        x = board_tensor(b)[0]
        m = square_map(x, b.turn)              # the input itself: 1 where a piece is
        assert set(np.nonzero(m)[0]) == set(b.piece_map())


def test_group_totals_cover_every_plane_once():
    planes = sorted(i for idx in PLANE_GROUPS.values() for i in idx)
    assert planes == list(range(21))
    x = torch.ones(21, 8, 8)
    assert sum(group_totals(x).values()) == 21 * 64


def test_removable_squares_skip_kings_and_illegal_positions():
    # Black to move; the white bishop on c1 blocks the a1 rook from the white king on d1
    b = chess.Board("4k3/8/8/8/8/8/8/r1BK4 b - - 0 1")
    sq = removable_squares(b)
    assert chess.C1 not in sq                               # removal leaves White (not to move) in check
    assert chess.A1 in sq
    assert all(b.piece_at(s).piece_type != chess.KING for s in sq)


def test_remove_pieces_cleans_castling_and_leaves_original():
    b = chess.Board()
    b.push_uci("e2e4")
    b2 = remove_pieces(b, [chess.H1])
    assert b2.piece_at(chess.H1) is None and b2.turn == chess.BLACK
    assert not b2.has_kingside_castling_rights(chess.WHITE)
    assert b.piece_at(chess.H1) is not None                 # original untouched


def test_policy_target_is_none_for_illegal_move_and_differentiable_otherwise():
    net, b = small_net(), chess.Board()
    assert policy_target(net, b, chess.Move.from_uci("e2e5")) is None
    t = policy_target(net, b, chess.Move.from_uci("e2e4"))
    x = board_tensor(b).requires_grad_(True)
    t(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # for_board asks the same question of another board
    b2 = remove_pieces(b, [chess.G1])
    assert t.for_board(b2) is not None


def test_kings_only_baseline_keeps_kings_and_global_planes():
    x = board_tensor(chess.Board())
    base = kings_only_baseline(x)
    assert base[0, 5].sum() == 1 and base[0, 11].sum() == 1
    assert base[0, :5].sum() == 0 and base[0, 6:11].sum() == 0
    assert torch.equal(base[0, 12:18], x[0, 12:18]) and torch.equal(base[0, 19:], x[0, 19:])


def test_randomized_copy_only_touches_requested_stages():
    net = small_net()
    heads = randomized_copy(net, "heads", seed=1)
    assert torch.equal(heads.stem[0].weight, net.stem[0].weight)
    assert not torch.equal(heads.value[-1].weight, net.value[-1].weight)
    full = randomized_copy(net, "all", seed=1)
    assert not torch.equal(full.stem[0].weight, net.stem[0].weight)
    x = torch.cat([board_tensor(chess.Board()) for _ in range(4)])
    out = value_target(recalibrate_batchnorm(full, x))(x)
    assert torch.isfinite(out).all() and not full.training


def test_rank_corr():
    assert rank_corr([1, 2, 3], [10, 20, 30]) == 1.0
    assert rank_corr([1, 2, 3], [3, 2, 1]) == -1.0
    assert np.isnan(rank_corr([1, 1, 1], [1, 2, 3]))
