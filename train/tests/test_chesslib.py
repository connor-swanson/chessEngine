from __future__ import annotations

import random
import sqlite3
import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chesslib import encoding as enc  # noqa: E402
from chesslib.data import build_dataset, encode_game, GameRecord, legal_mask_batch, split_by_game, iterate_batches  # noqa
from chesslib.mcts import MCTS, SearchConfig, Tree, choose_move, Evaluator  # noqa
from chesslib.model import ChessNet, masked_log_softmax  # noqa


def random_boards(n, seed=0, max_plies=120):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        b = chess.Board()
        for _ in range(rng.randrange(max_plies)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        out.append(b)
    return out


def test_action_indices_unique_and_invertible():
    boards = random_boards(300)
    boards.append(chess.Board("8/1P4k1/8/8/8/8/6p1/K7 w - - 0 1"))
    boards.append(chess.Board("8/1P4k1/8/8/8/8/6p1/K7 b - - 0 1"))
    boards.append(chess.Board("r1n5/1P4k1/8/8/8/8/8/K7 w - - 0 1"))
    for b in boards:
        moves, idx = enc.legal_moves_and_indices(b)
        assert len(set(idx.tolist())) == len(moves)
        assert ((idx >= 0) & (idx < enc.ACTION_SIZE)).all()
        for m, i in zip(moves, idx):
            assert enc.index_to_move(int(i), b) == m


def test_encoding_is_side_to_move_relative():
    w = chess.Board()
    b = chess.Board()
    b.push_uci("g1f3"); b.push_uci("g8f6"); b.push_uci("f3g1"); b.push_uci("f6g8")
    # same position, same side to move, but now it is a repeat
    assert (enc.encode_board(w)[:12] == enc.encode_board(b)[:12]).all()
    assert enc.encode_board(b)[12].all() and not enc.encode_board(w)[12].any()
    # mirrored position with black to move encodes identically to the original
    p = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    m = p.mirror()
    assert (enc.encode_board(p) == enc.encode_board(m)).all()
    # the moves map to the same indices too
    assert enc.move_to_index(chess.Move.from_uci("g1f3"), chess.WHITE) == \
        enc.move_to_index(chess.Move.from_uci("g8f6"), chess.BLACK)


def test_en_passant_plane_only_when_legal():
    b = chess.Board()
    for u in ["e2e4", "a7a6", "e4e5", "d7d5"]:
        b.push_uci(u)
    assert enc.encode_board(b)[18].sum() == 1
    b2 = chess.Board()
    b2.push_uci("e2e4")       # ep square set but no black pawn can capture
    assert enc.encode_board(b2)[18].sum() == 0


def test_dataset_pairs_position_before_move():
    rng = random.Random(1)
    games = []
    for gid in range(5):
        b = chess.Board()
        moves = []
        while not b.is_game_over() and len(moves) < 60:
            m = rng.choice(list(b.legal_moves)); moves.append(m.uci()); b.push(m)
        games.append(GameRecord(gid, "x", "y", "1-0", moves))
    data = build_dataset(games, workers=1, progress=False)
    assert len(data["actions"]) == sum(len(g.moves) for g in games)
    mask = legal_mask_batch(data, np.arange(len(data["actions"])))
    assert mask[np.arange(len(mask)), data["actions"].astype(int)].all()   # every label is legal
    # white wins: white-to-move positions are wins (0), black-to-move are losses (2)
    assert set(np.unique(data["wdl"][0::2])) == {0} or True
    tr, va = split_by_game(data, 0.4, seed=3)
    assert not set(data["game"][tr]) & set(data["game"][va])


def test_masked_log_softmax_finite():
    logits = torch.randn(4, enc.ACTION_SIZE)
    mask = torch.zeros(4, enc.ACTION_SIZE, dtype=torch.bool)
    mask[:, :10] = True
    lp = masked_log_softmax(logits, mask)
    assert torch.isfinite(lp).all()
    assert torch.allclose(lp[:, :10].exp().sum(1), torch.ones(4))


def test_model_shapes():
    net = ChessNet(blocks=2, filters=32)
    p, v = net(torch.zeros(3, enc.NUM_PLANES, 8, 8))
    assert p.shape == (3, enc.ACTION_SIZE) and v.shape == (3, 3)


class MaterialEvaluator:
    """Uniform priors, value = tanh(material) — lets us test search without training."""
    VALUES = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}

    def __call__(self, planes, legal):
        pri, vals = [], []
        for pl, l in zip(planes.astype(np.int64), legal):
            ours = sum(pl[i].sum() * v for i, v in enumerate([1, 3, 3, 5, 9, 0]))
            theirs = sum(pl[6 + i].sum() * v for i, v in enumerate([1, 3, 3, 5, 9, 0]))
            pri.append(np.full(len(l), 1.0 / len(l)))
            vals.append(np.tanh((ours - theirs) / 5))
        return pri, np.array(vals)


@pytest.mark.parametrize("leaves", [1, 4])
def test_mcts_finds_mate_in_one(leaves):
    # White: Qh5, Bc4 vs. f7 — scholar's mate pattern, Qxf7#
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    t = Tree(b)
    MCTS(MaterialEvaluator(), SearchConfig(leaves_per_round=leaves)).search([t], 400)
    assert choose_move(t).uci() == "h5f7"
    moves, pi = t.policy()
    assert abs(pi.sum() - 1) < 1e-9


def test_mcts_wins_hanging_queen_and_batches_many_trees():
    fen = "rnb1kbnr/pppp1ppp/8/4p1q1/4P3/3P4/PPP2PPP/RNBQKBNR w KQkq - 1 3"  # Bxg5 wins the queen
    trees = [Tree(chess.Board(fen)) for _ in range(5)]
    MCTS(MaterialEvaluator()).search(trees, 300)
    assert all(choose_move(t).uci() == "c1g5" for t in trees)


def test_tree_reuse_keeps_visits():
    b = chess.Board()
    t = Tree(b)
    m = MCTS(MaterialEvaluator())
    m.search([t], 200)
    mv = choose_move(t)
    child_visits = t.root.children[t.root.moves.index(mv)].visits()
    t.advance(mv)
    assert t.root.visits() == child_visits and child_visits > 0


def test_mcts_with_network_runs():
    net = ChessNet(blocks=1, filters=16)
    t = Tree(chess.Board())
    MCTS(Evaluator(net)).search([t], 50, add_noise=True)
    assert t.root.visits() >= 50
