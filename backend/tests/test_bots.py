from __future__ import annotations

import random
import sqlite3

import chess
import pytest

from app.bots import BOT_FACTORIES, BOT_REGISTRY, make_bot, AlphaBetaBot, HeuristicBot

_NEURAL = ("engine", "sft", "mcts")  # need checkpoints; covered in test_neural_bot.py
NON_SF = [n for n in BOT_FACTORIES if not n.startswith(("stockfish",) + _NEURAL)]


def random_positions(n, seed=0):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.randint(0, 60)):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
        if not b.is_game_over():
            out.append(b)
    return out


def test_registry_imports_without_stockfish_spawn():
    assert "stockfish" in BOT_REGISTRY and "ab2-e25" in BOT_REGISTRY


@pytest.mark.parametrize("name", NON_SF)
def test_legal_and_no_mutation(name):
    bot = make_bot(name, seed=1)
    for b in random_positions(30):
        before_fen, before_stack = b.fen(), list(b.move_stack)
        m = bot.select_move(b)
        assert m in b.legal_moves
        assert b.fen() == before_fen and list(b.move_stack) == before_stack


def test_ab2_mate_in_1_and_hanging_queen():
    for _ in range(5):
        b = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
        assert make_bot("ab2").select_move(b).uci() == "a1a8"
    # Black queen on d5 hangs to the white knight on c3.
    b = chess.Board("4k3/8/8/3q4/8/2N5/8/4K3 w - - 0 1")
    assert make_bot("ab2").select_move(b).uci() == "c3d5"


def test_heuristic_takes_mate_in_1():
    b = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    assert make_bot("heuristic", 0).select_move(b).uci() == "a1a8"


def test_ab3_mate_in_2():
    # Rook ladder: Rb2 + Ra1 vs Kh8 is mate in 2 (1.Rb7 then Ra8#).
    b = chess.Board("7k/8/8/8/8/8/1R6/R5K1 w - - 0 1")
    bot = AlphaBetaBot(3, seed=0)
    m1 = bot.select_move(b)
    b.push(m1)
    # Whatever Black replies, White must have mate in 1 next.
    for reply in list(b.legal_moves):
        b.push(reply)
        m2 = AlphaBetaBot(2, seed=0).select_move(b)
        b.push(m2)
        assert b.is_checkmate(), (m1, reply, m2)
        b.pop(); b.pop()


def test_heuristic_tiebreak_random():
    bot = HeuristicBot(seed=3)
    moves = {bot.select_move(chess.Board()).uci() for _ in range(50)}
    assert len(moves) > 1


def test_alphabeta_root_tiebreak_random():
    bot = AlphaBetaBot(2, seed=3, noise_cp=10)
    moves = {bot.select_move(chess.Board()).uci() for _ in range(40)}
    assert len(moves) > 1


def test_generator_end_to_end(tmp_path):
    from generate_games import generate
    db = str(tmp_path / "t.db")
    stats = generate("ab1", "heuristic", 4, 2, db=db, random_opening_plies=4,
                     max_plies=120, seed=5, swap_colors=True, verbose=False)
    assert stats["games"] == 4
    conn = sqlite3.connect(db)
    games = conn.execute("SELECT id, result, pgn, source, created_at FROM games").fetchall()
    assert len(games) == 4
    for gid, result, pgn, source, created in games:
        assert source == "simulation" and result in ("1-0", "0-1", "1/2-1/2") and pgn and created
        rows = conn.execute(
            "SELECT move_number, fen, move_uci, is_terminal FROM positions "
            "WHERE game_id=? ORDER BY move_number", (gid,)).fetchall()
        assert [r[0] for r in rows] == list(range(len(rows)))
        assert rows[0][2] is None and rows[0][1] == chess.STARTING_FEN
        assert [r[3] for r in rows] == [0] * (len(rows) - 1) + [1]
        for k in range(len(rows) - 1):
            b = chess.Board(rows[k][1])
            m = chess.Move.from_uci(rows[k + 1][2])
            assert m in b.legal_moves
            b.push(m)
            assert b.fen() == rows[k + 1][1]
    whites = {r[0] for r in conn.execute("SELECT white_player FROM games")}
    assert whites == {"ab1", "heuristic"}  # colors swapped
