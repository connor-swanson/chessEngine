#!/usr/bin/env python
"""Parallel bot-vs-bot game generator.

Workers play games synchronously (Bot.select_move) in separate processes; the
parent process is the only SQLite writer and stores games in the same schema the
backend uses (games + positions, source='simulation').

    python train/generate_games.py --white ab2 --black heuristic --games 200 \
        --workers 8 --swap-colors --random-opening-plies 4 --seed 0

Game termination uses claim_draw=True (threefold repetition / 50-move claims end
the game) so shuffling games don't run to the ply cap.  Games that reach
--max-plies are stored as '1/2-1/2' and counted as "adjudicated".
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sqlite3
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "backend"))

import chess  # noqa: E402
import chess.pgn  # noqa: E402

from app.bots import BOT_FACTORIES, make_bot  # noqa: E402

DEFAULT_DB = os.path.join(_ROOT, "data", "chess.db")

# Mirrors backend/app/models.py (SQLAlchemy-generated DDL for SQLite).
SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id INTEGER NOT NULL,
    white_player VARCHAR(50) NOT NULL,
    black_player VARCHAR(50) NOT NULL,
    result VARCHAR(10),
    pgn TEXT,
    created_at DATETIME NOT NULL,
    source VARCHAR(20),
    PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS ix_games_id ON games (id);
CREATE INDEX IF NOT EXISTS ix_games_source ON games (source);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER NOT NULL,
    game_id INTEGER NOT NULL,
    move_number INTEGER NOT NULL,
    fen VARCHAR(150) NOT NULL,
    move_uci VARCHAR(10),
    is_terminal BOOLEAN NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY(game_id) REFERENCES games (id)
);
CREATE INDEX IF NOT EXISTS ix_positions_id ON positions (id);
"""

GameRecord = Tuple[str, str, str, str, List[str], List[str], bool]


def play_game(job: Tuple[str, str, int, int, int]) -> GameRecord:
    """Play one game.  Returns (white, black, result, pgn, uci_moves, fens, capped).

    fens[k] is the position after move k (fens[0] = start position).
    """
    white, black, seed, opening_plies, max_plies = job
    rng = random.Random(seed)
    bots = {
        chess.WHITE: make_bot(white, rng.getrandbits(32)),
        chess.BLACK: make_bot(black, rng.getrandbits(32)),
    }
    board = chess.Board()
    fens = [board.fen()]
    ucis: List[str] = []
    capped = False
    while True:
        if board.is_game_over(claim_draw=True):
            break
        if len(ucis) >= max_plies:
            capped = True
            break
        if len(ucis) < opening_plies:
            move = rng.choice(list(board.legal_moves))
        else:
            move = bots[board.turn].select_move(board)
        board.push(move)
        ucis.append(move.uci())
        fens.append(board.fen())
    result = "1/2-1/2" if capped else board.result(claim_draw=True)
    game = chess.pgn.Game.from_board(board)
    game.headers["White"], game.headers["Black"] = white, black
    game.headers["Result"] = result
    pgn = game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False))
    return white, black, result, pgn, ucis, fens, capped


def ensure_db(path: str) -> sqlite3.Connection:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(games)")]
    if "source" not in cols:  # same migration as backend/app/database.py
        conn.execute("ALTER TABLE games ADD COLUMN source VARCHAR(20)")
    conn.commit()
    return conn


def save_games(conn: sqlite3.Connection, records: List[GameRecord]) -> None:
    for white, black, result, pgn, ucis, fens, _ in records:
        cur = conn.execute(
            "INSERT INTO games (white_player, black_player, result, pgn, created_at, source) "
            "VALUES (?, ?, ?, ?, ?, 'simulation')",
            (white, black, result, pgn, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")),
        )
        gid = cur.lastrowid
        n = len(ucis)
        rows = [
            (gid, k, fens[k], ucis[k - 1] if k > 0 else None, 1 if (k == n and n > 0) else 0)
            for k in range(n + 1)
        ]
        conn.executemany(
            "INSERT INTO positions (game_id, move_number, fen, move_uci, is_terminal) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
    conn.commit()


def generate(white: str, black: str, games: int, workers: int, db: str = DEFAULT_DB,
             random_opening_plies: int = 0, max_plies: int = 400, seed: Optional[int] = None,
             swap_colors: bool = False, commit_every: int = 50, verbose: bool = True) -> dict:
    for name in (white, black):
        if name not in BOT_FACTORIES:
            raise SystemExit(f"unknown bot {name!r}; choices: {sorted(BOT_FACTORIES)}")
    base = seed if seed is not None else random.SystemRandom().getrandbits(31)
    jobs = []
    for i in range(games):
        w, b = (black, white) if (swap_colors and i % 2 == 1) else (white, black)
        jobs.append((w, b, base * 1_000_003 + i, random_opening_plies, max_plies))

    conn = ensure_db(db)
    stats = {"games": 0, "w": 0, "d": 0, "l": 0, "capped": 0,
             "first_bot_wins": 0, "second_bot_wins": 0}
    pending: List[GameRecord] = []
    t0 = time.time()

    def handle(rec: GameRecord) -> None:
        w, b, result, _, _, _, capped = rec
        stats["games"] += 1
        stats["capped"] += int(capped)
        if result == "1-0":
            stats["w"] += 1
        elif result == "0-1":
            stats["l"] += 1
        else:
            stats["d"] += 1
        # Wins from the point of view of the --white bot ("first") vs --black bot.
        if (result == "1-0" and w == white) or (result == "0-1" and b == white and w != white):
            stats["first_bot_wins"] += 1
        elif result in ("1-0", "0-1"):
            stats["second_bot_wins"] += 1
        pending.append(rec)
        if len(pending) >= commit_every:
            save_games(conn, pending)
            pending.clear()
        if verbose and (stats["games"] % 10 == 0 or stats["games"] == games):
            el = time.time() - t0
            print(f"[{stats['games']}/{games}] {stats['games']/el:.2f} games/s  "
                  f"white W/D/L {stats['w']}/{stats['d']}/{stats['l']}  capped {stats['capped']}",
                  flush=True)

    if workers <= 1:
        for j in jobs:
            handle(play_game(j))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for rec in pool.imap_unordered(play_game, jobs, chunksize=1):
                handle(rec)
    if pending:
        save_games(conn, pending)
    conn.close()
    stats["elapsed"] = time.time() - t0
    stats["games_per_s"] = stats["games"] / max(stats["elapsed"], 1e-9)
    if verbose:
        print(f"done: {stats['games']} games in {stats['elapsed']:.1f}s "
              f"({stats['games_per_s']:.2f} games/s); {white} wins {stats['first_bot_wins']}, "
              f"{black} wins {stats['second_bot_wins']}, "
              f"draws {stats['games'] - stats['first_bot_wins'] - stats['second_bot_wins']}; "
              f"{stats['capped']} adjudicated by the {max_plies}-ply cap")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--white", required=True)
    p.add_argument("--black", required=True)
    p.add_argument("--games", type=int, required=True)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--random-opening-plies", type=int, default=0)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--swap-colors", action="store_true",
                   help="alternate which bot plays white (odd games swapped)")
    a = p.parse_args()
    generate(a.white, a.black, a.games, a.workers, a.db, a.random_opening_plies,
             a.max_plies, a.seed, a.swap_colors)


if __name__ == "__main__":
    main()
