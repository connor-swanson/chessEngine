"""Turn simulated games in ``data/chess.db`` into training arrays.

The DB stores one ``positions`` row per ply: row *k* holds the FEN **after**
move *k* together with ``move_uci`` = move *k* (row 0 is the start position).
A training example needs the position **before** a move paired with that move,
so we simply replay each game's move list from the start — which also gives
the encoder the real move history it needs for the repetition planes.

Arrays produced by :func:`build_dataset` (N = number of positions):

==============  ==================  ===========================================
key             shape / dtype       meaning
==============  ==================  ===========================================
states          (N, 21, 8, 8) u8    encoded position before the move
actions         (N,) int16          action index of the move actually played
legal_flat      (sum L,) int16      legal action indices of every position,
legal_offsets   (N + 1,) int64      ... concatenated; position i owns
                                    legal_flat[offsets[i]:offsets[i+1]]
wdl             (N,) int8           game result for the side to move:
                                    0 = win, 1 = draw, 2 = loss
game            (N,) int32          index into ``games`` (for splitting)
ply             (N,) int16          ply number of the position
==============  ==================  ===========================================
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import chess
import numpy as np
import torch

from .encoding import ACTION_SIZE, encode_board, legal_moves_and_indices, move_to_index, planes_to_float

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "data" / "chess.db"


@dataclass
class GameRecord:
    id: int
    white: str
    black: str
    result: str
    moves: List[str] = field(default_factory=list)


def load_games(db_path=DEFAULT_DB, white: Optional[Sequence[str]] = None,
               black: Optional[Sequence[str]] = None, players: Optional[Sequence[str]] = None,
               source: Optional[str] = "simulation", min_id: int = 0,
               limit: Optional[int] = None) -> List[GameRecord]:
    """Load finished games (with their move lists) from the SQLite DB.

    ``white`` / ``black`` restrict the bot on each side; ``players`` keeps games
    where *both* sides are in the list.  ``min_id`` lets a notebook skip older
    data generated before a change.
    """
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    where, params = ["result IN ('1-0', '0-1', '1/2-1/2')", "id >= ?"], [min_id]
    if source:
        where.append("source = ?"); params.append(source)
    for col, names in (("white_player", white), ("black_player", black)):
        if names:
            where.append(f"{col} IN ({','.join('?' * len(names))})"); params += list(names)
    if players:
        q = ",".join("?" * len(players))
        where.append(f"white_player IN ({q}) AND black_player IN ({q})"); params += list(players) * 2
    sql = f"SELECT id, white_player, black_player, result FROM games WHERE {' AND '.join(where)} ORDER BY id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    games = {r[0]: GameRecord(*r) for r in db.execute(sql, params)}
    if games:
        ids = list(games)
        for i in range(0, len(ids), 900):
            chunk = ids[i:i + 900]
            rows = db.execute(
                f"SELECT game_id, move_uci FROM positions WHERE game_id IN ({','.join('?' * len(chunk))}) "
                "AND move_uci IS NOT NULL ORDER BY game_id, move_number", chunk)
            for gid, uci in rows:
                games[gid].moves.append(uci)
    db.close()
    return list(games.values())


def _white_score(result: str) -> int:
    return {"1-0": 1, "0-1": -1}.get(result, 0)


def encode_game(moves: Sequence[str], result: str, max_ply: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Encode every (position, move) pair of one game."""
    board = chess.Board()
    w = _white_score(result)
    states, actions, legals, wdl, plies = [], [], [], [], []
    for ply, uci in enumerate(moves):
        if max_ply is not None and ply >= max_ply:
            break
        move = chess.Move.from_uci(uci)
        states.append(encode_board(board))
        actions.append(move_to_index(move, board.turn))
        legals.append(legal_moves_and_indices(board)[1].astype(np.int16))
        score = w if board.turn == chess.WHITE else -w
        wdl.append({1: 0, 0: 1, -1: 2}[score])
        plies.append(ply)
        board.push(move)
    n = len(states)
    return {
        "states": np.stack(states) if n else np.zeros((0, 21, 8, 8), np.uint8),
        "actions": np.array(actions, np.int16),
        "legal": legals,
        "wdl": np.array(wdl, np.int8),
        "ply": np.array(plies, np.int16),
    }


def _encode_job(args):
    return encode_game(*args)


def build_dataset(games: Sequence[GameRecord], workers: int = 8, progress: bool = True) -> Dict[str, np.ndarray]:
    """Encode all games into flat arrays (see module docstring)."""
    jobs = [(g.moves, g.result) for g in games]
    if workers > 1 and len(jobs) > 50:
        with Pool(workers) as pool:
            it = pool.imap(_encode_job, jobs, chunksize=16)
            if progress:
                from tqdm.auto import tqdm
                it = tqdm(it, total=len(jobs), desc="encoding games")
            parts = list(it)
    else:
        parts = [_encode_job(j) for j in jobs]
    legal = [l for p in parts for l in p["legal"]]
    lengths = np.array([len(l) for l in legal], np.int64)
    offsets = np.zeros(len(legal) + 1, np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return {
        "states": np.concatenate([p["states"] for p in parts]),
        "actions": np.concatenate([p["actions"] for p in parts]),
        "legal_flat": np.concatenate(legal) if legal else np.zeros(0, np.int16),
        "legal_offsets": offsets,
        "wdl": np.concatenate([p["wdl"] for p in parts]),
        "ply": np.concatenate([p["ply"] for p in parts]),
        "game": np.concatenate([np.full(len(p["actions"]), i, np.int32) for i, p in enumerate(parts)]),
    }


def split_by_game(data: Dict[str, np.ndarray], val_fraction: float = 0.1, seed: int = 0):
    """Train/val position indices such that no game appears in both.

    Splitting by *position* would leak: consecutive positions of one game are
    near-duplicates, so validation accuracy would mostly measure memorisation.
    """
    n_games = int(data["game"].max()) + 1 if len(data["game"]) else 0
    rng = np.random.default_rng(seed)
    val_games = rng.random(n_games) < val_fraction
    is_val = val_games[data["game"]]
    return np.nonzero(~is_val)[0], np.nonzero(is_val)[0]


def legal_mask_batch(data: Dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    """(B, ACTION_SIZE) bool legal-move masks for positions ``idx``."""
    off = data["legal_offsets"]
    mask = np.zeros((len(idx), ACTION_SIZE), dtype=bool)
    starts, ends = off[idx], off[idx + 1]
    rows = np.repeat(np.arange(len(idx)), ends - starts)
    cols = np.concatenate([data["legal_flat"][s:e] for s, e in zip(starts, ends)]).astype(np.int64)
    mask[rows, cols] = True
    return mask


def iterate_batches(data: Dict[str, np.ndarray], idx: np.ndarray, batch_size: int = 256,
                    shuffle: bool = True, device="cpu", seed: Optional[int] = None) -> Iterator[Dict[str, torch.Tensor]]:
    """Yield dict batches: x (float planes), action, mask (legal), wdl."""
    idx = np.asarray(idx)
    if shuffle:
        idx = np.random.default_rng(seed).permutation(idx)
    for s in range(0, len(idx), batch_size):
        b = np.sort(idx[s:s + batch_size])
        yield {
            "x": planes_to_float(data["states"][b]).to(device),
            "action": torch.from_numpy(data["actions"][b].astype(np.int64)).to(device),
            "mask": torch.from_numpy(legal_mask_batch(data, b)).to(device),
            "wdl": torch.from_numpy(data["wdl"][b].astype(np.int64)).to(device),
        }


def save_arrays(path, data: Dict[str, np.ndarray]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)


def load_arrays(path) -> Dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
